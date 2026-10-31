"""Cross-validate nonlinear group pairwise scoring on training data only."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, check_data, load_inputs
from submission import candidate_features, select
from submission_pairwise_nn import extra_scores
from submission_reasoning import base_scores
from train_experiments import Adam, evaluate, labels_for, standardize
from train_improved import hard_mask
from train_ranker_cv import all_pairs_gradient, fit_all_pairs
from train_reasoning import fit_base


def fit_neural_pairs(x, labels, valid, seed, epochs, batch_size, l2, snapshots):
    rng = np.random.default_rng(seed)
    dimension = x.shape[-1]
    w1 = (rng.standard_normal((dimension, 64)) * np.sqrt(2 / dimension)).astype(np.float32)
    b1 = np.zeros(64, dtype=np.float32)
    w2 = (rng.standard_normal((64, 1)) * np.sqrt(2 / 64)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)
    optimizer = Adam([w1, b1, w2, b2], learning_rate=0.0005)
    saved = {}
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x))
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            batch = x[ids]
            hidden = np.maximum(batch @ w1 + b1, 0)
            scores = (hidden @ w2 + b2).squeeze(-1)
            grad = all_pairs_gradient(scores, labels[ids], valid[ids])[..., None]
            hidden_grad = (grad @ w2.T) * (hidden > 0)
            gradients = [batch.reshape(-1, dimension).T @ hidden_grad.reshape(-1, 64) + l2 * w1,
                         hidden_grad.sum(axis=(0, 1)),
                         hidden.reshape(-1, 64).T @ grad.reshape(-1, 1) + l2 * w2,
                         grad.sum(axis=(0, 1))]
            optimizer.update(gradients)
        if epoch in snapshots:
            saved[epoch] = {key: value.copy() for key, value in
                            zip(('w1', 'b1', 'w2', 'b2'), (w1, b1, w2, b2))}
    return saved


def neural_bundle(snapshots, epoch, mean, scale):
    result = {'extra_mean': mean, 'extra_scale': scale}
    for key in ('w1', 'b1', 'w2', 'b2'):
        result['extra_' + key] = np.stack([snapshot[epoch][key] for snapshot in snapshots])
    return result


def fit_existing(embeddings, text, votes, labels, valid, hard):
    model = fit_base(embeddings, text, votes, labels, valid, hard)
    x, _, mean, scale = standardize(embeddings, embeddings[:1])
    snapshots = fit_all_pairs(x[hard], labels[hard], valid[hard], 123, 80, 128, 0.001, (80,))
    model.update(rank_mean=mean, rank_scale=scale, rank_w=snapshots[80],
                 rank_weight=np.float32(2), vote_extra=np.float32(0))
    return model, x, mean, scale


def existing_scores(embeddings, text, votes, model):
    scores = base_scores({'embeddings': embeddings}, text, votes, model)
    x = (embeddings - model['rank_mean']) / model['rank_scale']
    return scores + float(model['rank_weight']) * (x @ model['rank_w']).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('runs/pairwise_nn'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    check_data(DATA, ('train',))
    bank, arrays = load_inputs('train')
    labels = labels_for(bank, 'train')
    answers, votes, text = candidate_features(bank)
    valid = np.array([[value is not None for value in row] for row in answers], dtype=bool)
    hard = hard_mask(answers)
    folds = np.array_split(np.random.default_rng(932).permutation(len(bank)), 3)
    configs = [(l2, epoch) for l2 in (0.0001, 0.001) for epoch in (8, 16, 24)]
    oof_baseline = np.zeros((len(bank), 4), dtype=np.float32)
    oof_extra = {config: np.zeros((len(bank), 4), dtype=np.float32) for config in configs}
    for fold_id, held in enumerate(folds):
        fitting = np.concatenate([fold for i, fold in enumerate(folds) if i != fold_id])
        model, x, mean, scale = fit_existing(arrays['embeddings'][fitting], text[fitting],
                                             votes[fitting], labels[fitting], valid[fitting], hard[fitting])
        oof_baseline[held] = existing_scores(arrays['embeddings'][held], text[held], votes[held], model)
        for l2 in (0.0001, 0.001):
            models = [fit_neural_pairs(x[hard[fitting]], labels[fitting][hard[fitting]],
                                       valid[fitting][hard[fitting]], seed, 24, 128, l2, (8, 16, 24))
                      for seed in (157, 379)]
            for epoch in (8, 16, 24):
                bundle = neural_bundle(models, epoch, mean, scale)
                oof_extra[(l2, epoch)][held] = extra_scores(arrays['embeddings'][held], bundle)
        print(f'Completed training-only fold {fold_id + 1}/3', flush=True)

    baseline = evaluate(select(oof_baseline, answers), labels, answers)
    records = [{'name': 'linear_pairwise_ensemble', **baseline}]
    best = (baseline['disagreement_correct'], None)
    fold_baseline = []
    for fold in folds:
        fold_answers = [answers[i] for i in fold]
        fold_baseline.append(evaluate(select(oof_baseline[fold], fold_answers),
                                      labels[fold], fold_answers)['disagreement_correct'])
    for (l2, epoch), scores in oof_extra.items():
        for weight in (0.25, 0.5, 1.0, 2.0):
            combined = oof_baseline + weight * scores
            result = evaluate(select(combined, answers), labels, answers)
            fold_improvements = 0
            for i, fold in enumerate(folds):
                fold_answers = [answers[j] for j in fold]
                metric = evaluate(select(combined[fold], fold_answers), labels[fold], fold_answers)
                fold_improvements += metric['disagreement_correct'] > fold_baseline[i]
            config = {'l2': l2, 'epochs': epoch, 'extra_weight': weight}
            records.append({**config, 'folds_improved': fold_improvements, **result})
            if fold_improvements >= 2 and result['disagreement_correct'] > best[0]:
                best = (result['disagreement_correct'], config)
    config = best[1]
    if config is None:
        (args.output_dir / 'selected.npz').unlink(missing_ok=True)
        selected = {'name': 'linear_pairwise_ensemble', 'reason': 'no robust training CV improvement'}
    else:
        model, x, mean, scale = fit_existing(arrays['embeddings'], text, votes, labels, valid, hard)
        snapshots = [fit_neural_pairs(x[hard], labels[hard], valid[hard], seed, config['epochs'],
                                      128, config['l2'], (config['epochs'],))
                     for seed in (157, 379)]
        model.update(neural_bundle(snapshots, config['epochs'], mean, scale))
        model['extra_weight'] = np.float32(config['extra_weight'])
        np.savez(args.output_dir / 'selected.npz', **model)
        selected = {'name': 'nonlinear_pairwise_ensemble', **config}
    summary = {'selection_data': '3-fold question-level training cross-validation, seed 932',
               'selection_metric': 'accuracy on training questions with distinct parsable answers',
               'selection_rule': 'improve at least two folds; highest total, ties prefer baseline/first',
               'selected': selected, 'records': records, 'elapsed_seconds': time.perf_counter() - started}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('BASELINE ' + json.dumps(records[0]), flush=True)
    print('SELECTED ' + json.dumps(selected), flush=True)
    if config is not None:
        print('BEST ' + json.dumps(next(record for record in records
                                       if all(record.get(k) == v for k, v in config.items()))), flush=True)


if __name__ == '__main__':
    main()
