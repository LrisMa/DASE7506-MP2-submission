"""Cross-validate an all-pairs group-weighted ranker on training data only."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, check_data, load_inputs
from submission import candidate_features, select
from submission_reasoning import base_scores
from train_experiments import Adam, evaluate, labels_for, sigmoid, standardize
from train_improved import hard_mask
from train_reasoning import fit_base


def all_pairs_gradient(scores, labels, valid):
    positive = labels > 0
    negative = valid & ~positive
    pairs = positive[:, :, None] & negative[:, None, :]
    pair_count = pairs.sum(axis=(1, 2), keepdims=True)
    if np.any(pair_count == 0):
        raise ValueError('Each ranker training group must contain a valid positive and negative')
    difference = scores[:, :, None] - scores[:, None, :]
    derivative = (sigmoid(difference) - 1) * pairs / pair_count / len(scores)
    return derivative.sum(axis=2) - derivative.sum(axis=1)


def fit_all_pairs(features, labels, valid, seed, epochs, batch_size, l2, snapshots):
    rng = np.random.default_rng(seed)
    w = np.zeros((features.shape[-1], 1), dtype=np.float32)
    optimizer = Adam([w], learning_rate=0.001)
    saved = {}
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(features))
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            batch = features[ids]
            scores = (batch @ w).squeeze(-1)
            grad = all_pairs_gradient(scores, labels[ids], valid[ids])
            weight_gradient = batch.reshape(-1, batch.shape[-1]).T @ grad.reshape(-1, 1)
            optimizer.update([weight_gradient + l2 * w])
        if epoch in snapshots:
            saved[epoch] = w.copy()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('runs/ranker_cv'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    check_data(DATA, ('train',))
    bank, arrays = load_inputs('train')
    labels = labels_for(bank, 'train')
    answers, votes, text = candidate_features(bank)
    valid = np.array([[value is not None for value in row] for row in answers], dtype=bool)
    hard = hard_mask(answers)
    folds = np.array_split(np.random.default_rng(932).permutation(len(bank)), 3)
    configs = [(l2, epoch) for l2 in (0.001, 0.01, 0.1) for epoch in (20, 40, 80)]
    oof_base = np.zeros((len(bank), 4), dtype=np.float32)
    oof_rank = {config: np.zeros((len(bank), 4), dtype=np.float32) for config in configs}
    for index, held in enumerate(folds):
        fitting = np.concatenate([fold for i, fold in enumerate(folds) if i != index])
        model = fit_base(arrays['embeddings'][fitting], text[fitting], votes[fitting],
                         labels[fitting], valid[fitting], hard[fitting])
        oof_base[held] = base_scores({'embeddings': arrays['embeddings'][held]},
                                     text[held], votes[held], model)
        x, x_held, _, _ = standardize(arrays['embeddings'][fitting], arrays['embeddings'][held])
        for l2 in (0.001, 0.01, 0.1):
            snapshots = fit_all_pairs(x[hard[fitting]], labels[fitting][hard[fitting]],
                                      valid[fitting][hard[fitting]], 123, 80, 128, l2, (20, 40, 80))
            for epoch in (20, 40, 80):
                oof_rank[(l2, epoch)][held] = (x_held @ snapshots[epoch]).squeeze(-1)
        print(f'Completed training-only fold {index + 1}/3', flush=True)

    baseline = evaluate(select(oof_base, answers), labels, answers)
    records = [{'name': 'previous_ensemble', **baseline}]
    best = (baseline['disagreement_correct'], None)
    for (l2, epoch), scores in oof_rank.items():
        for rank_weight in (0.25, 0.5, 1.0, 2.0):
            for vote_extra in (-0.5, 0.0, 0.5):
                choices = select(oof_base + rank_weight * scores + vote_extra * votes, answers)
                result = evaluate(choices, labels, answers)
                config = {'l2': l2, 'epochs': epoch, 'rank_weight': rank_weight,
                          'vote_extra': vote_extra}
                records.append({**config, **result})
                if result['disagreement_correct'] > best[0]:
                    best = (result['disagreement_correct'], config)
    config = best[1]
    fold_results = []
    selected_scores = oof_base if config is None else (
        oof_base + config['rank_weight'] * oof_rank[(config['l2'], config['epochs'])]
        + config['vote_extra'] * votes)
    for fold in folds:
        fold_answers = [answers[i] for i in fold]
        old_result = evaluate(select(oof_base[fold], fold_answers), labels[fold], fold_answers)
        new_result = evaluate(select(selected_scores[fold], fold_answers), labels[fold], fold_answers)
        fold_results.append({'disagreement_n': old_result['disagreement_n'],
                             'previous_correct': old_result['disagreement_correct'],
                             'selected_correct': new_result['disagreement_correct']})
    if config is None:
        (args.output_dir / 'selected.npz').unlink(missing_ok=True)
        selected = {'name': 'previous_ensemble', 'reason': 'no improvement in training cross-validation'}
    else:
        model = fit_base(arrays['embeddings'], text, votes, labels, valid, hard)
        x, _, mean, scale = standardize(arrays['embeddings'], arrays['embeddings'][:1])
        weights = fit_all_pairs(x[hard], labels[hard], valid[hard], 123, config['epochs'],
                                128, config['l2'], (config['epochs'],))
        model.update(rank_mean=mean, rank_scale=scale, rank_w=weights[config['epochs']],
                     rank_weight=np.float32(config['rank_weight']),
                     vote_extra=np.float32(config['vote_extra']))
        np.savez(args.output_dir / 'selected.npz', **model)
        selected = {'name': 'all_pairs_ensemble', **config}
    summary = {'selection_data': '3-fold question-level training cross-validation, seed 932',
               'selection_metric': 'accuracy on training questions with distinct parsable answers',
               'selection_ties': 'prefer previous model, then first grid entry',
               'selected': selected, 'fold_results': fold_results, 'records': records,
               'elapsed_seconds': time.perf_counter() - start}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('BASELINE ' + json.dumps(records[0]), flush=True)
    print('SELECTED ' + json.dumps(selected), flush=True)
    if config is not None:
        print('BEST ' + json.dumps(next(record for record in records
                                       if all(record.get(k) == v for k, v in config.items()))), flush=True)


if __name__ == '__main__':
    main()
