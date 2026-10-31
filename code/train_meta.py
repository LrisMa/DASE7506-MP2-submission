"""Nested question-level training CV for relative-score stacking."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, check_data, load_inputs
from reasoning_features import reasoning_features
from submission import candidate_features, select
from submission_meta import meta_features, meta_scores, ranker_scores
from train_experiments import evaluate, labels_for
from train_improved import hard_mask
from train_pairwise_nn import fit_existing, fit_neural_pairs
from train_ranker_cv import fit_all_pairs


def make_meta_rows(indices, scores, embeddings, answers, votes, text, reasoning):
    rows = [answers[i] for i in indices]
    return meta_features(embeddings[indices], rows, votes[indices], text[indices],
                         reasoning[indices], scores)


def train_base(indices, embeddings, text, votes, labels, valid, hard):
    model, _, _, _ = fit_existing(embeddings[indices], text[indices], votes[indices],
                                  labels[indices], valid[indices], hard[indices])
    return model


def predict_base(model, indices, embeddings, text, votes):
    return ranker_scores(embeddings[indices], text[indices], votes[indices], model)


def base_oof(indices, folds, embeddings, text, votes, labels, valid, hard):
    result = np.zeros((len(indices), 4), dtype=np.float32)
    for local_held in folds:
        mask = np.ones(len(indices), dtype=bool)
        mask[local_held] = False
        fitting, held = indices[mask], indices[local_held]
        model = train_base(fitting, embeddings, text, votes, labels, valid, hard)
        result[local_held] = predict_base(model, held, embeddings, text, votes)
    return result


def scale_meta(features):
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0)
    scale = np.maximum(flat.std(axis=0), np.float32(0.1))
    return (features - mean) / scale, mean, scale


def fit_meta(features, labels, valid, hard, mode, epochs):
    x, mean, scale = scale_meta(features)
    bundle = {'meta_mode': np.array(mode), 'meta_mean': mean, 'meta_scale': scale}
    if mode == 'linear':
        bundle['meta_w'] = fit_all_pairs(x[hard], labels[hard], valid[hard], 781,
                                         epochs, 128, 0.01, (epochs,))[epochs]
    elif mode == 'mlp':
        snapshots = [fit_neural_pairs(x[hard], labels[hard], valid[hard], seed, epochs, 128, 0.001, (epochs,))
                     for seed in (781, 782)]
        for key in ('w1', 'b1', 'w2', 'b2'):
            bundle['meta_' + key] = np.stack([model[epochs][key] for model in snapshots])
    else:
        raise ValueError(f'Unknown meta mode: {mode}')
    return bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('runs/meta'))
    args = parser.parse_args()
    if any((args.output_dir / name).exists() for name in ('selected.npz', 'summary.json')):
        raise FileExistsError(f'Experiment output already exists in {args.output_dir}; choose another directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    check_data(DATA, ('train',))
    bank, arrays = load_inputs('train')
    embeddings = arrays['embeddings']
    labels = labels_for(bank, 'train')
    answers, votes, text = candidate_features(bank)
    reasoning = reasoning_features(bank, answers)
    valid = np.array([[value is not None for value in row] for row in answers], dtype=bool)
    hard = hard_mask(answers)
    n = len(bank)
    outer_folds = np.array_split(np.random.default_rng(1849).permutation(n), 3)
    configs = [('linear', 40), ('linear', 80), ('mlp', 12), ('mlp', 24)]
    oof_base = np.zeros((n, 4), dtype=np.float32)
    oof_meta = {config: np.zeros((n, 4), dtype=np.float32) for config in configs}

    for fold_id, held in enumerate(outer_folds):
        fitting = np.concatenate([fold for i, fold in enumerate(outer_folds) if i != fold_id])
        inner = np.array_split(np.random.default_rng(600 + fold_id).permutation(len(fitting)), 2)
        inner_scores = base_oof(fitting, inner, embeddings, text, votes, labels, valid, hard)
        meta_train = make_meta_rows(fitting, inner_scores, embeddings, answers, votes, text, reasoning)
        base = train_base(fitting, embeddings, text, votes, labels, valid, hard)
        outer_scores = predict_base(base, held, embeddings, text, votes)
        oof_base[held] = outer_scores
        meta_held = make_meta_rows(held, outer_scores, embeddings, answers, votes, text, reasoning)
        for mode, epochs in configs:
            gate = fit_meta(meta_train, labels[fitting], valid[fitting], hard[fitting], mode, epochs)
            oof_meta[(mode, epochs)][held] = meta_scores(meta_held, gate)
        print(f'Completed nested training-only fold {fold_id + 1}/3', flush=True)

    baseline = evaluate(select(oof_base, answers), labels, answers)
    records = [{'name': 'base_ranker', **baseline}]
    baseline_folds = []
    for fold in outer_folds:
        subset = [answers[i] for i in fold]
        baseline_folds.append(evaluate(select(oof_base[fold], subset), labels[fold], subset)['disagreement_correct'])
    best = (baseline['disagreement_correct'], None)
    for (mode, epochs), scores in oof_meta.items():
        for weight in (0.25, 0.5, 1.0, 2.0):
            combined = oof_base + weight * scores
            result = evaluate(select(combined, answers), labels, answers)
            fold_wins = 0
            for fold_id, fold in enumerate(outer_folds):
                subset = [answers[i] for i in fold]
                metric = evaluate(select(combined[fold], subset), labels[fold], subset)
                fold_wins += metric['disagreement_correct'] > baseline_folds[fold_id]
            config = {'mode': mode, 'epochs': epochs, 'meta_weight': weight}
            records.append({**config, 'folds_improved': fold_wins, **result})
            if fold_wins >= 2 and result['disagreement_correct'] >= baseline['disagreement_correct'] + 10 and result['disagreement_correct'] > best[0]:
                best = (result['disagreement_correct'], config)
    config = best[1]
    if config is None:
        selected = {'name': 'base_ranker', 'reason': 'no sufficient nested-CV improvement'}
    else:
        indices = np.arange(n)
        full_folds = np.array_split(np.random.default_rng(781).permutation(n), 3)
        base_training_scores = base_oof(indices, full_folds, embeddings, text, votes, labels, valid, hard)
        full_features = make_meta_rows(indices, base_training_scores, embeddings, answers, votes, text, reasoning)
        gate = fit_meta(full_features, labels, valid, hard, config['mode'], config['epochs'])
        base = train_base(indices, embeddings, text, votes, labels, valid, hard)
        base.update(gate)
        base['meta_weight'] = np.float32(config['meta_weight'])
        np.savez(args.output_dir / 'selected.npz', **base)
        selected = {'name': 'stacked_ranker', **config}
    summary = {'selection_data': 'nested 3-fold/2-fold question-level training CV; outer seed 1849',
               'selection_metric': 'accuracy on training questions with distinct parsable answers',
               'selection_rule': 'at least 10 more correct and at least two folds improve; ties prefer baseline',
               'selected': selected, 'records': records, 'elapsed_seconds': time.perf_counter() - started}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('BASELINE ' + json.dumps(records[0]), flush=True)
    print('SELECTED ' + json.dumps(selected), flush=True)
    if config:
        print('BEST ' + json.dumps(next(record for record in records
                                       if all(record.get(k) == v for k, v in config.items()))), flush=True)


if __name__ == '__main__':
    main()
