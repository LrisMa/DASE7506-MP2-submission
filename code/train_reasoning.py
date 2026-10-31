"""Select reasoning features using question-level training cross-validation only."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, check_data, load_inputs
from reasoning_features import reasoning_features
from submission import candidate_features, select
from submission_improved import build_features
from submission_reasoning import aggregate_answers, base_scores
from train_experiments import evaluate, fit_pairwise, labels_for, standardize
from train_improved import combine_models, fit_neural, hard_mask


def fit_base(embeddings, text, votes, labels, valid, hard):
    x, _, mean, scale = standardize(embeddings, embeddings[:1])
    weights = fit_pairwise(x, labels, 123, 36, 512)
    features = build_features(embeddings, text, votes, 'embedding')
    features, _, neural_mean, neural_scale = standardize(features, features[:1])
    snapshots = [fit_neural(features[hard], labels[hard], valid[hard], seed, 12, 128, (12,))
                 for seed in (123, 456)]
    model = combine_models(snapshots, 12, 'embedding', neural_mean, neural_scale)
    model.update(blend=np.float32(0.25), baseline_mean=mean, baseline_scale=scale,
                 baseline_w=weights['w'], baseline_b=weights['b'])
    return model


def scale_reason(train, other):
    flat = train.reshape(-1, train.shape[-1])
    mean = flat.mean(axis=0)
    scale = np.maximum(flat.std(axis=0), np.float32(0.25))
    return (train - mean) / scale, (other - mean) / scale, mean, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('runs/reasoning'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    check_data(DATA, ('train',))
    bank, arrays = load_inputs('train')
    labels = labels_for(bank, 'train')
    answers, votes, text = candidate_features(bank)
    valid = np.array([[value is not None for value in row] for row in answers], dtype=bool)
    hard = hard_mask(answers)
    reasoning = reasoning_features(bank, answers)
    n = len(bank)
    oof_base = np.zeros((n, 4), dtype=np.float32)
    oof_reason = np.zeros((n, 4), dtype=np.float32)
    permutation = np.random.default_rng(932).permutation(n)
    folds = np.array_split(permutation, 3)
    for index, held in enumerate(folds):
        fitting = np.concatenate([fold for i, fold in enumerate(folds) if i != index])
        model = fit_base(arrays['embeddings'][fitting], text[fitting], votes[fitting],
                         labels[fitting], valid[fitting], hard[fitting])
        oof_base[held] = base_scores({'embeddings': arrays['embeddings'][held]},
                                     text[held], votes[held], model)
        x, x_held, _, _ = scale_reason(reasoning[fitting], reasoning[held])
        reason = fit_pairwise(x, labels[fitting], 123, 72, 512)
        oof_reason[held] = (x_held @ reason['w']).squeeze(-1)
        print(f'Completed training-only fold {index + 1}/3', flush=True)
    records = []
    baseline = evaluate(select(oof_base, answers), labels, answers)
    records.append({'reason_weight': 0.0, 'aggregate_weight': 0.0, **baseline})
    best = (baseline['disagreement_correct'], 0.0, 0.0)
    for reason_weight in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        for aggregate_weight in (0.0, 0.5, 1.0):
            if reason_weight == aggregate_weight == 0:
                continue
            scores = aggregate_answers(oof_base + reason_weight * oof_reason, answers, aggregate_weight)
            result = evaluate(select(scores, answers), labels, answers)
            records.append({'reason_weight': reason_weight, 'aggregate_weight': aggregate_weight, **result})
            if result['disagreement_correct'] > best[0]:
                best = (result['disagreement_correct'], reason_weight, aggregate_weight)
    _, reason_weight, aggregate_weight = best
    model = fit_base(arrays['embeddings'], text, votes, labels, valid, hard)
    train_reason, _, reason_mean, reason_scale = scale_reason(reasoning, reasoning[:1])
    weights = fit_pairwise(train_reason, labels, 123, 72, 512)
    model.update(reason_mean=reason_mean, reason_scale=reason_scale, reason_w=weights['w'],
                 reason_weight=np.float32(reason_weight), aggregate_weight=np.float32(aggregate_weight))
    np.savez(args.output_dir / 'selected.npz', **model)
    summary = {'selection_data': '3-fold question-level training cross-validation, seed 932',
               'selection_metric': 'accuracy on training questions with distinct parsable answers',
               'selection_ties': 'prefer previous model, then first grid entry',
               'selected': {'reason_weight': reason_weight, 'aggregate_weight': aggregate_weight},
               'records': records, 'elapsed_seconds': time.perf_counter() - start}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('BASELINE ' + json.dumps(records[0]), flush=True)
    print('SELECTED ' + json.dumps(summary['selected']), flush=True)
    print('BEST ' + json.dumps(next(record for record in records
                                   if record['reason_weight'] == reason_weight
                                   and record['aggregate_weight'] == aggregate_weight)), flush=True)


if __name__ == '__main__':
    main()
