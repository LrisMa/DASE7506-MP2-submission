"""Select on a training-only holdout, then retrain without using test data."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, check_data, load_inputs
from submission import candidate_features, model_scores, select
from submission_improved import build_features, score_candidates
from train_experiments import Adam, evaluate, fit_pairwise, labels_for, standardize


def group_loss_gradient(logits, labels, valid):
    """Negative log probability of selecting any correct candidate."""
    masked = np.where(valid, logits, -1e9)
    max_all = np.max(masked, axis=1, keepdims=True)
    all_exp = np.where(valid, np.exp(masked - max_all), 0.0)
    all_prob = all_exp / all_exp.sum(axis=1, keepdims=True)
    positive = valid & (labels > 0)
    if not np.all(positive.any(axis=1)):
        raise ValueError('Each training question needs a parsable correct candidate')
    correct_logits = np.where(positive, logits, -1e9)
    max_correct = np.max(correct_logits, axis=1, keepdims=True)
    correct_exp = np.where(positive, np.exp(correct_logits - max_correct), 0.0)
    correct_prob = correct_exp / correct_exp.sum(axis=1, keepdims=True)
    log_all = np.log(all_exp.sum(axis=1)) + max_all[:, 0]
    log_correct = np.log(correct_exp.sum(axis=1)) + max_correct[:, 0]
    return float(np.mean(log_all - log_correct)), (all_prob - correct_prob) / len(logits)


def fit_neural(x, labels, valid, seed, epochs, batch_size, checkpoint_epochs):
    rng = np.random.default_rng(seed)
    dimension = x.shape[-1]
    w1 = (rng.standard_normal((dimension, 64)) * np.sqrt(2 / dimension)).astype(np.float32)
    b1 = np.zeros(64, dtype=np.float32)
    w2 = (rng.standard_normal((64, 1)) * np.sqrt(2 / 64)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)
    optimizer = Adam([w1, b1, w2, b2], learning_rate=0.0005)
    snapshots = {}
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x))
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            xb = x[ids]
            hidden = np.maximum(xb @ w1 + b1, 0)
            logits = (hidden @ w2 + b2).squeeze(-1)
            _, grad_logits = group_loss_gradient(logits, labels[ids], valid[ids])
            error = grad_logits[..., None]
            hidden_grad = (error @ w2.T) * (hidden > 0)
            flat_x = xb.reshape(-1, dimension)
            flat_hidden = hidden.reshape(-1, 64)
            flat_hidden_grad = hidden_grad.reshape(-1, 64)
            gradients = [flat_x.T @ flat_hidden_grad + 0.0001 * w1,
                         flat_hidden_grad.sum(axis=0),
                         flat_hidden.T @ error.reshape(-1, 1) + 0.0001 * w2,
                         error.sum(axis=(0, 1))]
            optimizer.update(gradients)
        if epoch in checkpoint_epochs:
            snapshots[epoch] = {key: value.copy() for key, value in
                                zip(('w1', 'b1', 'w2', 'b2'), (w1, b1, w2, b2))}
    return snapshots


def combine_models(snapshots, epoch, mode, mean, scale):
    return {'mode': np.array(mode), 'mean': mean, 'scale': scale,
            **{key: np.stack([model[epoch][key] for model in snapshots])
               for key in ('w1', 'b1', 'w2', 'b2')}}


def hard_mask(answers):
    return np.array([len({value for value in row if value is not None}) > 1
                     for row in answers], dtype=bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('runs/improved'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    check_data(DATA, ('train',))
    bank, arrays = load_inputs('train')
    labels = labels_for(bank, 'train')
    answers, votes, text = candidate_features(bank)
    valid = np.array([[value is not None for value in row] for row in answers], dtype=bool)
    hard = hard_mask(answers)
    permutation = np.random.default_rng(2026).permutation(len(bank))
    heldout = permutation[:len(bank) // 5]
    fitting = permutation[len(bank) // 5:]
    fitting_hard = fitting[hard[fitting]]
    heldout_answers = [answers[i] for i in heldout]
    epochs_to_check = (6, 12, 18, 24)
    seeds = (123, 456)
    records = []

    x, _, mean, scale = standardize(arrays['embeddings'][fitting], arrays['embeddings'][heldout])
    weights = fit_pairwise(x, labels[fitting], 123, 36, 512)
    baseline = {'kind': np.array('pairwise'), 'mean': mean, 'scale': scale, **weights}
    old_scores = model_scores(arrays['embeddings'][heldout], text[heldout], baseline)
    old_choices = select(old_scores + 0.5 * votes[heldout], heldout_answers)
    old_metrics = evaluate(old_choices, labels[heldout], heldout_answers)
    records.append({'name': 'previous_pairwise', **old_metrics})
    print(json.dumps(records[-1]), flush=True)

    best = (old_metrics['disagreement_correct'], None, None, None)
    for mode in ('embedding', 'context'):
        features = build_features(arrays['embeddings'], text, votes, mode)
        _, _, mean, scale = standardize(features[fitting], features[heldout])
        normalized_train = (features[fitting_hard] - mean) / scale
        models = [fit_neural(normalized_train, labels[fitting_hard], valid[fitting_hard],
                             seed, max(epochs_to_check), 128, epochs_to_check)
                  for seed in seeds]
        for epoch in epochs_to_check:
            model = combine_models(models, epoch, mode, mean, scale)
            scores = score_candidates(features[heldout], model)
            for blend in (0.25, 0.5, 1.0, 2.0):
                choices = select(old_scores + 0.5 * votes[heldout] + blend * scores, heldout_answers)
                metrics = evaluate(choices, labels[heldout], heldout_answers)
                record = {'name': mode, 'epochs': epoch, 'blend': blend, **metrics}
                records.append(record)
                print(json.dumps(record), flush=True)
                if metrics['disagreement_correct'] > best[0]:
                    best = (metrics['disagreement_correct'], mode, epoch, blend)
        del features, normalized_train, models

    _, mode, epoch, blend = best
    if mode is None:
        (args.output_dir / 'selected.npz').unlink(missing_ok=True)
        selected = {'name': 'previous_pairwise', 'reason': 'no new model improved the holdout'}
    else:
        features = build_features(arrays['embeddings'], text, votes, mode)
        normalized, _, mean, scale = standardize(features, features[:1])
        full_models = [fit_neural(normalized[hard], labels[hard], valid[hard], seed, epoch, 128, (epoch,))
                       for seed in seeds]
        final = combine_models(full_models, epoch, mode, mean, scale)
        baseline_x, _, baseline_mean, baseline_scale = standardize(arrays['embeddings'], arrays['embeddings'][:1])
        baseline_weights = fit_pairwise(baseline_x, labels, 123, 36, 512)
        final.update(blend=np.float32(blend), baseline_mean=baseline_mean,
                     baseline_scale=baseline_scale, baseline_w=baseline_weights['w'],
                     baseline_b=baseline_weights['b'])
        np.savez(args.output_dir / 'selected.npz', **final)
        selected = {'name': 'ensemble', 'mode': mode, 'epochs': epoch, 'blend': blend}
    summary = {'selection_data': '20% question-level training holdout, seed 2026',
               'selection_metric': 'accuracy among holdout questions with at least two distinct parsable answers',
               'selection_ties': 'first mode and epoch',
               'model_seeds': list(seeds), 'train_hard_questions': int(hard.sum()),
               'fitting_hard_questions': int(len(fitting_hard)),
               'heldout_questions': len(heldout),
               'selected': selected, 'records': records,
               'elapsed_seconds': time.perf_counter() - start}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('SELECTED ' + json.dumps(summary['selected']), flush=True)


if __name__ == '__main__':
    main()
