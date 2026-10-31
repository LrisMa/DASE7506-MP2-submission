"""Train lightweight candidate scorers and select using validation only."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, check_data, load_inputs, parse_answer, read_jsonl
from submission import candidate_features, model_scores, select


def labels_for(bank, split):
    if split not in ('train', 'validation'):
        raise ValueError('Training and development labels are limited to train and validation')
    references = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    if len(references) != len(bank):
        raise ValueError('Reference count differs from bank')
    labels = np.zeros((len(bank), 4), dtype=np.float32)
    for i, (row, reference) in enumerate(zip(bank, references)):
        if row['id'] != reference['id'] or row['question'] != reference['question']:
            raise ValueError('Reference and bank are not aligned')
        target = parse_answer(reference['answer'])
        if target is None:
            raise ValueError(f'Unparseable reference in {split}')
        for j, candidate in enumerate(row['candidates']):
            answer = parse_answer(candidate['text'])
            labels[i, j] = float(answer is not None and answer == target)
    return labels


def standardize(train_features, val_features):
    mean = train_features.reshape(-1, train_features.shape[-1]).mean(axis=0)
    scale = train_features.reshape(-1, train_features.shape[-1]).std(axis=0)
    scale = np.maximum(scale, np.float32(1e-6))
    return (train_features - mean) / scale, (val_features - mean) / scale, mean, scale


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class Adam:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.m = [np.zeros_like(p) for p in parameters]
        self.v = [np.zeros_like(p) for p in parameters]
        self.step = 0

    def update(self, gradients):
        self.step += 1
        for p, grad, m, v in zip(self.parameters, gradients, self.m, self.v):
            m *= 0.9
            m += 0.1 * grad
            v *= 0.999
            v += 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** self.step)
            v_hat = v / (1 - 0.999 ** self.step)
            p -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)


def fit_binary(features, labels, kind, seed, epochs, batch_size):
    rng = np.random.default_rng(seed)
    x = features.reshape(-1, features.shape[-1])
    y = labels.reshape(-1, 1)
    dimension = x.shape[1]
    if kind == 'mlp':
        w1 = (rng.standard_normal((dimension, 64)) * np.sqrt(2 / dimension)).astype(np.float32)
        b1 = np.zeros(64, dtype=np.float32)
        w2 = (rng.standard_normal((64, 1)) * np.sqrt(2 / 64)).astype(np.float32)
        b2 = np.zeros(1, dtype=np.float32)
        params = [w1, b1, w2, b2]
    else:
        w = np.zeros((dimension, 1), dtype=np.float32)
        b = np.zeros(1, dtype=np.float32)
        params = [w, b]
    optimizer = Adam(params, learning_rate=0.001)
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            ids = order[start:start + batch_size]
            xb, yb = x[ids], y[ids]
            if kind == 'mlp':
                hidden = np.maximum(xb @ w1 + b1, 0)
                error = (sigmoid(hidden @ w2 + b2) - yb) / len(ids)
                hidden_grad = (error @ w2.T) * (hidden > 0)
                gradients = [xb.T @ hidden_grad + 0.0001 * w1,
                             hidden_grad.sum(axis=0),
                             hidden.T @ error + 0.0001 * w2,
                             error.sum(axis=0)]
            else:
                error = (sigmoid(xb @ w + b) - yb) / len(ids)
                gradients = [xb.T @ error + 0.0001 * w, error.sum(axis=0)]
            optimizer.update(gradients)
    if kind == 'mlp':
        return dict(w1=w1, b1=b1, w2=w2, b2=b2)
    return dict(w=w, b=b)


def fit_pairwise(features, labels, seed, epochs, batch_size):
    rng = np.random.default_rng(seed)
    mixed = np.flatnonzero((labels.sum(axis=1) > 0) & (labels.sum(axis=1) < 4))
    if not len(mixed):
        raise ValueError('Training data contains no mixed-correctness groups')
    w = np.zeros((features.shape[-1], 1), dtype=np.float32)
    b = np.zeros(1, dtype=np.float32)
    optimizer = Adam([w], learning_rate=0.001)
    for _ in range(epochs):
        order = rng.permutation(mixed)
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            positive = np.array([rng.choice(np.flatnonzero(labels[i] == 1)) for i in rows])
            negative = np.array([rng.choice(np.flatnonzero(labels[i] == 0)) for i in rows])
            difference = features[rows, positive] - features[rows, negative]
            error = (sigmoid(difference @ w) - 1.0) / len(rows)
            optimizer.update([difference.T @ error + 0.0001 * w])
    return dict(w=w, b=b)


def evaluate(choices, labels, answers):
    indices = np.asarray(choices, dtype=np.int64)
    correct = labels[np.arange(len(labels)), indices].astype(bool)
    disagreement = np.array([len({value for value in row if value is not None}) > 1
                             for row in answers])
    return {'correct': int(correct.sum()), 'n': len(labels),
            'accuracy': float(correct.mean() * 100),
            'disagreement_correct': int(correct[disagreement].sum()),
            'disagreement_n': int(disagreement.sum()),
            'disagreement_accuracy': float(correct[disagreement].mean() * 100)
            if disagreement.any() else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('runs/experiments'))
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    check_data(DATA, ('train', 'validation'))
    train_bank, train_arrays = load_inputs('train')
    val_bank, val_arrays = load_inputs('validation')
    train_labels = labels_for(train_bank, 'train')
    val_labels = labels_for(val_bank, 'validation')
    train_answers, train_votes, train_text = candidate_features(train_bank)
    val_answers, val_votes, val_text = candidate_features(val_bank)
    del train_answers, train_votes
    records = []
    first = [0] * len(val_bank)
    vote = select(val_votes, val_answers)
    records.append({'name': 'first', **evaluate(first, val_labels, val_answers)})
    records.append({'name': 'vote', **evaluate(vote, val_labels, val_answers)})
    best = None
    for kind in ('text', 'linear', 'mlp', 'pairwise'):
        start = time.perf_counter()
        train_feature = train_text if kind == 'text' else train_arrays['embeddings']
        val_feature = val_text if kind == 'text' else val_arrays['embeddings']
        x_train, _, mean, scale = standardize(train_feature, val_feature)
        if kind == 'pairwise':
            weights = fit_pairwise(x_train, train_labels, args.seed, args.epochs * 3, args.batch_size)
        else:
            weights = fit_binary(x_train, train_labels, kind, args.seed, args.epochs, args.batch_size)
        model = dict(kind=np.array(kind), mean=mean, scale=scale, **weights)
        scores = model_scores(val_arrays['embeddings'], val_text, model)
        training_seconds = time.perf_counter() - start
        for vote_weight in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
            choices = select(scores + vote_weight * val_votes, val_answers)
            result = evaluate(choices, val_labels, val_answers)
            record = {'name': kind, 'vote_weight': vote_weight,
                      'training_seconds': training_seconds, **result}
            records.append(record)
            print(json.dumps(record), flush=True)
            if best is None or result['correct'] > best[0]:
                best = (result['correct'], kind, vote_weight, model)
        np.savez(args.output_dir / f'{kind}.npz', vote_weight=np.float32(0), **model)
    best_correct, best_kind, best_vote_weight, best_model = best
    if best_correct > records[1]['correct']:
        np.savez(args.output_dir / 'selected.npz',
                 vote_weight=np.float32(best_vote_weight), **best_model)
        selected = {'name': best_kind, 'vote_weight': best_vote_weight,
                    'checkpoint': 'selected.npz', 'correct': best_correct}
    else:
        selected = {'name': 'vote', 'checkpoint': None, 'correct': records[1]['correct']}
    summary = {'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch_size,
               'selection_rule': 'highest validation accuracy; ties prefer voting, then first model/grid entry',
               'selected': selected, 'records': records,
               'elapsed_seconds': time.perf_counter() - started}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('SELECTED ' + json.dumps(selected), flush=True)


if __name__ == '__main__':
    main()
