"""Candidate selector trained on groups with conflicting numeric answers."""
import numpy as np

from submission import candidate_features, select


def build_features(embeddings, text_features, votes, mode):
    features = [embeddings]
    if mode == 'context':
        features.extend([embeddings - embeddings.mean(axis=1, keepdims=True),
                         text_features, votes[..., None]])
    elif mode != 'embedding':
        raise ValueError(f'Unknown feature mode: {mode}')
    return np.concatenate(features, axis=-1)


def score_candidates(features, model):
    x = (features - model['mean']) / model['scale']
    scores = []
    for w1, b1, w2, b2 in zip(model['w1'], model['b1'], model['w2'], model['b2']):
        hidden = np.maximum(x @ w1 + b1, 0)
        scores.append((hidden @ w2 + b2).squeeze(-1))
    return np.mean(scores, axis=0)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('The improved selector requires a trained checkpoint')
    if not bank:
        return []
    answers, votes, text_features = candidate_features(bank)
    with np.load(checkpoint, allow_pickle=False) as model:
        features = build_features(arrays['embeddings'], text_features, votes,
                                  str(model['mode'].item()))
        neural_scores = score_candidates(features, model)
        normalized = (arrays['embeddings'] - model['baseline_mean']) / model['baseline_scale']
        baseline_scores = (normalized @ model['baseline_w'] + model['baseline_b']).squeeze(-1)
        scores = baseline_scores + 0.5 * votes + float(model['blend']) * neural_scores
    return select(scores, answers)
