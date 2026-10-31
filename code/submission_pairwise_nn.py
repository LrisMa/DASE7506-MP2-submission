"""Combine linear and nonlinear group-trained pairwise rankers."""
import numpy as np

from submission import candidate_features, select
from submission_reasoning import base_scores


def extra_scores(embeddings, model):
    x = (embeddings - model['extra_mean']) / model['extra_scale']
    scores = []
    for w1, b1, w2, b2 in zip(model['extra_w1'], model['extra_b1'],
                              model['extra_w2'], model['extra_b2']):
        scores.append((np.maximum(x @ w1 + b1, 0) @ w2 + b2).squeeze(-1))
    return np.mean(scores, axis=0)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('The pairwise neural selector requires a checkpoint')
    if not bank:
        return []
    answers, votes, text = candidate_features(bank)
    with np.load(checkpoint, allow_pickle=False) as model:
        scores = base_scores(arrays, text, votes, model)
        x = (arrays['embeddings'] - model['rank_mean']) / model['rank_scale']
        scores += float(model['rank_weight']) * (x @ model['rank_w']).squeeze(-1)
        scores += float(model['vote_extra']) * votes
        scores += float(model['extra_weight']) * extra_scores(arrays['embeddings'], model)
    return select(scores, answers)
