"""Question-relative stacking on top of a fixed candidate ranker."""
import numpy as np

from reasoning_features import reasoning_features
from submission import candidate_features, select
from submission_reasoning import base_scores


def ranker_scores(embeddings, text, votes, model):
    scores = base_scores({'embeddings': embeddings}, text, votes, model)
    normalized = (embeddings - model['rank_mean']) / model['rank_scale']
    return (scores + float(model['rank_weight']) * (normalized @ model['rank_w']).squeeze(-1)
            + float(model['vote_extra']) * votes)


def meta_features(embeddings, answers, votes, text, reasoning, scores):
    centered = scores - scores.mean(axis=1, keepdims=True)
    spread = np.maximum(scores.std(axis=1, keepdims=True), 0.1)
    standardized = centered / spread
    gap = scores.max(axis=1, keepdims=True) - scores
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    unit = embeddings / np.maximum(norms, 1e-8)
    similarities = np.einsum('nik,njk->nij', unit, unit)
    similarity_mean = (similarities.sum(axis=2) - np.diagonal(similarities, axis1=1, axis2=2)) / 3
    same = np.zeros_like(votes)
    other = np.zeros_like(votes)
    for i, row in enumerate(answers):
        for j, value in enumerate(row):
            matches = [k for k, item in enumerate(row) if k != j and value is not None and item == value]
            different = [k for k, item in enumerate(row) if k != j and item is not None and item != value]
            if matches:
                same[i, j] = similarities[i, j, matches].mean()
            if different:
                other[i, j] = similarities[i, j, different].mean()
    vote = votes / 4
    relative = np.stack([centered, standardized, gap, vote, similarity_mean, same, other,
                         same - other, centered * vote, standardized * vote,
                         vote * spread, norms.squeeze(-1)], axis=-1)
    return np.concatenate([relative, text, reasoning], axis=-1).astype(np.float32)


def meta_scores(features, model):
    x = (features - model['meta_mean']) / model['meta_scale']
    mode = str(model['meta_mode'].item())
    if mode == 'linear':
        return (x @ model['meta_w']).squeeze(-1)
    if mode == 'mlp':
        scores = []
        for w1, b1, w2, b2 in zip(model['meta_w1'], model['meta_b1'],
                                  model['meta_w2'], model['meta_b2']):
            scores.append((np.maximum(x @ w1 + b1, 0) @ w2 + b2).squeeze(-1))
        return np.mean(scores, axis=0)
    raise ValueError(f'Unknown stacking model: {mode}')


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('The stacking selector requires a trained checkpoint')
    if not bank:
        return []
    answers, votes, text = candidate_features(bank)
    reasoning = reasoning_features(bank, answers)
    with np.load(checkpoint, allow_pickle=False) as model:
        base = ranker_scores(arrays['embeddings'], text, votes, model)
        features = meta_features(arrays['embeddings'], answers, votes, text, reasoning, base)
        scores = base + float(model['meta_weight']) * meta_scores(features, model)
    return select(scores, answers)
