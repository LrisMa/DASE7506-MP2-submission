"""Ensemble selector using candidate embeddings and arithmetic signals."""
import numpy as np

from reasoning_features import reasoning_features
from submission import candidate_features, select
from submission_improved import build_features, score_candidates


def base_scores(arrays, text, votes, model):
    features = build_features(arrays['embeddings'], text, votes, str(model['mode'].item()))
    neural = score_candidates(features, model)
    normalized = (arrays['embeddings'] - model['baseline_mean']) / model['baseline_scale']
    linear = (normalized @ model['baseline_w'] + model['baseline_b']).squeeze(-1)
    return linear + 0.5 * votes + float(model['blend']) * neural


def aggregate_answers(scores, answers, strength):
    result = scores.copy()
    for i, values in enumerate(answers):
        for answer in set(values):
            if answer is None:
                continue
            indices = [j for j, value in enumerate(values) if value == answer]
            group = scores[i, indices]
            result[i, indices] += strength * (group.mean() - group.max())
    return result


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('The reasoning selector requires a trained checkpoint')
    if not bank:
        return []
    answers, votes, text = candidate_features(bank)
    features = reasoning_features(bank, answers)
    with np.load(checkpoint, allow_pickle=False) as model:
        scores = base_scores(arrays, text, votes, model)
        x = (features - model['reason_mean']) / model['reason_scale']
        scores += float(model['reason_weight']) * (x @ model['reason_w']).squeeze(-1)
        scores = aggregate_answers(scores, answers, float(model['aggregate_weight']))
    return select(scores, answers)
