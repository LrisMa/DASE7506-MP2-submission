"""Answer-group selector with question-conditioned numerical features."""
from fractions import Fraction

import numpy as np

from reasoning_features import numbers, reasoning_features
from submission import candidate_features, select
from submission_meta import meta_features, meta_scores, ranker_scores


QUESTION_TERMS = ('how many', 'cost', 'price', 'profit', 'money', 'dollar',
                  'percent', 'hour', 'minute', 'day', 'week', 'year',
                  'mile', 'feet', 'pound', 'left', 'remaining', 'total',
                  'average', 'each', 'per', 'more', 'less')


def question_numbers(question):
    question_values = {value for value in numbers(question)
                       if abs(value) <= 10**9 and value.denominator <= 10**6}
    operands = sorted(question_values, key=lambda value: (abs(value), value))[:8]
    lower = question.lower()
    for word, value in (('week', 7), ('dozen', 12), ('hour', 60), ('day', 24), ('percent', 100), ('%', 100)):
        if word in lower and Fraction(value) not in operands:
            operands.append(Fraction(value))
    reachable = set(operands)
    for left in operands:
        for right in operands:
            candidates = [left + right, left - right, left * right]
            if right:
                candidates.append(left / right)
            reachable.update(value for value in candidates
                             if abs(value.numerator) <= 10**12 and value.denominator <= 10**6)
    return question_values, reachable


def group_features(bank, answers, scores, reasoning):
    """Return one feature row per candidate, identical within each answer group."""
    base = np.zeros((len(bank), 4, 10), dtype=np.float32)
    regimes = np.zeros((len(bank), 4, 40), dtype=np.float32)
    numeric = np.zeros((len(bank), 4, 7 + 4 * len(QUESTION_TERMS)), dtype=np.float32)
    representatives = np.zeros((len(bank), 4), dtype=bool)
    group_base = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, (row, values) in enumerate(zip(bank, answers)):
        groups = {}
        for index, value in enumerate(values):
            if value is not None:
                groups.setdefault(value, []).append(index)
        if not groups:
            continue
        sizes = sorted((len(indices) for indices in groups.values()), reverse=True)
        regime = 0 if sizes[0] >= 3 else 1 if sizes[:2] == [2, 2] else 2 if sizes[0] == 2 else 3
        valid_scores = scores[row_index, [i for members in groups.values() for i in members]]
        center = valid_scores.mean()
        spread = max(float(valid_scores.std()), 0.1)
        question_values, reachable = question_numbers(row['question'])
        lower = row['question'].lower()
        keywords = np.array([float(term in lower) for term in QUESTION_TERMS], dtype=np.float32)
        for value, members in groups.items():
            representatives[row_index, members[0]] = True
            group_scores = scores[row_index, members]
            group_max = float(group_scores.max())
            features = np.array([
                (group_max - center) / spread,
                (float(group_scores.mean()) - center) / spread,
                float(group_scores.std()) / spread,
                len(members) / 4,
                (group_max - float(valid_scores.max())) / spread,
                float(reasoning[row_index, members, 4].mean()),
                float(reasoning[row_index, members, 5].mean()),
                float(reasoning[row_index, members, 6].mean()),
                float(reasoning[row_index, members, 7].mean()),
                float(reasoning[row_index, members, 0].mean()),
            ], dtype=np.float32)
            number = Fraction(value)
            number_float = float(number) if abs(number) < 10**12 else 0.0
            attrs = np.array([float(number.denominator == 1), float(number < 0),
                              float(number == 0), np.log1p(abs(number_float))], dtype=np.float32)
            numeric_features = np.concatenate([
                attrs, np.array([float(number in question_values), float(number in reachable),
                                 np.log1p(len(question_values))], dtype=np.float32),
                (attrs[:, None] * keywords[None, :]).reshape(-1),
            ])
            for index in members:
                base[row_index, index] = features
                regimes[row_index, index, regime * 10:(regime + 1) * 10] = features
                numeric[row_index, index] = numeric_features
                group_base[row_index, index] = group_max
    return np.concatenate([base, regimes, numeric], axis=-1), representatives, group_base


def feature_subset(features, variant):
    if variant == 'group':
        return features[..., :10]
    if variant == 'regime':
        return features[..., :50]
    if variant == 'question':
        return features
    raise ValueError(f'Unknown answer feature variant: {variant}')


def group_max_scores(scores, answers):
    result = np.zeros_like(scores)
    for row_index, values in enumerate(answers):
        for value in {item for item in values if item is not None}:
            members = [index for index, item in enumerate(values) if item == value]
            result[row_index, members] = scores[row_index, members].max()
    return result


def answer_head_scores(features, model):
    x = feature_subset(features, str(model['answer_variant'].item()))
    x = (x - model['answer_mean']) / model['answer_scale']
    kind = str(model['answer_kind'].item())
    if kind == 'linear':
        return (x @ model['answer_w']).squeeze(-1)
    if kind == 'mlp':
        values = []
        for w1, b1, w2, b2 in zip(model['answer_w1'], model['answer_b1'],
                                  model['answer_w2'], model['answer_b2']):
            values.append((np.maximum(x @ w1 + b1, 0) @ w2 + b2).squeeze(-1))
        return np.mean(values, axis=0)
    raise ValueError(f'Unknown answer scorer: {kind}')


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('The answer-group selector requires a checkpoint')
    if not bank:
        return []
    answers, votes, text = candidate_features(bank)
    reasoning = reasoning_features(bank, answers)
    embeddings = arrays['embeddings']
    with np.load(checkpoint, allow_pickle=False) as model:
        base = ranker_scores(embeddings, text, votes, model)
        old_features = meta_features(embeddings, answers, votes, text, reasoning, base)
        stacked = base + float(model['meta_weight']) * meta_scores(old_features, model)
        features, _, _ = group_features(bank, answers, base, reasoning)
        stacked_groups = group_max_scores(stacked, answers)
        scores = stacked_groups + float(model['answer_weight']) * answer_head_scores(features, model)
    return select(scores, answers)
