"""Answer selection with bounded, question-derived numerical features."""
from fractions import Fraction
import re

import numpy as np

from reasoning_features import numbers, reasoning_features
from submission import candidate_features, select
from submission_answer import group_features, group_max_scores
from submission_meta import meta_features, meta_scores, ranker_scores


SMALL = dict(zip(
    ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
     'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
     'sixteen', 'seventeen', 'eighteen', 'nineteen'),
    range(20),
))
TENS = dict(zip(('twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy',
                 'eighty', 'ninety'), range(20, 100, 10)))
DENOMINATORS = {'half': 2, 'halves': 2, 'third': 3, 'thirds': 3,
                'quarter': 4, 'quarters': 4, 'fourth': 4, 'fourths': 4,
                'fifth': 5, 'fifths': 5, 'sixth': 6, 'sixths': 6,
                'eighth': 8, 'eighths': 8, 'tenth': 10, 'tenths': 10}
TOKENS = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?|[a-z]+",
                    re.IGNORECASE)


def _cardinal(tokens, start):
    total = 0
    i = start
    last_scale = 10**12
    while i < len(tokens) and (tokens[i] in SMALL or tokens[i] in TENS):
        token = tokens[i]
        group = SMALL[token] if token in SMALL else TENS[token]
        i += 1
        if token in TENS and i < len(tokens) and tokens[i] in SMALL and SMALL[tokens[i]] < 10:
            group += SMALL[tokens[i]]
            i += 1
        if i < len(tokens) and tokens[i] == 'hundred':
            group *= 100
            i += 1
            if (i + 1 < len(tokens) and tokens[i] == 'and'
                    and (tokens[i + 1] in SMALL or tokens[i + 1] in TENS)):
                i += 1
            if i < len(tokens) and (tokens[i] in SMALL or tokens[i] in TENS):
                tail = tokens[i]
                group += SMALL[tail] if tail in SMALL else TENS[tail]
                i += 1
                if tail in TENS and i < len(tokens) and tokens[i] in SMALL and SMALL[tokens[i]] < 10:
                    group += SMALL[tokens[i]]
                    i += 1
        scales = {'thousand': 1000, 'million': 1000000}
        if i >= len(tokens) or tokens[i] not in scales or scales[tokens[i]] >= last_scale:
            return total + group, i
        last_scale = scales[tokens[i]]
        total += group * last_scale
        i += 1
        if (i + 1 < len(tokens) and tokens[i] == 'and'
                and (tokens[i + 1] in SMALL or tokens[i + 1] in TENS)):
            i += 1
    return total, i


def question_quantities(question):
    """Extract a bounded set of written and numeric quantities, in text order."""
    tokens = TOKENS.findall(question.lower())
    values = []
    written = set()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        start = i
        if token[0].isdigit() or (token[0] in '+-' and len(token) > 1):
            try:
                number = Fraction(token.rstrip('%').replace(',', ''))
            except (ValueError, ZeroDivisionError):
                i += 1
                continue
            i += 1
            if i < len(tokens) and tokens[i] in ('dozen', 'hundred', 'thousand', 'million'):
                number *= {'dozen': 12, 'hundred': 100, 'thousand': 1000,
                           'million': 1000000}[tokens[i]]
                i += 1
            values.append(number)
            if token.endswith('%'):
                values.append(number / 100)
            continue
        if token in ('twice', 'double', 'triple'):
            number = Fraction(3 if token == 'triple' else 2)
            i += 1
        elif token in DENOMINATORS:
            number = Fraction(1, DENOMINATORS[token])
            i += 1
        elif token in ('a', 'an') and i + 1 < len(tokens) and tokens[i + 1] in DENOMINATORS:
            number = Fraction(1, DENOMINATORS[tokens[i + 1]])
            i += 2
        elif token in ('a', 'an') and i + 1 < len(tokens) and tokens[i + 1] == 'dozen':
            number = Fraction(12)
            i += 2
        elif token == 'dozen':
            number = Fraction(12)
            i += 1
        elif token in SMALL or token in TENS:
            number, i = _cardinal(tokens, i)
            if i < len(tokens) and tokens[i] in DENOMINATORS:
                number = Fraction(number, DENOMINATORS[tokens[i]])
                i += 1
            else:
                number = Fraction(number)
                if (i + 2 < len(tokens) and tokens[i] == 'and'
                        and tokens[i + 1] in ('a', 'one')
                        and tokens[i + 2] in DENOMINATORS):
                    number += Fraction(1, DENOMINATORS[tokens[i + 2]])
                    i += 3
        else:
            i += 1
            continue
        if i <= start:
            raise RuntimeError('Quantity parser did not advance')
        if i < len(tokens) and tokens[i] == 'dozen':
            number *= 12
            i += 1
        if abs(number) <= 10**9 and number.denominator <= 10**6:
            values.append(number)
            written.add(number)
    return values, written


def _rounded(value):
    return round(value, 12) if abs(value) <= 10**9 else None


def _key(value):
    return round(value, 8)


def _operations(left, right):
    result = [left + right, left - right, right - left, left * right]
    if right:
        result.append(left / right)
    if left:
        result.append(right / left)
    return (rounded for value in result if (rounded := _rounded(value)) is not None)


def bounded_reachability(values):
    """Build question-only intermediate values for at most two binary operations."""
    operands = list(dict.fromkeys(float(value) for value in values if abs(value) <= 10**9))[:6]
    one = set(operands)
    for left in operands:
        for right in operands:
            one.update(_operations(left, right))
    two = set(one)
    for left in one:
        for right in operands:
            two.update(_operations(left, right))
    return operands, {_key(value) for value in one}, {_key(value) for value in two}


def _predecessors(target, operand):
    results = [target - operand, target + operand, operand - target, target * operand]
    if operand:
        results.append(target / operand)
    if target:
        results.append(operand / target)
    return (rounded for value in results if (rounded := _rounded(value)) is not None)


def numeric_features(bank, answers):
    """Produce candidate-answer evidence without reading any reference answers."""
    result = np.zeros((len(bank), 4, 12), dtype=np.float32)
    for row_index, (row, group) in enumerate(zip(bank, answers)):
        valid = {value for value in group if value is not None}
        if len(valid) < 2:
            continue
        values, written = question_quantities(row['question'])
        value_set = set(values)
        legacy = numbers(row['question'])
        operands, one, two = bounded_reachability(values)
        for answer in valid:
            number = Fraction(answer)
            if abs(number) > 10**9:
                continue
            target = _rounded(float(number))
            three = any(_key(item) in two for operand in operands
                        for item in _predecessors(target, operand))
            four = three or any(_key(previous) in two for operand in operands
                                for item in _predecessors(target, operand)
                                for other in operands for previous in _predecessors(item, other))
            distance = min((abs(target - value) for value in operands), default=0.0)
            features = [float(number in written), float(number in value_set),
                        float(_key(target) in one), np.log1p(len(value_set)),
                        float(_key(target) in two), float(three), float(four),
                        np.log1p(distance), float(bool(operands) and target <= max(operands)),
                        float(bool(operands) and target >= min(operands)),
                        np.log1p(len(written - legacy)), float(number.denominator == 1)]
            for index, value in enumerate(group):
                if value == answer:
                    result[row_index, index] = features
    return result


def relation_features(bank, answers):
    """Soft keyword-conditioned count and unit relations, not a semantic parser."""
    result = np.zeros((len(bank), 4, 6), dtype=np.float32)
    for row_index, (row, group) in enumerate(zip(bank, answers)):
        valid = {value for value in group if value is not None}
        if len(valid) < 2:
            continue
        values, _ = question_quantities(row['question'])
        operands = list(dict.fromkeys(float(value) for value in values if abs(value) <= 10**9))[:6]
        differences = {_key(abs(a - b)) for a in operands for b in operands}
        sums = {_key(a + b) for a in operands for b in operands}
        products = {_key(a * b) for a in operands for b in operands if abs(a * b) <= 10**9}
        percents = {_key(a * b / 100) for a in operands for b in operands if abs(a * b) <= 10**11}
        lower = row['question'].lower()
        pair = bool(re.search(r'\b(?:pairs?|paired|unpaired|partners?|matching)\b', lower))
        remaining = bool(re.search(r'\b(?:left|remaining|remain|fewer|difference)\b', lower))
        total = bool(re.search(r'\b(?:total|altogether|combined)\b|\bin all\b', lower))
        each = bool(re.search(r'\b(?:each|every|per)\b', lower))
        percent = bool(re.search(r'\bpercent(?:age)?\b|%', lower))
        conversions = []
        for first, second, factor in (('hour', 'minute', 60), ('day', 'hour', 24),
                                      ('week', 'day', 7), ('year', 'month', 12)):
            if first in lower and second in lower:
                conversions.extend(value * factor for value in operands if abs(value * factor) <= 10**9)
                conversions.extend(value / factor for value in operands)
        converted = {_key(value) for value in conversions}
        for answer in valid:
            number = Fraction(answer)
            if abs(number) > 10**9:
                continue
            target = _key(float(number))
            features = [float(pair and target in differences),
                        float(remaining and target in differences),
                        float(total and target in sums),
                        float(each and target in products),
                        float(target in converted),
                        float(percent and target in percents)]
            for index, value in enumerate(group):
                if value == answer:
                    result[row_index, index] = features
    return result


def feature_subset(features, variant):
    if variant == 'aligned':
        return features[..., :149]
    if variant == 'words':
        return features[..., :153]
    if variant == 'rich':
        return features
    if variant == 'number_only':
        return features[..., 50:161]
    if variant == 'number_only_base':
        return features[..., 50:149]
    if variant == 'number_only_words':
        return features[..., 50:153]
    if variant == 'number_only_relations':
        return features[..., 50:167]
    raise ValueError(f'Unknown numerical feature variant: {variant}')


def numeric_head_scores(features, model):
    x = feature_subset(features, str(model['answer_variant'].item()))
    x = (x - model['answer_mean']) / model['answer_scale']
    kind = str(model['answer_kind'].item())
    if kind == 'linear':
        return (x @ model['answer_w']).squeeze(-1)
    if kind == 'mlp':
        outputs = [(np.maximum(x @ w1 + b1, 0) @ w2 + b2).squeeze(-1)
                   for w1, b1, w2, b2 in zip(model['answer_w1'], model['answer_b1'],
                                             model['answer_w2'], model['answer_b2'])]
        return np.mean(outputs, axis=0)
    raise ValueError(f'Unknown numerical scorer: {kind}')


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('The numerical selector requires a checkpoint')
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
        numeric = numeric_features(bank, answers)
        if str(model['answer_variant'].item()) == 'number_only_relations':
            numeric = np.concatenate([numeric, relation_features(bank, answers)], axis=-1)
        features = np.concatenate([features, numeric], axis=-1)
        scores = group_max_scores(stacked, answers) + float(model['answer_weight']) * numeric_head_scores(features, model)
    return select(scores, answers)
