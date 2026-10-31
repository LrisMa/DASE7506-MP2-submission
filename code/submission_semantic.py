"""Question quantities and bounded evidence for selecting an existing answer."""
from fractions import Fraction
import re
from typing import NamedTuple

import numpy as np

from reasoning_features import reasoning_features
from submission import candidate_features, select
from submission_answer import group_features, group_max_scores
from submission_meta import meta_features, meta_scores, ranker_scores
from submission_numeric import DENOMINATORS, SMALL, TENS, numeric_features, numeric_head_scores


class Quantity(NamedTuple):
    value: Fraction
    start: int
    end: int
    written: bool


TOKEN = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)"
    r"(?:\.\d+)?(?:e[+-]?\d+)?(?:/\d+(?:\.\d+)?)?(?:st|nd|rd|th|am|pm)?%?(?!\w)|[a-z]+",
    re.IGNORECASE,
)
SCALES = {'thousand': 1_000, 'million': 1_000_000, 'billion': 1_000_000_000}
ORDINAL_UNITS = dict(zip(('first', 'second', 'third', 'fourth', 'fifth', 'sixth',
                          'seventh', 'eighth', 'ninth'), range(1, 10)))
FRACTIONS = {
    **DENOMINATORS,
    'seventh': 7, 'sevenths': 7, 'ninth': 9, 'ninths': 9,
    'eleventh': 11, 'elevenths': 11, 'twelfth': 12, 'twelfths': 12,
    'twentieth': 20, 'twentieths': 20, 'hundredth': 100, 'hundredths': 100,
    'thousandth': 1000, 'thousandths': 1000,
}


def _connected(text, tokens, left, right, hyphen=True):
    if right >= len(tokens) or right != left + 1:
        return False
    gap = text[tokens[left].end():tokens[right].start()]
    return bool(gap) and (gap.isspace() or (hyphen and gap == '-'))


def _under_hundred(text, tokens, i):
    if i >= len(tokens):
        return None
    word = tokens[i].group().lower()
    if word in SMALL:
        return SMALL[word], i + 1
    if word not in TENS:
        return None
    value = TENS[word]
    end = i + 1
    if (_connected(text, tokens, i, end) and tokens[end].group().lower() in SMALL
            and 0 < SMALL[tokens[end].group().lower()] < 10):
        value += SMALL[tokens[end].group().lower()]
        end += 1
    elif _connected(text, tokens, i, end) and tokens[end].group().lower() in ORDINAL_UNITS:
        value += ORDINAL_UNITS[tokens[end].group().lower()]
        end += 1
    return value, end


def _cardinal(text, tokens, start):
    total = 0
    i = start
    last_scale = 10**12
    while i < len(tokens):
        word = tokens[i].group().lower()
        if (word in ('a', 'an') and i + 1 < len(tokens)
                and tokens[i + 1].group().lower() in ('hundred', *SCALES)
                and _connected(text, tokens, i, i + 1)):
            group, i = 1, i + 1
        else:
            parsed = _under_hundred(text, tokens, i)
            if parsed is None:
                break
            group, i = parsed
        if (i < len(tokens) and tokens[i].group().lower() == 'hundred'
                and _connected(text, tokens, i - 1, i)):
            group *= 100
            i += 1
            if (i + 1 < len(tokens) and tokens[i].group().lower() == 'and'
                    and _connected(text, tokens, i - 1, i)
                    and _connected(text, tokens, i, i + 1)
                    and _under_hundred(text, tokens, i + 1) is not None):
                i += 1
            if i < len(tokens) and _connected(text, tokens, i - 1, i):
                tail = _under_hundred(text, tokens, i)
                if tail is not None:
                    value, i = tail
                    group += value
        if (i >= len(tokens) or tokens[i].group().lower() not in SCALES
                or not _connected(text, tokens, i - 1, i)
                or SCALES[tokens[i].group().lower()] >= last_scale):
            return total + group, i
        last_scale = SCALES[tokens[i].group().lower()]
        total += group * last_scale
        i += 1
        if (i + 1 < len(tokens) and tokens[i].group().lower() == 'and'
                and _connected(text, tokens, i - 1, i)
                and _connected(text, tokens, i, i + 1)
                and _under_hundred(text, tokens, i + 1) is not None):
            i += 1
        if i >= len(tokens) or not _connected(text, tokens, i - 1, i):
            break
    return (total, i) if i > start else None


def _fraction_words(text, tokens, start):
    if start >= len(tokens):
        return None
    word = tokens[start].group().lower()
    if word in ('half', 'quarter'):
        return Fraction(1, FRACTIONS[word]), start + 1
    if (word in ('a', 'an') and start + 1 < len(tokens)
            and tokens[start + 1].group().lower() in FRACTIONS
            and _connected(text, tokens, start, start + 1)):
        return Fraction(1, FRACTIONS[tokens[start + 1].group().lower()]), start + 2
    cardinal = _cardinal(text, tokens, start)
    if cardinal is None:
        return None
    numerator, end = cardinal
    if (end < len(tokens) and tokens[end].group().lower() in FRACTIONS
            and _connected(text, tokens, end - 1, end)):
        return Fraction(numerator, FRACTIONS[tokens[end].group().lower()]), end + 1
    return None


def _decimal_part(text, tokens, point):
    cursor = point + 1
    digits = []
    while cursor < len(tokens) and _connected(text, tokens, cursor - 1, cursor):
        digit = tokens[cursor].group().lower()
        if digit in SMALL and SMALL[digit] < 10:
            digits.append(str(SMALL[digit]))
        elif digit.isdigit():
            digits.append(digit)
        else:
            break
        cursor += 1
    if not digits:
        return None
    decimal = ''.join(digits)
    return Fraction(int(decimal), 10**len(decimal)), cursor


def _number_at(text, tokens, start):
    word = tokens[start].group().lower()
    if word[0].isdigit() or word[0] in '+-.' and len(word) > 1:
        try:
            numeric = re.sub(r'(?:st|nd|rd|th|am|pm)$', '', word.rstrip('%'))
            value, end, written = Fraction(numeric.replace(',', '')), start + 1, False
        except (ValueError, ZeroDivisionError):
            return None
    elif word in ('twice', 'double', 'triple'):
        value, end, written = Fraction(3 if word == 'triple' else 2), start + 1, True
    elif word == 'point':
        decimal = _decimal_part(text, tokens, start)
        if decimal is None:
            return None
        value, end = decimal
        written = True
    elif (word in ('a', 'an') and start + 2 < len(tokens)
          and tokens[start + 1].group().lower() == 'couple'
          and tokens[start + 2].group().lower() == 'of'
          and _connected(text, tokens, start, start + 1)
          and _connected(text, tokens, start + 1, start + 2)):
        return Fraction(2), start + 2, True
    else:
        written = True
        fraction = _fraction_words(text, tokens, start)
        if fraction is not None:
            value, end = fraction
        else:
            cardinal = _cardinal(text, tokens, start)
            if cardinal is None:
                if word == 'dozen':
                    return Fraction(12), start + 1, True
                if (word in ('a', 'an') and start + 1 < len(tokens)
                        and tokens[start + 1].group().lower() == 'dozen'
                        and _connected(text, tokens, start, start + 1)):
                    return Fraction(12), start + 2, True
                return None
            value, end = cardinal
            value = Fraction(value)
            if (end + 1 < len(tokens) and tokens[end].group().lower() == 'point'
                    and _connected(text, tokens, end - 1, end)
                    and _connected(text, tokens, end, end + 1)):
                decimal = _decimal_part(text, tokens, end)
                if decimal is not None:
                    part, end = decimal
                    value += part
    if value.denominator == 1 and end < len(tokens) and _connected(text, tokens, end - 1, end):
        fraction_start = end
        if (tokens[end].group().lower() == 'and' and end + 1 < len(tokens)
                and _connected(text, tokens, end, end + 1)):
            fraction_start = end + 1
        mixed = _fraction_words(text, tokens, fraction_start) if fraction_start != end else None
        if mixed is None and '/' in tokens[fraction_start].group():
            try:
                mixed = (Fraction(tokens[fraction_start].group().replace(',', '')), fraction_start + 1)
            except (ValueError, ZeroDivisionError):
                mixed = None
        if mixed is not None and 0 < mixed[0] < 1:
            fraction_value, end = mixed
            value += fraction_value if value >= 0 else -fraction_value
            written = written or fraction_start != start + 1
    if (0 < abs(value) < 1 and end + 2 < len(tokens)
            and tokens[end].group().lower() == 'of'
            and tokens[end + 1].group().lower() in ('a', 'an')
            and tokens[end + 2].group().lower() == 'dozen'
            and _connected(text, tokens, end - 1, end)
            and _connected(text, tokens, end, end + 1)
            and _connected(text, tokens, end + 1, end + 2)):
        value *= 12
        end += 3
        written = True
    elif (end < len(tokens) and tokens[end].group().lower() in ('dozen', *SCALES, 'hundred')
            and _connected(text, tokens, end - 1, end)):
        multiplier = {'dozen': 12, 'hundred': 100, **SCALES}[tokens[end].group().lower()]
        value *= multiplier
        end += 1
        written = True
    elif (end + 1 < len(tokens) and tokens[end].group().lower() in ('a', 'an')
          and tokens[end + 1].group().lower() == 'dozen'
          and _connected(text, tokens, end - 1, end)
          and _connected(text, tokens, end, end + 1)):
        value *= 12
        end += 2
        written = True
    return value, end, written


def parse_quantities(text):
    """Return exact numeric mentions with source spans; punctuation ends word phrases."""
    tokens = list(TOKEN.finditer(text))
    result = []
    i = 0
    while i < len(tokens):
        start = i
        word = tokens[i].group().lower()
        sign = 1
        if (word in ('negative', 'minus', 'positive', 'plus') and i + 1 < len(tokens)
                and _connected(text, tokens, i, i + 1)):
            sign = -1 if word in ('negative', 'minus') else 1
            i += 1
        parsed = _number_at(text, tokens, i)
        if parsed is None:
            i = start + 1
            continue
        value, end, written = parsed
        value *= sign
        percent = tokens[end - 1].group().endswith('%')
        if (end < len(tokens) and tokens[end].group().lower() in ('percent', 'percentage')
                and _connected(text, tokens, end - 1, end)):
            percent = True
            end += 1
        elif (end + 1 < len(tokens) and tokens[end].group().lower() == 'per'
              and tokens[end + 1].group().lower() == 'cent'
              and _connected(text, tokens, end - 1, end)
              and _connected(text, tokens, end, end + 1)):
            percent = True
            end += 2
        result.append(Quantity(value, tokens[start].start(), tokens[end - 1].end(), written))
        if percent:
            result.append(Quantity(value / 100, tokens[start].start(), tokens[end - 1].end(), written))
        i = end
    return result


OPERATION_CUES = {
    'add': re.compile(r'\b(?:total|altogether|together|combined|sum|plus|add|added|more|increased)\b|\bin all\b', re.I),
    'sub': re.compile(r'\b(?:left|remain|remaining|difference|fewer|less|minus|spent|lost|gave|used|removed)\b', re.I),
    'mul': re.compile(r'\b(?:each|every|per|times|twice|double|triple|product|of)\b', re.I),
    'div': re.compile(r'\b(?:each|per|divide|divided|half|third|quarter|percent|ratio|shared|split|average)\b|%', re.I),
}
UNIT_WORDS = re.compile(r"\s*([a-z]+)", re.I)
UNIT_STOP = {'a', 'an', 'and', 'as', 'at', 'by', 'for', 'from', 'in', 'is',
             'of', 'on', 'or', 'per', 'than', 'the', 'times', 'to', 'with'}


def _bounded(value):
    return abs(value.numerator) <= 10**12 and value.denominator <= 10**6 and abs(value) <= 10**9


def _operations(left, right):
    values = [('add', left + right), ('sub', left - right), ('sub', right - left),
              ('mul', left * right)]
    if right:
        values.append(('div', left / right))
    if left:
        values.append(('div', right / left))
    return [(kind, value) for kind, value in values if _bounded(value)]


def _unit_after(question, mention):
    match = UNIT_WORDS.match(question[mention.end:])
    if not match:
        return ''
    word = match.group(1).lower()
    if word in UNIT_STOP:
        return ''
    return word[:-1] if word.endswith('s') and len(word) > 3 else word


def quantity_evidence(bank, answers, max_mentions=5):
    """Build bounded, question-only features with distinct source mentions."""
    if max_mentions not in (5, 8):
        raise ValueError('Unsupported quantity evidence limit')
    result = np.zeros((len(bank), 4, 10), dtype=np.float32)
    for row_index, (row, group) in enumerate(zip(bank, answers)):
        valid = {value for value in group if value is not None}
        if len(valid) < 2:
            continue
        question = row['question']
        mentions = parse_quantities(question)
        values = {mention.value for mention in mentions}
        written = {mention.value for mention in mentions if mention.written}
        by_span = {}
        for mention in mentions:
            by_span.setdefault((mention.start, mention.end), []).append(mention)
        sources = list(by_span.values())[:max_mentions]
        allowed = {kind for kind, pattern in OPERATION_CUES.items() if pattern.search(question)}
        pair, pair_allowed, triple, triple_allowed, same_unit = set(), set(), set(), set(), set()
        for left_index, left in enumerate(sources):
            for right_index in range(left_index + 1, len(sources)):
                right = sources[right_index]
                unit = _unit_after(question, left[0])
                same = bool(unit and unit == _unit_after(question, right[0]))
                for a in left:
                    for b in right:
                        for kind, value in _operations(a.value, b.value):
                            pair.add(value)
                            if kind in allowed:
                                pair_allowed.add(value)
                            if same and kind in allowed and kind in ('add', 'sub'):
                                same_unit.add(value)
                            for third_index, third in enumerate(sources):
                                if third_index in (left_index, right_index):
                                    continue
                                for c in third:
                                    for next_kind, next_value in _operations(value, c.value):
                                        triple.add(next_value)
                                        if kind in allowed and next_kind in allowed:
                                            triple_allowed.add(next_value)
        lower = question.lower()
        converted = set()
        for first, second, factor in (('hour', 'minute', 60), ('day', 'hour', 24),
                                      ('week', 'day', 7), ('year', 'month', 12)):
            if first in lower and second in lower:
                for value in values:
                    for candidate in (value * factor, value / factor):
                        if _bounded(candidate):
                            converted.add(candidate)
        for answer in valid:
            target = Fraction(answer)
            if abs(target) > 10**9:
                continue
            features = [float(target in values), float(target in written),
                        np.log1p(len(values)), np.log1p(len(written)),
                        float(target in pair), float(target in pair_allowed),
                        float(target in triple), float(target in triple_allowed),
                        float(target in same_unit), float(target in converted)]
            for index, value in enumerate(group):
                if value == answer:
                    result[row_index, index] = features
    return result


def gate_features(scores, answers):
    """Compute answer-group-relative stack scores and vote context."""
    output = np.zeros((len(answers), 4, 5), dtype=np.float32)
    groupscores = group_max_scores(scores, answers)
    for row_index, group in enumerate(answers):
        groups = {}
        for index, value in enumerate(group):
            if value is not None:
                groups.setdefault(value, []).append(index)
        if not groups:
            continue
        group_values = np.array([groupscores[row_index, indices[0]] for indices in groups.values()])
        center = float(group_values.mean())
        spread = max(float(group_values.std()), 0.1)
        best = float(group_values.max())
        for indices in groups.values():
            score = float(groupscores[row_index, indices[0]])
            features = [(score - center) / spread, (best - score) / spread,
                        len(indices) / 4, len(groups) / 4, min(spread, 10.0)]
            output[row_index, indices] = features
    return output


def feature_subset(features, variant):
    if variant == 'parser':
        return features[..., 50:165]
    if variant == 'parsed':
        return features[..., 50:171]
    if variant == 'conditional':
        return features[..., 50:176]
    if variant == 'correction_parsed':
        return features[..., 161:171]
    if variant == 'correction_conditional':
        return features[..., 161:176]
    raise ValueError(f'Unknown semantic feature variant: {variant}')


def semantic_head_scores(features, model):
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
    raise ValueError(f'Unknown semantic scorer: {kind}')


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('The semantic selector requires a checkpoint')
    if not bank:
        return []
    answers, votes, text = candidate_features(bank)
    reasoning = reasoning_features(bank, answers)
    embeddings = arrays['embeddings']
    with np.load(checkpoint, allow_pickle=False) as model:
        base = ranker_scores(embeddings, text, votes, model)
        old_features = meta_features(embeddings, answers, votes, text, reasoning, base)
        stacked = base + float(model['meta_weight']) * meta_scores(old_features, model)
        group, _, _ = group_features(bank, answers, base, reasoning)
        limit = int(model['quantity_limit']) if 'quantity_limit' in model else 5
        old = np.concatenate([group, numeric_features(bank, answers)], axis=-1)
        if 'correction_weight' in model:
            scores = group_max_scores(stacked, answers) + float(model['answer_weight']) * numeric_head_scores(old, model)
            correction = {key[len('correction_'):]: model[key] for key in model.files
                          if key.startswith('correction_answer_')}
            parts = [old, quantity_evidence(bank, answers, limit)]
            variant = str(correction['answer_variant'].item())
            if variant == 'correction_conditional':
                parts.append(gate_features(scores, answers))
            elif variant != 'correction_parsed':
                raise ValueError(f'Unknown correction feature variant: {variant}')
            features = np.concatenate(parts, axis=-1)
            scores += float(model['correction_weight']) * semantic_head_scores(features, correction)
        else:
            features = np.concatenate([old, quantity_evidence(bank, answers, limit),
                                       gate_features(stacked, answers)], axis=-1)
            scores = group_max_scores(stacked, answers) + float(model['answer_weight']) * semantic_head_scores(features, model)
    return select(scores, answers)
