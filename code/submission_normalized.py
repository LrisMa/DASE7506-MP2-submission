"""Numeric-text normalization for the existing numerical selector."""
import re

import numpy as np

from reasoning_features import reasoning_features
from submission import candidate_features, select
from submission_answer import group_features, group_max_scores
from submission_meta import meta_features, meta_scores, ranker_scores
from submission_numeric import numeric_features, numeric_head_scores, relation_features
from submission_semantic import parse_quantities


PERCENT_END = re.compile(r'(?:%|\bpercent|\bpercentage|\bper\s+cent)$', re.IGNORECASE)
COMPOUND_ORDINAL = re.compile(
    r'\b(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)'
    r'[-\s]+(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)$',
    re.IGNORECASE,
)


def normalize_numeric_text(text):
    """Replace supported written quantities with exact numeric forms."""
    replacements = {}
    seen = set()
    previous_end = None
    for quantity in parse_quantities(text):
        span = (quantity.start, quantity.end)
        if span in seen:
            continue
        seen.add(span)
        gap = text[previous_end:quantity.start] if previous_end is not None else ''
        previous_end = quantity.end
        if not quantity.written:
            continue
        source = text[quantity.start:quantity.end]
        if source.lower() in ('twice', 'double', 'triple'):
            continue
        value = quantity.value
        rendered = str(value)
        if PERCENT_END.search(source):
            rendered += '%'
        elif value.denominator == 1 and COMPOUND_ORDINAL.search(source):
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(value.numerator % 10, 'th')
            rendered += suffix
        if gap.strip() == '/' or ((gap.isspace() or gap == '-') and value.denominator != 1):
            rendered = f'({rendered})'
        replacements[span] = rendered
    parts = []
    end = 0
    for (start, stop), replacement in sorted(replacements.items()):
        if start < end:
            raise ValueError('Overlapping quantity spans cannot be normalized')
        parts.extend((text[end:start], replacement))
        end = stop
    parts.append(text[end:])
    return ''.join(parts)


def normalize_bank(bank):
    """Return a normalized copy; leave the official input texts untouched."""
    return [{**row, 'question': normalize_numeric_text(row['question']),
             'candidates': [{**candidate, 'text': normalize_numeric_text(candidate['text'])}
                            for candidate in row['candidates']]}
            for row in bank]


def normalized_candidate_features(bank, normalized_bank):
    """Group equivalent internal answers while respecting official scoring."""
    if len(bank) != len(normalized_bank):
        raise ValueError('Original and normalized banks have different lengths')
    official, _, _ = candidate_features(bank)
    normalized, _, text_features = candidate_features(normalized_bank)
    answers = [[original if original is not None else converted
                for original, converted in zip(raw, changed)]
               for raw, changed in zip(official, normalized)]
    votes = np.array([[row.count(value) if value is not None else 0 for value in row]
                      for row in answers], dtype=np.float32).reshape(len(bank), 4)
    text_features[..., 4] = np.array([[value is not None for value in row]
                                      for row in answers], dtype=np.float32).reshape(len(bank), 4)
    return official, answers, votes, text_features


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('The normalized numerical selector requires a checkpoint')
    if not bank:
        return []
    normalized = normalize_bank(bank)
    official, answers, votes, text = normalized_candidate_features(bank, normalized)
    reasoning = reasoning_features(normalized, answers)
    embeddings = arrays['embeddings']
    with np.load(checkpoint, allow_pickle=False) as model:
        base = ranker_scores(embeddings, text, votes, model)
        features = meta_features(embeddings, answers, votes, text, reasoning, base)
        stacked = base + float(model['meta_weight']) * meta_scores(features, model)
        group, _, _ = group_features(normalized, answers, base, reasoning)
        numeric = numeric_features(normalized, answers)
        if str(model['answer_variant'].item()) == 'number_only_relations':
            numeric = np.concatenate([numeric, relation_features(normalized, answers)], axis=-1)
        features = np.concatenate([group, numeric], axis=-1)
        scores = group_max_scores(stacked, answers) + float(model['answer_weight']) * numeric_head_scores(features, model)
    return select(scores, official)
