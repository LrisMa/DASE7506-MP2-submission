"""Offline candidate selection using voting or a trained NumPy model."""
import numpy as np

from common import parse_answer


def candidate_features(bank):
    """Return parsed answers, vote counts, and simple text features."""
    answers = []
    votes = np.zeros((len(bank), 4), dtype=np.float32)
    text_features = np.zeros((len(bank), 4, 5), dtype=np.float32)
    for row_index, row in enumerate(bank):
        parsed = [parse_answer(candidate['text']) for candidate in row['candidates']]
        answers.append(parsed)
        for index, candidate in enumerate(row['candidates']):
            text = candidate['text']
            value = parsed[index]
            votes[row_index, index] = parsed.count(value) if value is not None else 0
            text_features[row_index, index] = [
                np.log1p(len(text)),
                np.log1p(text.count('\n')),
                np.log1p(text.count('=')),
                np.log1p(sum(character.isdigit() for character in text)),
                float(value is not None),
            ]
    return answers, votes, text_features


def select(scores, answers):
    """Select the highest-scoring parsable candidate, or 0 if none parse."""
    valid = np.array([[value is not None for value in row] for row in answers], dtype=bool)
    masked = np.where(valid, scores, -np.inf)
    return [int(index) for index in np.argmax(masked, axis=1)]


def model_scores(embeddings, text_features, model):
    kind = str(model['kind'].item())
    if kind == 'text':
        features = text_features
    elif kind in ('linear', 'pairwise', 'mlp'):
        features = embeddings
    else:
        raise ValueError(f'Unknown model kind: {kind}')
    x = (features - model['mean']) / model['scale']
    if kind == 'mlp':
        hidden = np.maximum(x @ model['w1'] + model['b1'], 0)
        return (hidden @ model['w2'] + model['b2']).squeeze(-1)
    return (x @ model['w'] + model['b']).squeeze(-1)


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order."""
    if not bank:
        return []
    answers, votes, text_features = candidate_features(bank)
    if checkpoint is None:
        return select(votes, answers)
    with np.load(checkpoint, allow_pickle=False) as model:
        scores = model_scores(arrays['embeddings'], text_features, model)
        scores = scores + float(model['vote_weight']) * votes
    return select(scores, answers)
