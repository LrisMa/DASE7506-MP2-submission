"""Shared inference for fixed-weight comparisons of the existing ensemble."""
import numpy as np

from submission import select
from submission_peer import predict_scores as previous_scores
from train_complement import normalized_scores, relative_scores


def predict_scores(bank, embeddings, model, old_weight):
    previous, official = previous_scores(bank, embeddings, model)
    old_model = {key[4:]: value for key, value in model.items() if key.startswith('old_')}
    old, original = normalized_scores(bank, embeddings, old_model)
    if original != official:
        raise ValueError('Official answer parsing differs between ensemble branches')
    valid = np.array([[value is not None for value in row] for row in official], dtype=bool)
    baseline = relative_scores(old, valid)
    # The existing peer predictor is 0.75 baseline + 0.25 peer.
    return baseline + (4.0 * (1.0 - old_weight)) * (previous - baseline), official


def predict_fixed(bank, arrays, checkpoint, old_weight):
    if checkpoint is None:
        raise ValueError('The fixed-ratio selector requires a checkpoint')
    if not bank:
        return []
    with np.load(checkpoint, allow_pickle=False) as data:
        model = {key: data[key] for key in data.files}
    scores, official = predict_scores(bank, arrays['embeddings'], model, old_weight)
    return select(scores, official)
