"""Reweight the frozen peer-inspired ensemble without changing its original entrypoint."""
import numpy as np

from submission import select
from submission_peer import predict_scores as previous_scores
from train_complement import normalized_scores, relative_scores


def predict_scores(bank, embeddings, model):
    previous, official = previous_scores(bank, embeddings, model)
    old_model = {key[4:]: value for key, value in model.items() if key.startswith('old_')}
    old, original = normalized_scores(bank, embeddings, old_model)
    if original != official:
        raise ValueError('Official answer parsing differs between ensemble branches')
    valid = np.array([[value is not None for value in row] for row in official], dtype=bool)
    baseline = relative_scores(old, valid)
    # The frozen previous score is 0.75 baseline + 0.25 peer.
    return baseline + 3.0 * (previous - baseline), official


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('The peer-inspired selector requires a checkpoint')
    if not bank:
        return []
    with np.load(checkpoint, allow_pickle=False) as checkpoint_data:
        model = {key: checkpoint_data[key] for key in checkpoint_data.files}
    scores, official = predict_scores(bank, arrays['embeddings'], model)
    return select(scores, official)
