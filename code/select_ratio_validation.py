"""Choose the old/peer mixing weight on the validation split only.

Components come from the frozen peer-bottom checkpoint (trained on training data).
The test split is never read. Selection rule, fixed before running: maximise validation
accuracy over WEIGHTS; ties go to the smaller weight in ascending order.
"""
import json
from pathlib import Path

import numpy as np

from common import DATA, check_data, load_inputs
from submission import select
from submission_normalized import normalize_bank, normalized_candidate_features
from submission_peer import decode_bytes, group_scores, transform_text
from train_complement import hierarchical_scores, normalized_scores as old_scores, relative_scores
from train_experiments import labels_for
from train_peer import scalar_features
from sklearn.metrics.pairwise import rbf_kernel
import lightgbm as lgb

CHECKPOINT = Path('runs/peer-test/peer-bottom.npz')
OUTPUT = Path('runs/validation-ratio/ratio-selection.json')
WEIGHTS = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]


def components(bank, embeddings, model):
    """Return (old, peer, official) score matrices, mirroring submission_peer.predict_scores."""
    normalized = normalize_bank(bank)
    official, answers, votes, text = normalized_candidate_features(bank, normalized)
    valid = np.array([[value is not None for value in row] for row in official], dtype=bool)
    scalar = scalar_features(normalized, answers, votes, text)
    raw = np.concatenate([embeddings, scalar], axis=-1)
    solutions = [candidate['text'] for row in normalized for candidate in row['candidates']]
    questions = [row['question'] for row in normalized]
    vocabulary = json.loads(decode_bytes(model['solution_vocabulary']))
    question_vocabulary = json.loads(decode_bytes(model['question_vocabulary']))
    solution_tfidf = transform_text(solutions, vocabulary, model['solution_idf'])
    tail_tfidf = transform_text([t[-512:] for t in solutions], vocabulary, model['solution_idf'])
    question_tfidf = transform_text(questions, question_vocabulary, model['question_idf'])

    scaled = (raw - model['bottom_mean']) / model['bottom_scale']
    linear = (scaled @ model['bottom_linear']).reshape(-1, 4)
    tree = lgb.Booster(model_str=decode_bytes(model['bottom_tree']))
    tree_values = tree.predict(scaled.reshape(-1, scaled.shape[-1]), num_threads=4).reshape(-1, 4)
    scaled_scalar = (scalar - model['bottom_scalar_mean']) / model['bottom_scalar_scale']
    textual_bottom = (scaled_scalar.reshape(-1, 65) @ model['bottom_text_dense']
                      + solution_tfidf @ model['bottom_text_sparse']).reshape(-1, 4)
    bottom = np.stack([linear, tree_values, textual_bottom], axis=-1)
    group = group_scores(official, scalar, embeddings, bottom, model)

    kernel_input = ((raw - model['kernel_mean']) / model['kernel_scale']).reshape(-1, raw.shape[-1])
    basis = rbf_kernel(kernel_input, model['kernel_centers'], gamma=4 / 1024)
    kernel = (kernel_input @ model['kernel_linear'] + basis @ model['kernel_nonlinear']).reshape(-1, 4)

    text_input = ((raw - model['text_mean']) / model['text_scale']).reshape(-1, raw.shape[-1])
    question_projection = question_tfidf @ model['question_projection']
    candidate_projection = solution_tfidf @ model['solution_projection'] + tail_tfidf @ model['tail_projection']
    interaction = np.sum(np.repeat(question_projection, 4, axis=0) * candidate_projection, axis=1)
    textual = (text_input @ model['text_linear'] + solution_tfidf @ model['text_sparse']
               + interaction).reshape(-1, 4)

    old_model = {key[4:]: value for key, value in model.items() if key.startswith('old_')}
    old, original = old_scores(bank, embeddings, old_model)
    if original != official:
        raise ValueError('Official answer parsing differs between branches')
    peer = hierarchical_scores(group, kernel, textual, valid)['hierarchical']
    return old, peer, official, valid


def main():
    check_data(DATA, ('validation',))
    bank, arrays = load_inputs('validation')
    labels = labels_for(bank, 'validation')
    with np.load(CHECKPOINT, allow_pickle=False) as data:
        model = {key: data[key] for key in data.files}
    old, peer, official, valid = components(bank, arrays['embeddings'], model)
    rel_old = relative_scores(old, valid)
    rel_peer = relative_scores(peer, valid)
    results = {}
    for w in WEIGHTS:
        scores = w * rel_old + (1.0 - w) * rel_peer
        choice = select(scores, official)
        correct = int(sum(labels[i][c] for i, c in enumerate(choice)))
        results[str(w)] = {'old_weight': w, 'correct': correct, 'n': len(bank),
                           'accuracy': round(100.0 * correct / len(bank), 2)}
        print(f'old_weight {w:.2f}: {correct}/{len(bank)} = {100.0 * correct / len(bank):.2f}%', flush=True)
    best = max(WEIGHTS, key=lambda w: (results[str(w)]['correct'], -w))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({'rule': 'max validation accuracy, ties to smaller old_weight',
                                  'selected_old_weight': best, 'results': results}, indent=2))
    print('selected old_weight', best)


if __name__ == '__main__':
    main()
