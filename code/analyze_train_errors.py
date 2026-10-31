"""Reproduce the seed-6813 training disagreement counts from held-out scores."""
import argparse
import json
from pathlib import Path

import numpy as np

from blend_cache import frozen_blend
from common import load_inputs
from submission import candidate_features, select
from train_experiments import labels_for
from train_improved import hard_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=Path('runs/cache/peer-components-6813.npz'))
    args = parser.parse_args()

    bank, _ = load_inputs('train')
    labels = labels_for(bank, 'train')
    answers, votes, _ = candidate_features(bank)
    valid = np.array([[answer is not None for answer in row] for row in answers], dtype=bool)
    hard = hard_mask(answers)
    with np.load(args.cache, allow_pickle=False) as archive:
        if int(archive['seed']) != 6813:
            raise ValueError('This analysis requires the seed-6813 cache')
        components = {name: archive[name] for name in ('old', 'group', 'kernel', 'text')}
    if any(scores.shape != labels.shape for scores in components.values()):
        raise ValueError('Cached score shapes do not match the training bank')

    choices = np.array(select(frozen_blend(components, valid), answers))
    rows = np.arange(len(bank))
    wrong = hard & (labels[rows, choices] == 0)
    voting_choices = select(votes, answers)
    result = {
        'seed': 6813,
        'disagreement_questions': int(hard.sum()),
        'blend_wrong': int(wrong.sum()),
        'wrong_with_one_correct_candidate': int((wrong & (labels.sum(axis=1) == 1)).sum()),
        'wrong_with_singleton_selected_answer': int((wrong & (votes[rows, choices] == 1)).sum()),
        'voting_correct_disagreement': int(labels[rows, voting_choices][hard].sum()),
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
