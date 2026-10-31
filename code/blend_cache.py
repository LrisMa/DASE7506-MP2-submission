"""Evaluate score blends from cached components without retraining.

Requires runs/cache/peer-components-<seed>.npz from cache_peer_components.py.
Training data only; the official test split is never read here.
"""
import argparse
import json

import numpy as np

from common import DATA, check_data, load_inputs
from submission import candidate_features, select
from train_complement import hierarchical_scores, relative_scores
from train_experiments import labels_for
from train_improved import hard_mask
from train_peer_diagnostics import SEEDS, matched_metrics
from cache_peer_components import cache_path


def frozen_blend(c, valid):
    """Frozen 25% baseline + 75% peer blend, using the original text ranker."""
    peer = hierarchical_scores(c['group'], c['kernel'], c['text'], valid)['hierarchical']
    return 0.25 * relative_scores(c['old'], valid) + 0.75 * relative_scores(peer, valid)


def numeric_blend(c, valid):
    """Same blend with the numeric-backoff text ranker."""
    peer = hierarchical_scores(c['group'], c['kernel'], c['text_numeric'], valid)['hierarchical']
    return 0.25 * relative_scores(c['old'], valid) + 0.75 * relative_scores(peer, valid)


# Add new candidate blends here; each receives one seed's cached arrays and valid mask.
BLENDS = {
    'frozen_25_75': frozen_blend,
    'numeric_backoff_25_75': numeric_blend,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--blends', nargs='+', default=list(BLENDS))
    args = parser.parse_args()

    check_data(DATA, ('train',))
    bank, _ = load_inputs('train')
    labels = labels_for(bank, 'train')
    official, _, _ = candidate_features(bank)
    valid = np.array([[value is not None for value in row] for row in official], dtype=bool)
    summary = {}
    for seed in args.seeds:
        with np.load(cache_path(seed), allow_pickle=False) as cached:
            c = {name: cached[name] for name in cached.files if name not in ('seed', 'folds')}
            fold_ids = cached['folds']
        folds = [np.flatnonzero(fold_ids == k) for k in range(int(fold_ids.max()) + 1)]
        baseline = select(c['old'], official)
        summary[seed] = {}
        for name in args.blends:
            choices = select(BLENDS[name](c, valid), official)
            summary[seed][name] = matched_metrics(choices, baseline, labels, official, folds)
            m = summary[seed][name]
            print(f'seed {seed} {name}: disagreement {m["disagreement_correct"]}/{m["disagreement_n"]}, '
                  f'vs baseline lost {m["baseline_only_correct"]} gained {m["variant_only_correct"]}, '
                  f'fold gains {m["fold_improvements"]}', flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
