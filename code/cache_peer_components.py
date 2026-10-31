"""Train each component once per outer fold and cache held-out candidate scores.

Blends are then evaluated by blend_cache.py without retraining. Training data only.
"""
import argparse
from pathlib import Path
import tempfile
import time

import numpy as np

from common import DATA, check_data, load_inputs
from reasoning_features import reasoning_features
from submission import candidate_features, select
from submission_answer import group_max_scores
from submission_normalized import normalize_bank, normalized_candidate_features, predict as predict_normalized
from submission_numeric import feature_subset as numeric_subset, numeric_features
from train_answer import STACK_EPOCHS, STACK_MODE, STACK_WEIGHT, feature_rows, fit_answer, stacked_oof
from train_complement import normalized_scores
from train_experiments import labels_for
from train_improved import hard_mask
from train_meta import fit_meta, make_meta_rows, train_base
from train_numeric_backoff import text_fold_numeric
from train_peer import kernel_fold, scalar_features, text_fold
from train_peer_diagnostics import SEEDS, group_fold_with_diagnostics

CACHE_DIR = Path('runs/cache')
COMPONENTS = ('old', 'group', 'kernel', 'text', 'text_numeric')


def cache_path(seed):
    return CACHE_DIR / f'peer-components-{seed}.npz'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = parser.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        if cache_path(seed).exists():
            raise FileExistsError(f'{cache_path(seed)} already exists; delete it to rebuild')

    started = time.perf_counter()
    check_data(DATA, ('train',))
    bank, arrays = load_inputs('train')
    embeddings = arrays['embeddings']
    labels = labels_for(bank, 'train')
    official, votes, text = candidate_features(bank)
    reasoning = reasoning_features(bank, official)
    valid = np.array([[value is not None for value in row] for row in official], dtype=bool)
    hard = hard_mask(official)
    numeric = numeric_features(bank, official)
    normalized = normalize_bank(bank)
    _, answers, normalized_votes, normalized_text = normalized_candidate_features(bank, normalized)
    scalar = scalar_features(normalized, answers, normalized_votes, normalized_text)
    n = len(bank)

    for seed in args.seeds:
        folds = np.array_split(np.random.default_rng(seed).permutation(n), 3)
        components = {name: np.zeros((n, 4), dtype=np.float32) for name in COMPONENTS}
        fold_ids = np.full(n, -1, dtype=np.int64)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'fold.npz'
            for fold_id, held in enumerate(folds):
                fitting = np.concatenate([fold for i, fold in enumerate(folds) if i != fold_id])
                inner = np.array_split(np.random.default_rng(6800 + fold_id).permutation(len(fitting)), 2)
                fit_base, fit_stack = stacked_oof(fitting, inner, embeddings, text, votes, labels,
                                                  valid, hard, official, reasoning)
                group, reps, _ = feature_rows(fitting, fit_base, bank, official, reasoning)
                head = fit_answer(np.concatenate([group, numeric[fitting]], axis=-1),
                                  group_max_scores(fit_stack, [official[i] for i in fitting]), reps,
                                  labels[fitting], hard[fitting], 'number_only', 'mlp',
                                  selector=numeric_subset, residual_weight=0.5)
                meta_train = make_meta_rows(fitting, fit_base, embeddings, official, votes, text, reasoning)
                meta = fit_meta(meta_train, labels[fitting], valid[fitting], hard[fitting],
                                STACK_MODE, STACK_EPOCHS)
                base = train_base(fitting, embeddings, text, votes, labels, valid, hard)
                np.savez(checkpoint, **base, **meta, **head,
                         meta_weight=np.float32(STACK_WEIGHT), answer_weight=np.float32(0.5))

                held_bank = [bank[i] for i in held]
                with np.load(checkpoint, allow_pickle=False) as fold_model:
                    old, held_official = normalized_scores(held_bank, embeddings[held], fold_model)
                if select(old, held_official) != predict_normalized(
                        held_bank, {'embeddings': embeddings[held]}, checkpoint):
                    raise ValueError('Cached baseline scores differ from normalized inference')

                group_score, _ = group_fold_with_diagnostics(
                    normalized, embeddings, scalar, official, labels, valid, hard, fitting, held, seed + fold_id)
                kernel = kernel_fold(embeddings, scalar, labels, valid, hard, fitting, held, seed + fold_id)
                text_score = text_fold(normalized, embeddings, scalar, labels, valid, hard,
                                       fitting, held, seed + fold_id)
                alternate, _ = text_fold_numeric(normalized, embeddings, scalar, labels, valid, hard,
                                                 fitting, held, seed + fold_id)
                for name, value in (('old', old), ('group', group_score), ('kernel', kernel),
                                    ('text', text_score), ('text_numeric', alternate)):
                    components[name][held] = value
                fold_ids[held] = fold_id
                print(f'Seed {seed} fold {fold_id + 1}: components cached', flush=True)

        if (fold_ids < 0).any():
            raise ValueError('Some questions were not covered by the outer folds')
        np.savez(cache_path(seed), seed=np.int64(seed), folds=fold_ids, **components)
        print(f'Seed {seed}: saved {cache_path(seed)} after {time.perf_counter() - started:.0f}s', flush=True)


if __name__ == '__main__':
    main()
