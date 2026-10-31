"""Matched training-only diagnostics for the frozen peer-bottom ensemble."""
import argparse
import json
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
from train_complement import hierarchical_scores, normalized_scores, relative_scores
from train_experiments import evaluate, labels_for
from train_improved import hard_mask
from train_meta import fit_meta, make_meta_rows, train_base
from train_peer import (fit_group_scores, kernel_fold, peer_bottom_scores, scalar_features,
                        text_fold)


SEEDS = (6813, 7253, 7507)


def bottom_distribution(scores, valid, hard):
    observed = scores[valid & hard[:, None]]
    if not len(observed):
        raise ValueError('No valid disagreement candidates for bottom-score diagnostics')
    return {'candidates': len(observed),
            'mean': observed.mean(axis=0).tolist(),
            'std': observed.std(axis=0).tolist()}


def matched_metrics(choices, baseline, labels, official, folds):
    n = len(labels)
    if len(choices) != n or len(baseline) != n:
        raise ValueError('Predictions and labels have different lengths')
    correct = labels[np.arange(n), choices] > 0
    old_correct = labels[np.arange(n), baseline] > 0
    return {**evaluate(choices, labels, official),
            'fold_improvements': [int(correct[fold].sum() - old_correct[fold].sum())
                                  for fold in folds],
            'baseline_only_correct': int((old_correct & ~correct).sum()),
            'variant_only_correct': int((~old_correct & correct).sum())}


def stability_summary(runs):
    if [run['seed'] for run in runs] != list(SEEDS):
        raise ValueError('The stability rule requires the three predeclared seeds')
    seed_gains = []
    fold_gains = []
    for run in runs:
        results = run['results']
        blend = results['baseline_25_peer_75']
        seed_gains.append(blend['disagreement_correct'] - results['baseline']['disagreement_correct'])
        fold_gains.extend(blend['fold_improvements'])
    if len(fold_gains) != 9:
        raise ValueError('The stability rule requires nine outer folds')
    nonnegative_folds = sum(gain >= 0 for gain in fold_gains)
    return {
        'rule': ('Each of the three seed-level disagreement gains is nonnegative, '
                 'and at least six of nine outer-fold gains are nonnegative.'),
        'seed_gains': seed_gains, 'nonnegative_folds': nonnegative_folds,
        'passes': all(gain >= 0 for gain in seed_gains) and nonnegative_folds >= 6,
    }


def subgroup_metrics(choices, baseline, labels, subset):
    n = len(labels)
    if len(choices) != n or len(baseline) != n or len(subset) != n:
        raise ValueError('Predictions, labels and subset have different lengths')
    correct = labels[np.arange(n), choices] > 0
    old_correct = labels[np.arange(n), baseline] > 0
    return {
        'n': int(subset.sum()),
        'baseline_correct': int((subset & old_correct).sum()),
        'variant_correct': int((subset & correct).sum()),
        'baseline_only_correct': int((subset & old_correct & ~correct).sum()),
        'variant_only_correct': int((subset & ~old_correct & correct).sum()),
    }


def backoff_stability_summary(runs):
    if [run['seed'] for run in runs] != list(SEEDS):
        raise ValueError('The backoff stability rule requires the three predeclared seeds')
    seed_gains, fold_gains = [], []
    for run in runs:
        results = run['results']
        variant = results['numeric_backoff']
        seed_gains.append(variant['disagreement_correct']
                          - results['baseline_25_peer_75']['disagreement_correct'])
        fold_gains.extend(variant['fold_improvements'])
    if len(fold_gains) != 9:
        raise ValueError('The backoff stability rule requires nine outer folds')
    nonnegative_folds = sum(gain >= 0 for gain in fold_gains)
    return {
        'rule': ('Each of the three seed-level disagreement gains versus peer75 is nonnegative, '
                 'at least two seed gains are positive, and at least six of nine outer-fold gains '
                 'are nonnegative.'),
        'seed_gains': seed_gains, 'nonnegative_folds': nonnegative_folds,
        'passes': (all(gain >= 0 for gain in seed_gains)
                   and sum(gain > 0 for gain in seed_gains) >= 2 and nonnegative_folds >= 6),
    }


def group_fold_with_diagnostics(bank, embeddings, scalar, official, labels, valid,
                                hard, fitting, held, seed):
    inner = np.array_split(np.random.default_rng(seed + 8100).permutation(len(fitting)), 2)
    oof = np.zeros((len(fitting), 4, 3), dtype=np.float32)
    for local_held in inner:
        mask = np.ones(len(fitting), dtype=bool)
        mask[local_held] = False
        training, validation = fitting[mask], fitting[local_held]
        oof[local_held] = peer_bottom_scores(bank, embeddings, scalar, labels, valid, hard,
                                             training, validation, seed)
    held_base = peer_bottom_scores(bank, embeddings, scalar, labels, valid, hard, fitting, held, seed)
    scores = fit_group_scores(embeddings, scalar, official, labels, fitting, held, oof, held_base, seed)
    distributions = {
        'fitting_oof_half_trained': bottom_distribution(oof, valid[fitting], hard[fitting]),
        'outer_held_full_trained': bottom_distribution(held_base, valid[held], hard[held]),
    }
    return scores, distributions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--numeric-backoff', action='store_true')
    args = parser.parse_args()
    if args.output is None:
        args.output = Path('runs/normalization/numeric-backoff-summary.json' if args.numeric_backoff
                           else 'runs/normalization/peer-75-diagnostics.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump({'status': 'starting', 'seeds': args.seeds}, out, indent=2)
        out.write('\n')
    with np.load('runs/numeric/selected.npz', allow_pickle=False) as existing:
        if (str(existing['meta_mode'].item()) != STACK_MODE
                or float(existing['meta_weight']) != STACK_WEIGHT
                or str(existing['answer_variant'].item()) != 'number_only'
                or str(existing['answer_kind'].item()) != 'mlp'
                or float(existing['answer_weight']) != 0.5):
            raise ValueError('Frozen numeric configuration does not match cross-fitting configuration')

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
    names = ('baseline', 'group', 'kernel', 'text', 'peer_structure',
             'baseline_75_peer_25', 'baseline_50_peer_50', 'baseline_25_peer_75')
    summary = {
        'status': 'running',
        'data': 'training split only; matched question-level nested 3-fold/2-fold cross-fitting',
        'group_mode': 'peer-bottom',
        'seeds': args.seeds,
        'caveat': ('All folds reuse training data previously used for development; splits are not '
                   'independent. No test inputs, predictions or references were read in this run; '
                   'the blend ratio was proposed after earlier aggregate test scores were known.'),
        'bottom_score_order': ['pairwise_linear', 'candidate_lambdarank', 'tfidf_pairwise'],
        'runs': [],
    }
    if args.numeric_backoff:
        from train_numeric_backoff import BACKOFF_WEIGHT, NUMERIC_DIMENSIONS, text_fold_numeric
        summary['numeric_backoff'] = {
            'weight': BACKOFF_WEIGHT, 'dimensions': NUMERIC_DIMENSIONS,
            'representation': 'fixed numeric prefix, suffix and length counts; existing exact tokens retained',
            'comparison': 'same fold-trained group, kernel and baseline; retrain text ranker only',
            'reference': 'baseline_25_peer_75',
        }
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / 'fold.npz'
        for seed in args.seeds:
            folds = np.array_split(np.random.default_rng(seed).permutation(n), 3)
            choices = {name: np.zeros(n, dtype=np.int64) for name in names}
            if args.numeric_backoff:
                choices['numeric_backoff'] = np.zeros(n, dtype=np.int64)
                unseen = np.zeros(n, dtype=bool)
            distributions = []
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
                baseline = select(old, held_official)
                if baseline != predict_normalized(held_bank, {'embeddings': embeddings[held]}, checkpoint):
                    raise ValueError('Baseline scores differ from normalized inference')
                choices['baseline'][held] = baseline

                group_score, distribution = group_fold_with_diagnostics(
                    normalized, embeddings, scalar, official, labels, valid, hard, fitting, held, seed + fold_id)
                distributions.append(distribution)
                print(f'Seed {seed} fold {fold_id + 1}: group complete', flush=True)
                kernel = kernel_fold(embeddings, scalar, labels, valid, hard, fitting, held, seed + fold_id)
                print(f'Seed {seed} fold {fold_id + 1}: kernel complete', flush=True)
                text_score = text_fold(normalized, embeddings, scalar, labels, valid, hard,
                                       fitting, held, seed + fold_id)
                print(f'Seed {seed} fold {fold_id + 1}: text complete', flush=True)
                for name, result in (('group', group_score), ('kernel', kernel), ('text', text_score)):
                    choices[name][held] = select(result, held_official)
                peer = hierarchical_scores(group_score, kernel, text_score, valid[held])['hierarchical']
                choices['peer_structure'][held] = select(peer, held_official)
                old_relative = relative_scores(old, valid[held])
                peer_relative = relative_scores(peer, valid[held])
                for weight, name in ((0.25, 'baseline_75_peer_25'),
                                     (0.50, 'baseline_50_peer_50'),
                                     (0.75, 'baseline_25_peer_75')):
                    choices[name][held] = select((1 - weight) * old_relative + weight * peer_relative,
                                                 held_official)
                if args.numeric_backoff:
                    alternate, held_unseen = text_fold_numeric(
                        normalized, embeddings, scalar, labels, valid, hard, fitting, held, seed + fold_id)
                    alternate_peer = hierarchical_scores(
                        group_score, kernel, alternate, valid[held])['hierarchical']
                    choices['numeric_backoff'][held] = select(
                        0.25 * old_relative + 0.75 * relative_scores(alternate_peer, valid[held]),
                        held_official)
                    unseen[held] = held_unseen
                    print(f'Seed {seed} fold {fold_id + 1}: numeric backoff complete', flush=True)
            results = {'baseline': evaluate(choices['baseline'], labels, official)}
            for name in names[1:]:
                results[name] = matched_metrics(choices[name], choices['baseline'], labels, official, folds)
            run = {'seed': seed, 'results': results, 'bottom_distributions': distributions}
            if args.numeric_backoff:
                results['numeric_backoff'] = matched_metrics(
                    choices['numeric_backoff'], choices['baseline_25_peer_75'], labels, official, folds)
                run['numeric_backoff_subgroups'] = {
                    'disagreement_with_unseen_number': subgroup_metrics(
                        choices['numeric_backoff'], choices['baseline_25_peer_75'], labels, hard & unseen),
                    'disagreement_without_unseen_number': subgroup_metrics(
                        choices['numeric_backoff'], choices['baseline_25_peer_75'], labels, hard & ~unseen),
                }
            summary['runs'].append(run)
            summary['elapsed_seconds'] = time.perf_counter() - started
            args.output.write_text(json.dumps(summary, indent=2) + '\n')
            print(f'Seed {seed}: matched results saved', flush=True)
    summary['status'] = 'complete'
    if tuple(args.seeds) == SEEDS:
        summary['predeclared_stability'] = stability_summary(summary['runs'])
        if args.numeric_backoff:
            summary['backoff_predeclared_stability'] = backoff_stability_summary(summary['runs'])
    summary['elapsed_seconds'] = time.perf_counter() - started
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
