"""Train answer-group residual models using nested training-only folds."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, check_data, load_inputs
from reasoning_features import reasoning_features
from submission import candidate_features, select
from submission_answer import answer_head_scores, feature_subset, group_features, group_max_scores
from submission_meta import meta_scores
from submission_numeric import (feature_subset as numeric_subset, numeric_features,
                                numeric_head_scores, relation_features)
from submission_semantic import (feature_subset as semantic_subset, gate_features,
                                 quantity_evidence, semantic_head_scores)
from train_experiments import Adam, evaluate, labels_for
from train_improved import hard_mask
from train_meta import base_oof, fit_meta, make_meta_rows, predict_base, train_base

STACK_MODE = 'mlp'
STACK_EPOCHS = 12
STACK_WEIGHT = 1.0


def answer_loss_gradient(scores, labels, representatives):
    positives = (labels > 0) & representatives
    if np.any(positives.sum(axis=1) != 1):
        raise ValueError('Each question must have one correct parsable answer group')
    masked = np.where(representatives, scores, -np.inf)
    shifted = masked - masked.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    loss = -np.log(np.maximum(probabilities[positives], 1e-30)).mean()
    return float(loss), (probabilities - positives) / len(scores)


def fit_answer(features, group_base, representatives, labels, hard, variant, kind,
               selector=feature_subset, residual_weight=1.0):
    x = selector(features, variant)
    rows = hard
    train = x[rows]
    mask = representatives[rows]
    flat = train[mask]
    mean = flat.mean(axis=0)
    scale = np.maximum(flat.std(axis=0), np.float32(0.1))
    train = (train - mean) / scale
    targets = labels[rows]
    base = group_base[rows]
    bundle = {'answer_variant': np.array(variant), 'answer_kind': np.array(kind),
              'answer_mean': mean, 'answer_scale': scale}
    dimension = train.shape[-1]
    models = []
    for seed in ((1201,) if kind == 'linear' else (1201, 1202)):
        rng = np.random.default_rng(seed)
        if kind == 'linear':
            parameters = [np.zeros((dimension, 1), dtype=np.float32)]
        elif kind == 'mlp':
            parameters = [(rng.standard_normal((dimension, 32)) * np.sqrt(2 / dimension)).astype(np.float32),
                          np.zeros(32, dtype=np.float32),
                          (rng.standard_normal((32, 1)) * 0.01).astype(np.float32),
                          np.zeros(1, dtype=np.float32)]
        else:
            raise ValueError(f'Unknown answer scorer: {kind}')
        optimizer = Adam(parameters, learning_rate=0.001)
        for _ in range(60 if kind == 'linear' else 24):
            order = rng.permutation(len(train))
            for start in range(0, len(order), 128):
                ids = order[start:start + 128]
                batch = train[ids]
                if kind == 'linear':
                    residual = (batch @ parameters[0]).squeeze(-1)
                else:
                    hidden = np.maximum(batch @ parameters[0] + parameters[1], 0)
                    residual = (hidden @ parameters[2] + parameters[3]).squeeze(-1)
                _, grad = answer_loss_gradient(base[ids] + residual_weight * residual, targets[ids], mask[ids])
                grad = residual_weight * grad[..., None]
                if kind == 'linear':
                    gradients = [batch.reshape(-1, dimension).T @ grad.reshape(-1, 1) + 0.001 * parameters[0]]
                else:
                    hidden_grad = (grad @ parameters[2].T) * (hidden > 0)
                    gradients = [batch.reshape(-1, dimension).T @ hidden_grad.reshape(-1, 32) + 0.001 * parameters[0],
                                 hidden_grad.sum(axis=(0, 1)),
                                 hidden.reshape(-1, 32).T @ grad.reshape(-1, 1) + 0.001 * parameters[2],
                                 grad.sum(axis=(0, 1))]
                optimizer.update(gradients)
        models.append(parameters)
    if kind == 'linear':
        bundle['answer_w'] = models[0][0]
    else:
        for index, key in enumerate(('w1', 'b1', 'w2', 'b2')):
            bundle['answer_' + key] = np.stack([model[index] for model in models])
    return bundle


def feature_rows(indices, scores, bank, answers, reasoning):
    return group_features([bank[i] for i in indices], [answers[i] for i in indices],
                          scores, reasoning[indices])


def stacked_oof(indices, folds, embeddings, text, votes, labels, valid, hard,
                answers, reasoning):
    base_scores = np.zeros((len(indices), 4), dtype=np.float32)
    stack_scores = np.zeros_like(base_scores)
    for fold_id, held_local in enumerate(folds):
        fit_mask = np.ones(len(indices), dtype=bool)
        fit_mask[held_local] = False
        fitting, held = indices[fit_mask], indices[held_local]
        inner = np.array_split(np.random.default_rng(4300 + fold_id).permutation(len(fitting)), 2)
        meta_base = base_oof(fitting, inner, embeddings, text, votes, labels, valid, hard)
        meta_train = make_meta_rows(fitting, meta_base, embeddings, answers, votes, text, reasoning)
        meta = fit_meta(meta_train, labels[fitting], valid[fitting], hard[fitting],
                        STACK_MODE, STACK_EPOCHS)
        base = train_base(fitting, embeddings, text, votes, labels, valid, hard)
        held_base = predict_base(base, held, embeddings, text, votes)
        held_features = make_meta_rows(held, held_base, embeddings, answers, votes, text, reasoning)
        base_scores[held_local] = held_base
        stack_scores[held_local] = held_base + STACK_WEIGHT * meta_scores(held_features, meta)
    return base_scores, stack_scores


def numeric_experiment(output_dir, ablation=False):
    if any((output_dir / name).exists() for name in ('selected.npz', 'summary.json')):
        raise FileExistsError(f'Experiment output already exists in {output_dir}; choose another directory')
    with np.load('runs/meta/selected.npz', allow_pickle=False) as existing:
        if str(existing['meta_mode'].item()) != STACK_MODE or float(existing['meta_weight']) != STACK_WEIGHT:
            raise ValueError('Frozen stacker configuration does not match cross-fitting configuration')
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    check_data(DATA, ('train',))
    bank, arrays = load_inputs('train')
    embeddings = arrays['embeddings']
    labels = labels_for(bank, 'train')
    answers, votes, text = candidate_features(bank)
    reasoning = reasoning_features(bank, answers)
    valid = np.array([[value is not None for value in row] for row in answers], dtype=bool)
    hard = hard_mask(answers)
    numeric = numeric_features(bank, answers)
    if ablation:
        numeric = np.concatenate([numeric, relation_features(bank, answers)], axis=-1)
    n = len(bank)
    folds = np.array_split(np.random.default_rng(4812).permutation(n), 3)
    configs = ([('number_only_base', 'mlp'), ('number_only_words', 'mlp'),
                ('number_only', 'mlp'), ('number_only_relations', 'mlp')]
               if ablation else
               [('aligned', 'mlp'), ('words', 'mlp'), ('rich', 'mlp'),
                ('number_only', 'mlp'), ('number_only', 'linear')])
    oof_baseline = np.zeros((n, 4), dtype=np.float32)
    oof_scores = {config: np.zeros((n, 4), dtype=np.float32) for config in configs}
    oof_independent = np.zeros((n, 4), dtype=np.float32)

    for fold_id, held in enumerate(folds):
        fitting = np.concatenate([fold for i, fold in enumerate(folds) if i != fold_id])
        inner = np.array_split(np.random.default_rng(5300 + fold_id).permutation(len(fitting)), 2)
        fit_base, fit_stack = stacked_oof(fitting, inner, embeddings, text, votes, labels,
                                          valid, hard, answers, reasoning)
        fit_features, reps, fit_groups = feature_rows(fitting, fit_base, bank, answers, reasoning)
        fit_extended = np.concatenate([fit_features, numeric[fitting]], axis=-1)
        fit_stack_groups = group_max_scores(fit_stack, [answers[i] for i in fitting])
        meta_train = make_meta_rows(fitting, fit_base, embeddings, answers, votes, text, reasoning)
        meta = fit_meta(meta_train, labels[fitting], valid[fitting], hard[fitting],
                        STACK_MODE, STACK_EPOCHS)
        base = train_base(fitting, embeddings, text, votes, labels, valid, hard)
        held_base = predict_base(base, held, embeddings, text, votes)
        meta_held = make_meta_rows(held, held_base, embeddings, answers, votes, text, reasoning)
        held_stack = held_base + STACK_WEIGHT * meta_scores(meta_held, meta)
        held_answers = [answers[i] for i in held]
        held_stack_groups = group_max_scores(held_stack, held_answers)
        held_features, _, _ = feature_rows(held, held_base, bank, answers, reasoning)
        held_extended = np.concatenate([held_features, numeric[held]], axis=-1)
        old = fit_answer(fit_features, fit_groups, reps, labels[fitting], hard[fitting],
                         'question', 'mlp')
        oof_baseline[held] = held_stack_groups + 0.5 * answer_head_scores(held_features, old)
        for variant, kind in configs:
            head = fit_answer(fit_extended, fit_stack_groups, reps, labels[fitting], hard[fitting],
                              variant, kind, selector=numeric_subset, residual_weight=0.5)
            residual = numeric_head_scores(held_extended, head)
            oof_scores[(variant, kind)][held] = held_stack_groups + 0.5 * residual
            if (variant, kind) == ('number_only', 'mlp'):
                oof_independent[held] = residual
        print(f'Completed numerical nested training fold {fold_id + 1}/3', flush=True)

    baseline_choices = select(oof_baseline, answers)
    baseline = evaluate(baseline_choices, labels, answers)
    comparison_scores = oof_scores[('number_only', 'mlp')] if ablation else oof_baseline
    comparison_choices = select(comparison_scores, answers)
    comparison = evaluate(comparison_choices, labels, answers)
    fold_baseline = []
    for fold in folds:
        rows = [answers[i] for i in fold]
        fold_baseline.append(evaluate(select(comparison_scores[fold], rows), labels[fold],
                                      rows)['disagreement_correct'])
    records = [{'name': 'previous_answer_model', **baseline}]
    best = (comparison['disagreement_correct'], None)
    for (variant, kind), scores in oof_scores.items():
        metrics = evaluate(select(scores, answers), labels, answers)
        gains = []
        for fold_id, fold in enumerate(folds):
            rows = [answers[i] for i in fold]
            result = evaluate(select(scores[fold], rows), labels[fold], rows)
            gains.append(result['disagreement_correct'] - fold_baseline[fold_id])
        record = {'variant': variant, 'kind': kind, 'weight': 0.5,
                  'fold_improvements': gains, **metrics}
        records.append(record)
        eligible = not ablation or variant == 'number_only_relations'
        if (eligible and sum(gain > 0 for gain in gains) >= 2
                and metrics['disagreement_correct'] >= comparison['disagreement_correct'] + 5
                and metrics['disagreement_correct'] > best[0]):
            best = (metrics['disagreement_correct'], {'variant': variant, 'kind': kind, 'weight': 0.5})
    independent_choices = np.asarray(select(oof_independent, answers))
    baseline_correct = labels[np.arange(n), baseline_choices] > 0
    independent_correct = labels[np.arange(n), independent_choices] > 0
    complementarity = {
        'subset': 'training disagreement questions',
        'baseline_only_correct': int(np.sum(hard & baseline_correct & ~independent_correct)),
        'numeric_only_correct': int(np.sum(hard & ~baseline_correct & independent_correct)),
        'both_correct': int(np.sum(hard & baseline_correct & independent_correct)),
        'both_wrong': int(np.sum(hard & ~baseline_correct & ~independent_correct)),
        'caveat': 'numeric-only head was trained with stack scores as an offset; this is not a fully independent model',
    }
    config = best[1]
    if config is None:
        selected = {'name': 'previous_numeric_model' if ablation else 'previous_answer_model',
                    'reason': 'no sufficient nested-CV improvement'}
    else:
        indices = np.arange(n)
        inner = np.array_split(np.random.default_rng(4812).permutation(n), 3)
        fit_base, fit_stack = stacked_oof(indices, inner, embeddings, text, votes, labels,
                                          valid, hard, answers, reasoning)
        features, reps, _ = feature_rows(indices, fit_base, bank, answers, reasoning)
        features = np.concatenate([features, numeric], axis=-1)
        groups = group_max_scores(fit_stack, answers)
        head = fit_answer(features, groups, reps, labels, hard, config['variant'], config['kind'],
                          selector=numeric_subset, residual_weight=config['weight'])
        with np.load('runs/meta/selected.npz', allow_pickle=False) as existing:
            bundle = {key: existing[key] for key in existing.files}
        bundle.update(head)
        bundle['answer_weight'] = np.float32(config['weight'])
        np.savez(output_dir / 'selected.npz', **bundle)
        selected = {'name': 'numeric_answer_residual', **config}
    summary = {
        'selection_data': 'nested 3-fold/2-fold question-level training CV, seed 4812',
        'selection_rule': ('relations must improve the numeric-only model by at least 5 correct '
                           'and improve two folds' if ablation else
                           'at least 5 more correct than previous answer model and two folds improve'),
        'selection_caveat': ('three matched ablations and one relation variant compared; '
                             'previous architectures were selected on overlapping data' if ablation else
                             'five configurations compared; previous architectures were selected on overlapping data'),
        'base_offset': 'both ranker and stacker predictions are out-of-fold for residual-head training',
        'complementarity': complementarity, 'selected': selected, 'records': records,
        'elapsed_seconds': time.perf_counter() - start,
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('BASELINE ' + json.dumps(baseline), flush=True)
    if ablation:
        print('NUMERIC BASELINE ' + json.dumps(comparison), flush=True)
    print('SELECTED ' + json.dumps(selected), flush=True)


def semantic_experiment(output_dir, wide=False, check=False):
    if any((output_dir / name).exists() for name in ('selected.npz', 'summary.json')):
        raise FileExistsError(f'Experiment output already exists in {output_dir}; choose another directory')
    with np.load('runs/meta/selected.npz', allow_pickle=False) as existing:
        if str(existing['meta_mode'].item()) != STACK_MODE or float(existing['meta_weight']) != STACK_WEIGHT:
            raise ValueError('Frozen stacker configuration does not match cross-fitting configuration')
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    check_data(DATA, ('train',))
    bank, arrays = load_inputs('train')
    embeddings = arrays['embeddings']
    labels = labels_for(bank, 'train')
    answers, votes, text = candidate_features(bank)
    reasoning = reasoning_features(bank, answers)
    valid = np.array([[value is not None for value in row] for row in answers], dtype=bool)
    hard = hard_mask(answers)
    numeric = numeric_features(bank, answers)
    narrow_evidence = quantity_evidence(bank, answers)
    evidence = quantity_evidence(bank, answers, max_mentions=8) if wide else narrow_evidence
    n = len(bank)
    fold_seed = 7253 if check else 6813
    folds = np.array_split(np.random.default_rng(fold_seed).permutation(n), 3)
    configs = ([('parsed', 'linear'), ('conditional', 'linear')] if check else
               [('conditional', 'linear')] if wide else
               [('parser', 'mlp'), ('parsed', 'mlp'),
                ('conditional', 'mlp'), ('conditional', 'linear')])
    oof_baseline = np.zeros((n, 4), dtype=np.float32)
    oof_scores = {config: np.zeros((n, 4), dtype=np.float32) for config in configs}

    for fold_id, held in enumerate(folds):
        fitting = np.concatenate([fold for i, fold in enumerate(folds) if i != fold_id])
        inner = np.array_split(np.random.default_rng(6800 + fold_id).permutation(len(fitting)), 2)
        fit_base, fit_stack = stacked_oof(fitting, inner, embeddings, text, votes, labels,
                                          valid, hard, answers, reasoning)
        fit_answers = [answers[i] for i in fitting]
        fit_group, reps, _ = feature_rows(fitting, fit_base, bank, answers, reasoning)
        fit_old = np.concatenate([fit_group, numeric[fitting]], axis=-1)
        fit_gate = gate_features(fit_stack, fit_answers)
        fit_all = np.concatenate([fit_old, evidence[fitting], fit_gate], axis=-1)
        fit_offset = group_max_scores(fit_stack, fit_answers)

        meta_train = make_meta_rows(fitting, fit_base, embeddings, answers, votes, text, reasoning)
        meta = fit_meta(meta_train, labels[fitting], valid[fitting], hard[fitting],
                        STACK_MODE, STACK_EPOCHS)
        base = train_base(fitting, embeddings, text, votes, labels, valid, hard)
        held_base = predict_base(base, held, embeddings, text, votes)
        meta_held = make_meta_rows(held, held_base, embeddings, answers, votes, text, reasoning)
        held_stack = held_base + STACK_WEIGHT * meta_scores(meta_held, meta)
        held_answers = [answers[i] for i in held]
        held_group, _, _ = feature_rows(held, held_base, bank, answers, reasoning)
        held_old = np.concatenate([held_group, numeric[held]], axis=-1)
        held_gate = gate_features(held_stack, held_answers)
        held_all = np.concatenate([held_old, evidence[held], held_gate], axis=-1)
        held_offset = group_max_scores(held_stack, held_answers)

        if wide:
            fit_narrow = np.concatenate([fit_old, narrow_evidence[fitting], fit_gate], axis=-1)
            held_narrow = np.concatenate([held_old, narrow_evidence[held], held_gate], axis=-1)
            baseline = fit_answer(fit_narrow, fit_offset, reps, labels[fitting], hard[fitting],
                                  'conditional', 'linear', selector=semantic_subset, residual_weight=0.5)
            oof_baseline[held] = held_offset + 0.5 * semantic_head_scores(held_narrow, baseline)
        else:
            baseline = fit_answer(fit_old, fit_offset, reps, labels[fitting], hard[fitting],
                                  'number_only', 'mlp', selector=numeric_subset, residual_weight=0.5)
            oof_baseline[held] = held_offset + 0.5 * numeric_head_scores(held_old, baseline)
        for variant, kind in configs:
            head = fit_answer(fit_all, fit_offset, reps, labels[fitting], hard[fitting],
                              variant, kind, selector=semantic_subset, residual_weight=0.5)
            oof_scores[(variant, kind)][held] = held_offset + 0.5 * semantic_head_scores(held_all, head)
        print(f'Completed semantic nested training fold {fold_id + 1}/3', flush=True)

    baseline = evaluate(select(oof_baseline, answers), labels, answers)
    fold_baseline = []
    for fold in folds:
        rows = [answers[i] for i in fold]
        fold_baseline.append(evaluate(select(oof_baseline[fold], rows), labels[fold],
                                      rows)['disagreement_correct'])
    records = [{'name': 'five_quantity_semantic_model' if wide else 'previous_numeric_model', **baseline}]
    best = (baseline['disagreement_correct'], None)
    for (variant, kind), scores in oof_scores.items():
        metrics = evaluate(select(scores, answers), labels, answers)
        improvements = []
        for fold_id, fold in enumerate(folds):
            rows = [answers[i] for i in fold]
            result = evaluate(select(scores[fold], rows), labels[fold], rows)
            improvements.append(result['disagreement_correct'] - fold_baseline[fold_id])
        record = {'variant': variant, 'kind': kind, 'weight': 0.5,
                  'fold_improvements': improvements, **metrics}
        records.append(record)
        if (not check and sum(gain > 0 for gain in improvements) >= 2
                and metrics['disagreement_correct'] >= baseline['disagreement_correct'] + 5
                and metrics['disagreement_correct'] > best[0]):
            best = (metrics['disagreement_correct'], {'variant': variant, 'kind': kind, 'weight': 0.5})
    config = best[1]
    if check:
        selected = {'name': 'diagnostic_only', 'reason': 'second fold seed; no configuration selected'}
    elif config is None:
        selected = {'name': 'five_quantity_semantic_model' if wide else 'previous_numeric_model',
                    'reason': 'no sufficient nested-CV improvement'}
    else:
        indices = np.arange(n)
        inner = np.array_split(np.random.default_rng(6813).permutation(n), 3)
        fit_base, fit_stack = stacked_oof(indices, inner, embeddings, text, votes, labels,
                                          valid, hard, answers, reasoning)
        group, reps, _ = feature_rows(indices, fit_base, bank, answers, reasoning)
        features = np.concatenate([group, numeric, evidence, gate_features(fit_stack, answers)], axis=-1)
        offset = group_max_scores(fit_stack, answers)
        head = fit_answer(features, offset, reps, labels, hard, config['variant'], config['kind'],
                          selector=semantic_subset, residual_weight=config['weight'])
        with np.load('runs/meta/selected.npz', allow_pickle=False) as existing:
            bundle = {key: existing[key] for key in existing.files}
        bundle.update(head)
        bundle['answer_weight'] = np.float32(config['weight'])
        if wide:
            bundle['quantity_limit'] = np.int32(8)
        np.savez(output_dir / 'selected.npz', **bundle)
        selected = {'name': 'semantic_answer_residual', **config}
        if wide:
            selected['quantity_limit'] = 8
    summary = {
        'selection_data': f'nested 3-fold/2-fold question-level training CV, seed {fold_seed}',
        'selection_rule': ('diagnostic only; no model selected' if check else
                           'at least 5 more correct than the five-quantity semantic model and two folds improve'
                           if wide else 'at least 5 more correct than previous numeric model and two folds improve'),
        'selection_caveat': ('same training data used for earlier architecture selection; not a new independent dataset'
                             if check else 'five-quantity architecture was selected on these folds; earlier architectures used overlapping data'
                             if wide else 'four configurations compared; previous architectures were selected on overlapping data'),
        'base_offset': 'both ranker and stacker predictions are out-of-fold for residual-head training',
        'variants': {'parser': 'new quantity extraction only',
                     'parsed': 'quantity extraction plus bounded distinct-mention arithmetic',
                     'conditional': 'parsed features plus group-relative stack score and vote context'},
        'selected': selected, 'records': records, 'elapsed_seconds': time.perf_counter() - started,
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('BASELINE ' + json.dumps(baseline), flush=True)
    print('SELECTED ' + json.dumps(selected), flush=True)


def numeric_oof(indices, folds, bank, embeddings, text, votes, labels, valid, hard,
                answers, reasoning, numeric):
    base_scores = np.zeros((len(indices), 4), dtype=np.float32)
    stack_scores = np.zeros_like(base_scores)
    numeric_scores = np.zeros_like(base_scores)
    for fold_id, held_local in enumerate(folds):
        fit_mask = np.ones(len(indices), dtype=bool)
        fit_mask[held_local] = False
        fitting, held = indices[fit_mask], indices[held_local]
        inner = np.array_split(np.random.default_rng(4300 + fold_id).permutation(len(fitting)), 2)
        fit_base, fit_stack = stacked_oof(fitting, inner, embeddings, text, votes, labels,
                                          valid, hard, answers, reasoning)
        fit_group, reps, _ = feature_rows(fitting, fit_base, bank, answers, reasoning)
        fit_old = np.concatenate([fit_group, numeric[fitting]], axis=-1)
        fit_answers = [answers[i] for i in fitting]
        head = fit_answer(fit_old, group_max_scores(fit_stack, fit_answers), reps,
                          labels[fitting], hard[fitting], 'number_only', 'mlp',
                          selector=numeric_subset, residual_weight=0.5)
        meta_train = make_meta_rows(fitting, fit_base, embeddings, answers, votes, text, reasoning)
        meta = fit_meta(meta_train, labels[fitting], valid[fitting], hard[fitting],
                        STACK_MODE, STACK_EPOCHS)
        base = train_base(fitting, embeddings, text, votes, labels, valid, hard)
        held_base = predict_base(base, held, embeddings, text, votes)
        held_meta = make_meta_rows(held, held_base, embeddings, answers, votes, text, reasoning)
        held_stack = held_base + STACK_WEIGHT * meta_scores(held_meta, meta)
        held_answers = [answers[i] for i in held]
        held_group, _, _ = feature_rows(held, held_base, bank, answers, reasoning)
        held_old = np.concatenate([held_group, numeric[held]], axis=-1)
        base_scores[held_local] = held_base
        stack_scores[held_local] = held_stack
        numeric_scores[held_local] = (group_max_scores(held_stack, held_answers)
                                      + 0.5 * numeric_head_scores(held_old, head))
    return base_scores, stack_scores, numeric_scores


def fit_full_semantic_correction(variant, bank, embeddings, text, votes, labels, valid, hard,
                                 answers, reasoning, numeric, evidence):
    indices = np.arange(len(bank))
    folds = np.array_split(np.random.default_rng(6813).permutation(len(bank)), 3)
    fit_base, _, fit_numeric = numeric_oof(indices, folds, bank, embeddings, text, votes, labels,
                                           valid, hard, answers, reasoning, numeric)
    group, reps, _ = feature_rows(indices, fit_base, bank, answers, reasoning)
    parts = [group, numeric, evidence]
    if variant == 'correction_conditional':
        parts.append(gate_features(fit_numeric, answers))
    elif variant != 'correction_parsed':
        raise ValueError(f'Unknown correction feature variant: {variant}')
    features = np.concatenate(parts, axis=-1)
    correction = fit_answer(features, fit_numeric, reps, labels, hard, variant, 'linear',
                            selector=semantic_subset, residual_weight=0.5)
    with np.load('runs/numeric/selected.npz', allow_pickle=False) as existing:
        bundle = {key: existing[key] for key in existing.files}
    bundle.update({'correction_' + key: value for key, value in correction.items()})
    bundle['correction_weight'] = np.float32(0.5)
    bundle['quantity_limit'] = np.int32(5)
    return bundle


def semantic_residual_experiment(output_dir, fixed_parsed=False):
    if any((output_dir / name).exists() for name in ('selected.npz', 'summary.json')):
        raise FileExistsError(f'Experiment output already exists in {output_dir}; choose another directory')
    with np.load('runs/numeric/selected.npz', allow_pickle=False) as existing:
        if (str(existing['meta_mode'].item()) != STACK_MODE
                or float(existing['meta_weight']) != STACK_WEIGHT
                or str(existing['answer_variant'].item()) != 'number_only'
                or str(existing['answer_kind'].item()) != 'mlp'
                or float(existing['answer_weight']) != 0.5):
            raise ValueError('Frozen numeric configuration does not match cross-fitting configuration')
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    check_data(DATA, ('train',))
    bank, arrays = load_inputs('train')
    embeddings = arrays['embeddings']
    labels = labels_for(bank, 'train')
    answers, votes, text = candidate_features(bank)
    reasoning = reasoning_features(bank, answers)
    valid = np.array([[value is not None for value in row] for row in answers], dtype=bool)
    hard = hard_mask(answers)
    numeric = numeric_features(bank, answers)
    evidence = quantity_evidence(bank, answers)
    n = len(bank)
    if fixed_parsed:
        variant = 'correction_parsed'
        bundle = fit_full_semantic_correction(variant, bank, embeddings, text, votes, labels, valid,
                                              hard, answers, reasoning, numeric, evidence)
        np.savez(output_dir / 'selected.npz', **bundle)
        summary = {
            'training_data': 'training split only; 3-fold/2-fold/2-fold cross-fitting, seed 6813',
            'selection': 'fixed experimental candidate requested for validation; not selected by the earlier two-seed rule',
            'base_offset': 'ranker, stacker and numeric answer head are out-of-fold for correction training',
            'variant': variant, 'kind': 'linear', 'weight': 0.5,
            'quantity_limit': 5, 'n': n, 'disagreement_n': int(hard.sum()),
            'previous_comparison': 'runs/semantic-residual/summary.json',
            'elapsed_seconds': time.perf_counter() - started,
        }
        (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('FIXED CANDIDATE ' + json.dumps(summary), flush=True)
        return
    variants = ('correction_parsed', 'correction_conditional')
    results = []
    for seed in (6813, 7253):
        folds = np.array_split(np.random.default_rng(seed).permutation(n), 3)
        oof_baseline = np.zeros((n, 4), dtype=np.float32)
        oof_scores = {variant: np.zeros((n, 4), dtype=np.float32) for variant in variants}
        for fold_id, held in enumerate(folds):
            fitting = np.concatenate([fold for i, fold in enumerate(folds) if i != fold_id])
            inner = np.array_split(np.random.default_rng(6800 + fold_id).permutation(len(fitting)), 2)
            fit_base, fit_stack, fit_numeric = numeric_oof(
                fitting, inner, bank, embeddings, text, votes, labels, valid, hard, answers, reasoning, numeric)
            fit_answers = [answers[i] for i in fitting]
            fit_group, reps, _ = feature_rows(fitting, fit_base, bank, answers, reasoning)
            fit_old = np.concatenate([fit_group, numeric[fitting]], axis=-1)
            baseline_head = fit_answer(fit_old, group_max_scores(fit_stack, fit_answers), reps,
                                       labels[fitting], hard[fitting], 'number_only', 'mlp',
                                       selector=numeric_subset, residual_weight=0.5)
            meta_train = make_meta_rows(fitting, fit_base, embeddings, answers, votes, text, reasoning)
            meta = fit_meta(meta_train, labels[fitting], valid[fitting], hard[fitting], STACK_MODE, STACK_EPOCHS)
            base = train_base(fitting, embeddings, text, votes, labels, valid, hard)
            held_base = predict_base(base, held, embeddings, text, votes)
            held_meta = make_meta_rows(held, held_base, embeddings, answers, votes, text, reasoning)
            held_stack = held_base + STACK_WEIGHT * meta_scores(held_meta, meta)
            held_answers = [answers[i] for i in held]
            held_group, _, _ = feature_rows(held, held_base, bank, answers, reasoning)
            held_old = np.concatenate([held_group, numeric[held]], axis=-1)
            held_numeric = (group_max_scores(held_stack, held_answers)
                            + 0.5 * numeric_head_scores(held_old, baseline_head))
            oof_baseline[held] = held_numeric
            fit_all = np.concatenate([fit_old, evidence[fitting], gate_features(fit_numeric, fit_answers)], axis=-1)
            held_all = np.concatenate([held_old, evidence[held], gate_features(held_numeric, held_answers)], axis=-1)
            for variant in variants:
                correction = fit_answer(fit_all, fit_numeric, reps, labels[fitting], hard[fitting],
                                        variant, 'linear', selector=semantic_subset, residual_weight=0.5)
                oof_scores[variant][held] = held_numeric + 0.5 * semantic_head_scores(held_all, correction)
            print(f'Completed numeric residual seed {seed}, fold {fold_id + 1}/3', flush=True)
        baseline = evaluate(select(oof_baseline, answers), labels, answers)
        baseline_folds = []
        for fold in folds:
            rows = [answers[i] for i in fold]
            baseline_folds.append(evaluate(select(oof_baseline[fold], rows), labels[fold],
                                           rows)['disagreement_correct'])
        records = [{'name': 'previous_numeric_model', **baseline}]
        for variant in variants:
            scores = oof_scores[variant]
            metrics = evaluate(select(scores, answers), labels, answers)
            gains = []
            for fold_id, fold in enumerate(folds):
                rows = [answers[i] for i in fold]
                result = evaluate(select(scores[fold], rows), labels[fold], rows)
                gains.append(result['disagreement_correct'] - baseline_folds[fold_id])
            records.append({'variant': variant, 'kind': 'linear', 'weight': 0.5,
                            'fold_improvements': gains, **metrics})
        results.append({'seed': seed, 'records': records})
    eligible = []
    for index, variant in enumerate(variants, start=1):
        gains = [run['records'][index]['disagreement_correct']
                 - run['records'][0]['disagreement_correct'] for run in results]
        if all(gain >= 5 and sum(fold > 0 for fold in run['records'][index]['fold_improvements']) >= 2
               for gain, run in zip(gains, results)):
            eligible.append((min(gains), sum(gains), variant))
    if eligible:
        variant = max(eligible)[2]
        bundle = fit_full_semantic_correction(variant, bank, embeddings, text, votes, labels, valid,
                                              hard, answers, reasoning, numeric, evidence)
        np.savez(output_dir / 'selected.npz', **bundle)
        selected = {'name': 'numeric_with_semantic_correction', 'variant': variant, 'weight': 0.5}
    else:
        selected = {'name': 'previous_numeric_model', 'reason': 'no sufficient improvement on both fold seeds'}
    summary = {
        'selection_data': 'nested training-only 3-fold/2-fold/2-fold cross-fitting; seeds 6813 and 7253',
        'selection_rule': 'at least 5 more disagreement questions and two folds improve on each seed',
        'selection_caveat': 'both seeds reuse training data previously used for architecture exploration',
        'base_offset': 'ranker, stacker and numeric answer head are all out-of-fold for correction training',
        'selected': selected, 'runs': results, 'elapsed_seconds': time.perf_counter() - started,
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('SELECTED ' + json.dumps(selected), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--numeric', action='store_true',
                       help='run the separately cross-fitted numerical feature experiment')
    modes.add_argument('--numeric-ablation', action='store_true',
                       help='compare matched numerical ablations and soft relational features')
    modes.add_argument('--semantic', action='store_true',
                       help='test improved quantity parsing and bounded contextual features')
    modes.add_argument('--semantic-wide', action='store_true',
                       help='compare eight versus five quantity mentions on matched training folds')
    modes.add_argument('--semantic-check', action='store_true',
                       help='evaluate fixed linear feature variants with a second training fold seed')
    modes.add_argument('--semantic-residual', action='store_true',
                       help='test a cross-fitted semantic correction on the previous numeric model')
    modes.add_argument('--semantic-parsed', action='store_true',
                       help='train the fixed parse/arithmetic-only correction for validation')
    args = parser.parse_args()
    if args.output_dir is None:
        name = ('semantic-parsed' if args.semantic_parsed else
                'semantic-residual' if args.semantic_residual else
                'semantic-check' if args.semantic_check else 'semantic-wide' if args.semantic_wide else
                'semantic' if args.semantic else
                'numeric-ablation' if args.numeric_ablation else
                'numeric' if args.numeric else 'answer')
        args.output_dir = Path('runs') / name
    if args.semantic_residual or args.semantic_parsed:
        semantic_residual_experiment(args.output_dir, fixed_parsed=args.semantic_parsed)
        return
    if args.semantic or args.semantic_wide or args.semantic_check:
        semantic_experiment(args.output_dir, wide=args.semantic_wide, check=args.semantic_check)
        return
    if args.numeric or args.numeric_ablation:
        numeric_experiment(args.output_dir, ablation=args.numeric_ablation)
        return
    if any((args.output_dir / name).exists() for name in ('selected.npz', 'summary.json')):
        raise FileExistsError(f'Experiment output already exists in {args.output_dir}; choose another directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    check_data(DATA, ('train',))
    bank, arrays = load_inputs('train')
    embeddings = arrays['embeddings']
    labels = labels_for(bank, 'train')
    answers, votes, text = candidate_features(bank)
    reasoning = reasoning_features(bank, answers)
    valid = np.array([[value is not None for value in row] for row in answers], dtype=bool)
    hard = hard_mask(answers)
    n = len(bank)
    folds = np.array_split(np.random.default_rng(2029).permutation(n), 3)
    configs = [(variant, kind) for variant in ('group', 'regime', 'question') for kind in ('linear', 'mlp')]
    oof_base = np.zeros((n, 4), dtype=np.float32)
    oof_group_base = np.zeros((n, 4), dtype=np.float32)
    oof_residuals = {config: np.zeros((n, 4), dtype=np.float32) for config in configs}

    for fold_id, held in enumerate(folds):
        fitting = np.concatenate([fold for i, fold in enumerate(folds) if i != fold_id])
        inner = np.array_split(np.random.default_rng(1000 + fold_id).permutation(len(fitting)), 2)
        inner_scores = base_oof(fitting, inner, embeddings, text, votes, labels, valid, hard)
        features, reps, group_base = feature_rows(fitting, inner_scores, bank, answers, reasoning)
        base = train_base(fitting, embeddings, text, votes, labels, valid, hard)
        held_scores = predict_base(base, held, embeddings, text, votes)
        oof_base[held] = held_scores
        held_features, _, held_group_base = feature_rows(held, held_scores, bank, answers, reasoning)
        oof_group_base[held] = held_group_base
        for variant, kind in configs:
            head = fit_answer(features, group_base, reps, labels[fitting], hard[fitting], variant, kind)
            oof_residuals[(variant, kind)][held] = answer_head_scores(held_features, head)
        print(f'Completed training-only nested fold {fold_id + 1}/3', flush=True)

    baseline = evaluate(select(oof_base, answers), labels, answers)
    records = [{'name': 'base_ranker', **baseline}]
    baseline_folds = []
    for fold in folds:
        subset = [answers[i] for i in fold]
        baseline_folds.append(evaluate(select(oof_base[fold], subset), labels[fold], subset)['disagreement_correct'])
    best = (baseline['disagreement_correct'], None)
    for (variant, kind), residual in oof_residuals.items():
        for weight in (0.25, 0.5, 1.0, 2.0):
            scores = oof_group_base + weight * residual
            result = evaluate(select(scores, answers), labels, answers)
            improvements = []
            for fold_id, fold in enumerate(folds):
                subset = [answers[i] for i in fold]
                value = evaluate(select(scores[fold], subset), labels[fold], subset)['disagreement_correct']
                improvements.append(value - baseline_folds[fold_id])
            config = {'variant': variant, 'kind': kind, 'weight': weight}
            records.append({**config, 'fold_improvements': improvements, **result})
            if (sum(value > 0 for value in improvements) >= 2
                    and result['disagreement_correct'] >= baseline['disagreement_correct'] + 5
                    and result['disagreement_correct'] > best[0]):
                best = (result['disagreement_correct'], config)
    config = best[1]
    if config is None:
        selected = {'name': 'base_ranker', 'reason': 'no sufficient nested-CV improvement'}
    else:
        indices = np.arange(n)
        inner = np.array_split(np.random.default_rng(2029).permutation(n), 3)
        scores = base_oof(indices, inner, embeddings, text, votes, labels, valid, hard)
        features, reps, group_base = feature_rows(indices, scores, bank, answers, reasoning)
        head = fit_answer(features, group_base, reps, labels, hard, config['variant'], config['kind'])
        with np.load('runs/meta/selected.npz', allow_pickle=False) as existing:
            bundle = {key: existing[key] for key in existing.files}
        bundle.update(head)
        bundle['answer_weight'] = np.float32(config['weight'])
        np.savez(args.output_dir / 'selected.npz', **bundle)
        selected = {'name': 'answer_group_residual', **config}
    summary = {'selection_data': 'nested 3-fold/2-fold question-level training CV, seed 2029',
               'selection_metric': 'accuracy on training questions with distinct parsable answers',
               'selection_rule': 'at least 5 more correct and at least two folds improve; ties prefer base',
               'caveat': 'base hyperparameters previously selected on overlapping training data; outer folds searched over configs',
               'selected': selected, 'records': records, 'elapsed_seconds': time.perf_counter() - start}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('BASELINE ' + json.dumps(baseline), flush=True)
    print('SELECTED ' + json.dumps(selected), flush=True)


if __name__ == '__main__':
    main()
