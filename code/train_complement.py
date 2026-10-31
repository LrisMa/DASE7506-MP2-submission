"""Compare complementary pairwise rankers with the normalized selector on training folds."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from common import DATA, check_data, load_inputs
from reasoning_features import reasoning_features
from submission import candidate_features, select
from submission_answer import group_features, group_max_scores
from submission_meta import meta_features, meta_scores, ranker_scores
from submission_lexical import lexical_features
from submission_normalized import (normalize_bank, normalized_candidate_features,
                                   predict as predict_normalized)
from submission_numeric import feature_subset as numeric_subset, numeric_features, numeric_head_scores
from train_answer import STACK_EPOCHS, STACK_MODE, STACK_WEIGHT, feature_rows, fit_answer, stacked_oof
from train_experiments import evaluate, labels_for
from train_improved import hard_mask
from train_meta import fit_meta, make_meta_rows, train_base
from train_ranker_cv import fit_all_pairs


KERNEL_GAMMA = 4 / 1024
KERNEL_WEIGHT = 2.0
BLEND_WEIGHTS = (0.10, 0.25)
SEEDS = (6813, 7253)


def rbf_kernel(x, landmarks, gamma):
    x = np.asarray(x, dtype=np.float64)
    landmarks = np.asarray(landmarks, dtype=np.float64)
    squared = (np.sum(x * x, axis=1, keepdims=True)
               + np.sum(landmarks * landmarks, axis=1)[None, :]
               - 2 * x @ landmarks.T)
    return np.exp(-gamma * np.maximum(squared, 0))


def kernel_features(features, model):
    x = (features - model['mean']) / model['scale']
    flat = x.reshape(-1, x.shape[-1])
    basis = rbf_kernel(flat, model['landmarks'], KERNEL_GAMMA) @ model['inverse_sqrt']
    basis = basis.reshape(*x.shape[:-1], -1).astype(np.float32)
    return np.concatenate([x, KERNEL_WEIGHT * basis], axis=-1)


def fit_kernel_ranker(features, labels, valid, hard, seed, landmarks=512, epochs=80):
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0)
    scale = np.maximum(flat.std(axis=0), np.float32(1e-6))
    normalized = (flat - mean) / scale
    if landmarks > len(normalized):
        raise ValueError('More landmarks requested than fitting candidates')
    chosen = np.random.default_rng(seed).choice(len(normalized), landmarks, replace=False)
    centers = normalized[chosen].copy()
    gram = rbf_kernel(centers, centers, KERNEL_GAMMA)
    values, vectors = np.linalg.eigh(gram)
    if values.min() < -1e-7:
        raise ValueError('Nyström landmark kernel is not positive semidefinite')
    inverse_sqrt = (vectors * (1 / np.sqrt(np.maximum(values, 1e-8)))) @ vectors.T
    model = {'mean': mean, 'scale': scale, 'landmarks': centers, 'inverse_sqrt': inverse_sqrt}
    expanded = kernel_features(features, model)
    model['w'] = fit_all_pairs(expanded[hard], labels[hard], valid[hard], seed,
                               epochs, 128, 0.001, (epochs,))[epochs]
    return model


def kernel_scores(features, model):
    return (kernel_features(features, model) @ model['w']).squeeze(-1)


def relative_scores(scores, valid):
    counts = valid.sum(axis=1, keepdims=True)
    safe_counts = np.maximum(counts, 1)
    mean = np.where(valid, scores, 0).sum(axis=1, keepdims=True) / safe_counts
    centered = np.where(valid, scores - mean, 0)
    scale = np.sqrt((centered * centered).sum(axis=1, keepdims=True) / safe_counts)
    return centered / np.maximum(scale, 0.1)


def hierarchical_scores(base, kernel, text, valid):
    base_relative = relative_scores(base, valid)
    kernel_relative = relative_scores(kernel, valid)
    text_relative = relative_scores(text, valid)
    first = relative_scores(0.25 * base_relative + 0.75 * kernel_relative, valid)
    combined = 0.75 * first + 0.25 * text_relative
    combined_relative = relative_scores(combined, valid)
    return {
        'kernel_25': 0.75 * base_relative + 0.25 * kernel_relative,
        'text_25': 0.75 * base_relative + 0.25 * text_relative,
        'first_layer': first,
        'hierarchical': combined,
        'baseline_75_hierarchical_25': 0.75 * base_relative + 0.25 * combined_relative,
        'baseline_50_hierarchical_50': 0.50 * base_relative + 0.50 * combined_relative,
    }


def text_counts(bank):
    normalized = normalize_bank(bank)
    questions = [{'candidates': [{'text': row['question']}]} for row in normalized]
    tails = [{'candidates': [{'text': candidate['text'][-512:]}
                            for candidate in row['candidates']]} for row in normalized]
    return (lexical_features(questions)[:, 0].copy(),
            lexical_features(normalized), lexical_features(tails))


def tfidf_projection(counts, fitting, components):
    flat = counts.reshape(-1, counts.shape[-1])
    train = flat[fitting]
    idf = np.log((1 + len(train)) / (1 + (train > 0).sum(axis=0))) + 1
    weighted = flat * idf.astype(np.float32)
    lengths = np.linalg.norm(weighted, axis=1, keepdims=True)
    weighted /= np.maximum(lengths, np.float32(1e-12))
    train = weighted[fitting]
    gram = train.T @ train
    _, vectors = np.linalg.eigh(gram.astype(np.float64))
    projection = vectors[:, -components:].astype(np.float32)
    return weighted, weighted @ projection


def text_fold_scores(counts, embeddings, text, votes, labels, valid, hard, fitting, held, seed):
    question_counts, candidate_counts, tail_counts = counts
    n = len(embeddings)
    candidate_fitting = np.arange(n * 4).reshape(n, 4)[fitting].reshape(-1)
    _, question = tfidf_projection(question_counts, fitting, 16)
    candidate_tfidf, candidate = tfidf_projection(candidate_counts, candidate_fitting, 32)
    _, tail = tfidf_projection(tail_counts, candidate_fitting, 32)
    candidate_tfidf = candidate_tfidf.reshape(n, 4, -1)
    candidate = candidate.reshape(n, 4, -1)
    tail = tail.reshape(n, 4, -1)

    def features(indices):
        interactions = np.einsum('ni,njk->njik', question[indices],
                                 np.concatenate([candidate[indices], tail[indices]], axis=-1))
        interactions = interactions.reshape(len(indices), 4, -1)
        return np.concatenate([embeddings[indices], text[indices], votes[indices, :, None],
                               candidate_tfidf[indices], interactions], axis=-1)

    train = features(fitting)
    flat = train.reshape(-1, train.shape[-1])
    mean = flat.mean(axis=0)
    scale = np.maximum(flat.std(axis=0), np.float32(0.001))
    train = (train - mean) / scale
    weight = fit_all_pairs(train[hard[fitting]], labels[fitting][hard[fitting]],
                           valid[fitting][hard[fitting]], seed, 80, 128, 0.001, (80,))[80]
    return (((features(held) - mean) / scale) @ weight).squeeze(-1)


def normalized_scores(bank, embeddings, model):
    normalized = normalize_bank(bank)
    official, answers, votes, text = normalized_candidate_features(bank, normalized)
    reasoning = reasoning_features(normalized, answers)
    base = ranker_scores(embeddings, text, votes, model)
    stack = base + float(model['meta_weight']) * meta_scores(
        meta_features(embeddings, answers, votes, text, reasoning, base), model)
    group, _, _ = group_features(normalized, answers, base, reasoning)
    numeric = numeric_features(normalized, answers)
    if str(model['answer_variant'].item()) != 'number_only':
        raise ValueError('Kernel experiment requires the frozen number_only configuration')
    scores = (group_max_scores(stack, answers)
              + float(model['answer_weight']) * numeric_head_scores(
                  np.concatenate([group, numeric], axis=-1), model))
    return scores, official


def training_comparison(output_path, method):
    if output_path.exists():
        raise FileExistsError(f'Experiment output already exists: {output_path}')
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
    answers, votes, text = candidate_features(bank)
    reasoning = reasoning_features(bank, answers)
    valid = np.array([[value is not None for value in row] for row in answers], dtype=bool)
    hard = hard_mask(answers)
    numeric = numeric_features(bank, answers)

    normalized = normalize_bank(bank)
    _, _, normalized_votes, normalized_text = normalized_candidate_features(bank, normalized)
    if method not in ('kernel', 'text', 'hierarchical'):
        raise ValueError(f'Unknown complementary method: {method}')
    kernel_input = (np.concatenate([embeddings, normalized_text, normalized_votes[..., None]],
                                   axis=-1).astype(np.float32)
                    if method in ('kernel', 'hierarchical') else None)
    text_input = text_counts(bank) if method in ('text', 'hierarchical') else None
    del normalized
    n = len(bank)
    runs = []
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / 'fold.npz'
        for seed in SEEDS:
            folds = np.array_split(np.random.default_rng(seed).permutation(n), 3)
            variants = (('kernel_25', 'text_25', 'first_layer', 'hierarchical',
                         'baseline_75_hierarchical_25', 'baseline_50_hierarchical_50')
                        if method == 'hierarchical' else BLEND_WEIGHTS)
            choices = {key: np.zeros(n, dtype=np.int64) for key in (0.0, *variants)}
            for fold_id, held in enumerate(folds):
                fitting = np.concatenate([fold for i, fold in enumerate(folds) if i != fold_id])
                inner = np.array_split(np.random.default_rng(6800 + fold_id).permutation(len(fitting)), 2)
                fit_base, fit_stack = stacked_oof(fitting, inner, embeddings, text, votes, labels,
                                                  valid, hard, answers, reasoning)
                fit_answers = [answers[i] for i in fitting]
                group, reps, _ = feature_rows(fitting, fit_base, bank, answers, reasoning)
                head = fit_answer(np.concatenate([group, numeric[fitting]], axis=-1),
                                  group_max_scores(fit_stack, fit_answers), reps,
                                  labels[fitting], hard[fitting], 'number_only', 'mlp',
                                  selector=numeric_subset, residual_weight=0.5)
                meta_train = make_meta_rows(fitting, fit_base, embeddings, answers, votes, text, reasoning)
                meta = fit_meta(meta_train, labels[fitting], valid[fitting], hard[fitting],
                                STACK_MODE, STACK_EPOCHS)
                base = train_base(fitting, embeddings, text, votes, labels, valid, hard)
                np.savez(checkpoint, **base, **meta, **head,
                         meta_weight=np.float32(STACK_WEIGHT), answer_weight=np.float32(0.5))
                held_bank = [bank[i] for i in held]
                with np.load(checkpoint, allow_pickle=False) as fold_model:
                    old_scores, official = normalized_scores(held_bank, embeddings[held], fold_model)
                old_choices = select(old_scores, official)
                if old_choices != predict_normalized(held_bank, {'embeddings': embeddings[held]}, checkpoint):
                    raise ValueError('Baseline scores differ from the frozen normalization inference')
                choices[0.0][held] = old_choices
                if method in ('kernel', 'hierarchical'):
                    kernel = fit_kernel_ranker(kernel_input[fitting], labels[fitting], valid[fitting],
                                               hard[fitting], seed + fold_id)
                    kernel_result = kernel_scores(kernel_input[held], kernel)
                if method in ('text', 'hierarchical'):
                    text_result = text_fold_scores(text_input, embeddings, normalized_text,
                                                   normalized_votes, labels, valid, hard,
                                                   fitting, held, seed + fold_id)
                if method == 'hierarchical':
                    scores = hierarchical_scores(old_scores, kernel_result, text_result, valid[held])
                    for name, combined in scores.items():
                        choices[name][held] = select(combined, official)
                else:
                    alternative = kernel_result if method == 'kernel' else text_result
                    old_relative = relative_scores(old_scores, valid[held])
                    new_relative = relative_scores(alternative, valid[held])
                    for weight in BLEND_WEIGHTS:
                        choices[weight][held] = select(
                            (1 - weight) * old_relative + weight * new_relative, official)
                print(f'Completed {method} seed {seed}, fold {fold_id + 1}/3', flush=True)

            baseline = evaluate(choices[0.0], labels, answers)
            rows = np.arange(n)
            baseline_correct = labels[rows, choices[0.0]] > 0
            records = []
            for variant in variants:
                result = evaluate(choices[variant], labels, answers)
                correct = labels[rows, choices[variant]] > 0
                improvements = []
                for fold in folds:
                    improvements.append(int(correct[fold].sum() - baseline_correct[fold].sum()))
                config = {'variant': variant} if method == 'hierarchical' else {'complement_weight': variant}
                records.append({**config, **result,
                                'fold_improvements': improvements,
                                'baseline_only_correct': int((baseline_correct & ~correct).sum()),
                                'blend_only_correct': int((~baseline_correct & correct).sum())})
            runs.append({'seed': seed, 'baseline': baseline, 'blends': records})
    summary = {
        'data': 'training split only; matched question-level nested 3-fold/2-fold cross-fitting',
        'baseline': 'fold-trained numeric model with numeric-text normalization',
        'method': method,
        'ranker': 'all-pairs equal-question-weight Adam ranker',
        'training_epochs': 80, 'training_l2': 0.001,
        'caveat': 'inspired by the supplied summary, not an exact reproduction; reused folds and two seeds are not independent',
        'runs': runs, 'elapsed_seconds': time.perf_counter() - started,
    }
    if method in ('kernel', 'hierarchical'):
        summary.update({'kernel': 'Nyström RBF plus linear features',
                        'kernel_inputs': 'existing embeddings, normalized basic text features and answer votes; no new question-answer matching features',
                        'kernel_gamma': KERNEL_GAMMA, 'landmarks': 512,
                        'kernel_feature_weight': KERNEL_WEIGHT})
    if method in ('text', 'hierarchical'):
        summary.update({'text': 'hashed log-count TF-IDF with fold-fitted IDF and truncated SVD; question-candidate outer products',
                        'dimensions': {'hash': 768, 'question_svd': 16,
                                       'candidate_svd': 32, 'tail_svd': 32},
                        'tail_characters': 512,
                        'text_inputs': 'normalized question and candidate text plus existing embeddings, basic text features and votes; no new numerical answer-question matching features'})
    if method == 'hierarchical':
        summary.update({'fusion': 'first normalize and combine baseline 0.25 and kernel 0.75, normalize again, then combine with normalized text 0.25',
                        'additional_blends': 'combine standardized hierarchical scores with the normalized baseline at 0.25 or 0.50',
                        'baseline_role': 'existing normalized selector substitutes for the peer answer-group model'})
    else:
        summary['blend_weights'] = list(BLEND_WEIGHTS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=('kernel', 'text', 'hierarchical'), default='kernel')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or Path(f'runs/normalization/{args.method}-summary.json')
    training_comparison(output, args.method)


if __name__ == '__main__':
    main()
