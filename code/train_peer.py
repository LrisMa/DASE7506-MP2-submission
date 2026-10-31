"""Training-only comparison of a closer peer-inspired ranking ensemble."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re
import tempfile
import time

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler

from common import DATA, check_data, load_inputs, score_choices
from reasoning_features import reasoning_features
from submission import candidate_features, select
from submission_answer import group_max_scores
from submission_improved import build_features, score_candidates
from submission_normalized import normalize_bank, normalized_candidate_features, predict as predict_normalized
from submission_numeric import feature_subset as numeric_subset, numeric_features
from train_answer import STACK_EPOCHS, STACK_MODE, STACK_WEIGHT, feature_rows, fit_answer, stacked_oof
from train_complement import hierarchical_scores, normalized_scores, relative_scores
from train_experiments import evaluate, labels_for
from train_improved import hard_mask
from train_meta import fit_meta, make_meta_rows, train_base


TOKEN = re.compile(
    r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?(?:%|st|nd|rd|th)?'
    r'|[a-z]+|[+\-*/=%]',
    re.IGNORECASE,
)
SYMBOLS = ('+', '-', '*', '/', '=', '%', '$', '(', ')', '[', ']', '{', '}', '.',
           ',', '\n', '?', '!', ':', ';', '\\', '^', '×', '÷', '<', '>', '|', '_')
WORDS = ('therefore', 'thus', 'answer', 'total', 'remaining', 'each', 'per', 'average',
         'divide', 'multiply', 'add', 'subtract', 'equals', 'calculate', 'approximately',
         'because', 'step', 'final', 'however', 'so')
SEEDS = (6813, 7253)


def tokens(text):
    return TOKEN.findall(text.lower())


def vectorizer(max_features=None, vocabulary=None):
    return TfidfVectorizer(tokenizer=tokens, token_pattern=None, ngram_range=(1, 2),
                           max_features=max_features, vocabulary=vocabulary,
                           sublinear_tf=True, dtype=np.float32)


def fit_numeric_vectorizer(documents, max_features):
    analyzer = vectorizer().build_analyzer()
    frequencies = Counter()
    for document in documents:
        frequencies.update(analyzer(document))
    numbers = sorted(token for token in frequencies if ' ' not in token and re.search(r'\d', token))
    if len(numbers) > max_features:
        raise ValueError(f'{len(numbers)} numeric tokens exceed vocabulary limit {max_features}')
    number_set = set(numbers)
    remaining = sorted((token for token in frequencies if token not in number_set),
                       key=lambda token: (-frequencies[token], token))
    vocabulary = {token: index for index, token in enumerate(
        numbers + remaining[:max_features - len(numbers)])}
    fitted = vectorizer(vocabulary=vocabulary)
    matrix = fitted.fit_transform(documents)
    return fitted, matrix


def scalar_features(bank, answers, votes, text):
    """Use formatting, calculation and answer-group attributes, not question-answer matching."""
    reasoning = reasoning_features(bank, answers)
    rows = np.zeros((len(bank), 4, 65), dtype=np.float32)
    for i, row in enumerate(bank):
        distinct = len({value for value in answers[i] if value is not None})
        max_vote = float(votes[i].max())
        mean_length = sum(len(candidate['text']) for candidate in row['candidates']) / 4
        for j, candidate in enumerate(row['candidates']):
            value = answers[i][j]
            number = Fraction(value) if value is not None else None
            magnitude = float(abs(number)) if number is not None and abs(number) < 10**15 else 0
            attributes = [float(number is not None and number.denominator == 1),
                          float(number is not None and number < 0),
                          float(number is not None and number == 0),
                          np.log1p(magnitude),
                          np.log1p(number.denominator) if number is not None else 0]
            solution = candidate['text']
            lower = solution.lower()
            features = [*text[i, j], votes[i, j],
                        *(np.log1p(solution.count(symbol)) for symbol in SYMBOLS),
                        *(np.log1p(len(re.findall(r'\b' + re.escape(word) + r'\b', lower)))
                          for word in WORDS),
                        *reasoning[i, j, [3, 4, 5]], *attributes,
                        distinct, max_vote, np.log1p(len(solution)) - np.log1p(mean_length)]
            if len(features) != 65:
                raise ValueError(f'Expected 65 scalar features, got {len(features)}')
            rows[i, j] = features
    return rows


def pair_indices(labels, valid, hard):
    positive, negative, weights = [], [], []
    for i in np.flatnonzero(hard):
        pos = np.flatnonzero((labels[i] > 0) & valid[i])
        neg = np.flatnonzero((labels[i] == 0) & valid[i])
        if not len(pos) or not len(neg):
            raise ValueError('Each disagreement question needs a valid positive and negative')
        weight = 1 / (len(pos) * len(neg))
        for p in pos:
            for n in neg:
                positive.append(4 * i + p)
                negative.append(4 * i + n)
                weights.append(weight)
    if not positive:
        raise ValueError('No valid training pairs')
    return np.asarray(positive), np.asarray(negative), np.asarray(weights)


def fit_pairwise_logistic(dense, sparse, labels, valid, hard, c=0.1):
    positive, negative, weights = pair_indices(labels, valid, hard)
    if dense.shape[0] != len(labels) * 4 or (sparse is not None and sparse.shape[0] != dense.shape[0]):
        raise ValueError('Pairwise features are not aligned with candidate groups')
    dense = np.asarray(dense, dtype=np.float64)
    sparse = sparse.astype(np.float64) if sparse is not None else None
    dimension = dense.shape[1] + (sparse.shape[1] if sparse is not None else 0)

    def objective(w):
        scores = dense @ w[:dense.shape[1]]
        if sparse is not None:
            scores += sparse @ w[dense.shape[1]:]
        margin = scores[positive] - scores[negative]
        derivative = -c * weights * expit(-margin)
        candidate_gradient = np.bincount(positive, weights=derivative, minlength=len(scores))
        candidate_gradient -= np.bincount(negative, weights=derivative, minlength=len(scores))
        gradient = np.empty_like(w)
        gradient[:dense.shape[1]] = w[:dense.shape[1]] + dense.T @ candidate_gradient
        if sparse is not None:
            gradient[dense.shape[1]:] = w[dense.shape[1]:] + sparse.T @ candidate_gradient
        loss = 0.5 * (w @ w) + c * np.dot(weights, np.logaddexp(0, -margin))
        return loss, gradient

    result = minimize(objective, np.zeros(dimension), method='L-BFGS-B', jac=True,
                      options={'maxiter': 600, 'ftol': 1e-8, 'gtol': 1e-5})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'Pairwise logistic optimization failed: {result.message}')
    print(f'Pairwise logistic converged in {result.nit} iterations', flush=True)
    return result.x


def score_pairwise(dense, sparse, weights):
    scores = dense @ weights[:dense.shape[1]]
    if sparse is not None:
        scores += sparse @ weights[dense.shape[1]:]
    return np.asarray(scores).reshape(-1, 4)


def kernel_fold(embeddings, scalar, labels, valid, hard, fitting, held, seed):
    raw = np.concatenate([embeddings, scalar], axis=-1)
    train = raw[fitting].reshape(-1, raw.shape[-1])
    test = raw[held].reshape(-1, raw.shape[-1])
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    kernel = Nystroem(kernel='rbf', gamma=4 / 1024, n_components=512, random_state=seed,
                      n_jobs=4)
    kernel_train = kernel.fit_transform(train)
    kernel_test = kernel.transform(test)
    train = np.concatenate([train, 2 * kernel_train], axis=1)
    test = np.concatenate([test, 2 * kernel_test], axis=1)
    weights = fit_pairwise_logistic(train, None, labels[fitting], valid[fitting], hard[fitting])
    return score_pairwise(test, None, weights)


def text_fold(bank, embeddings, scalar, labels, valid, hard, fitting, held, seed):
    question = [row['question'] for row in bank]
    candidates = [candidate['text'] for row in bank for candidate in row['candidates']]
    tails = [text[-512:] for text in candidates]
    train_candidates = np.arange(len(bank) * 4).reshape(-1, 4)[fitting].reshape(-1)
    held_candidates = np.arange(len(bank) * 4).reshape(-1, 4)[held].reshape(-1)
    qvector, qtrain = fit_numeric_vectorizer([question[i] for i in fitting], 6000)
    svector, strain = fit_numeric_vectorizer([candidates[i] for i in train_candidates], 20000)
    qheld = qvector.transform([question[i] for i in held])
    sheld = svector.transform([candidates[i] for i in held_candidates])
    tail_train = svector.transform([tails[i] for i in train_candidates])
    tail_held = svector.transform([tails[i] for i in held_candidates])
    qsvd = TruncatedSVD(n_components=32, random_state=seed)
    ssvd = TruncatedSVD(n_components=64, random_state=seed)
    tsvd = TruncatedSVD(n_components=64, random_state=seed)
    qtrain_reduced, qheld_reduced = qsvd.fit_transform(qtrain), qsvd.transform(qheld)
    strain_reduced, sheld_reduced = ssvd.fit_transform(strain), ssvd.transform(sheld)
    tail_train_reduced, tail_held_reduced = tsvd.fit_transform(tail_train), tsvd.transform(tail_held)
    raw = np.concatenate([embeddings, scalar], axis=-1)
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(raw[fitting].reshape(-1, raw.shape[-1]))
    xheld = scaler.transform(raw[held].reshape(-1, raw.shape[-1]))

    def interactions(q, s, tail):
        candidate = np.concatenate([s, tail], axis=-1).reshape(len(q), 4, 128)
        return np.einsum('ni,njk->njik', q, candidate).reshape(-1, 4096)

    xtrain = np.concatenate([xtrain, interactions(qtrain_reduced, strain_reduced, tail_train_reduced)], axis=1)
    xheld = np.concatenate([xheld, interactions(qheld_reduced, sheld_reduced, tail_held_reduced)], axis=1)
    weights = fit_pairwise_logistic(xtrain, strain, labels[fitting], valid[fitting], hard[fitting])
    return score_pairwise(xheld, sheld, weights)


def group_rows(indices, official, scalar, pca, base_scores, labels=None):
    features, targets, groups, locations = [], [], [], []
    for local, row in enumerate(indices):
        values = official[row]
        members_by_answer = {}
        for candidate, value in enumerate(values):
            if value is not None:
                members_by_answer.setdefault(value, []).append(candidate)
        if len(members_by_answer) < 2:
            continue
        groups.append(len(members_by_answer))
        for members in members_by_answer.values():
            members = np.asarray(members)
            member_features = np.concatenate([scalar[row, members], pca[row, members]], axis=1)
            member_scores = base_scores[local, members]
            features.append(np.concatenate([member_features.mean(axis=0), member_features.min(axis=0),
                                            member_features.max(axis=0),
                                            [len(members)], member_scores.mean(axis=0),
                                            member_scores.min(axis=0), member_scores.max(axis=0)]))
            if labels is not None:
                targets.append(int(labels[row, members].max()))
            locations.append((local, members))
    if not features:
        raise ValueError('No parsable disagreement answer groups')
    if labels is not None:
        offset = 0
        for count in groups:
            if sum(targets[offset:offset + count]) != 1:
                raise ValueError('Each disagreement question needs one correct answer group')
            offset += count
    return np.asarray(features, dtype=np.float32), np.asarray(targets), groups, locations


def base_components(embeddings, text, votes, model):
    features = build_features(embeddings, text, votes, str(model['mode'].item()))
    neural = score_candidates(features, model)
    normalized = (embeddings - model['baseline_mean']) / model['baseline_scale']
    linear = (normalized @ model['baseline_w'] + model['baseline_b']).squeeze(-1)
    ranked = (embeddings - model['rank_mean']) / model['rank_scale']
    pairwise = (ranked @ model['rank_w']).squeeze(-1)
    return np.stack([linear, neural, pairwise], axis=-1)


def fit_group_scores(embeddings, scalar, official, labels, fitting, held, oof, held_base, seed):
    pca = PCA(n_components=16, svd_solver='randomized', random_state=seed)
    pca.fit(embeddings[fitting].reshape(-1, embeddings.shape[-1]))
    reduced = pca.transform(embeddings.reshape(-1, embeddings.shape[-1])).reshape(len(embeddings), 4, -1)
    xtrain, target, groups, _ = group_rows(fitting, official, scalar, reduced, oof, labels)
    xheld, _, _, locations = group_rows(held, official, scalar, reduced, held_base)
    model = lgb.LGBMRanker(objective='lambdarank', n_estimators=300, num_leaves=7,
                           learning_rate=0.05, random_state=seed, n_jobs=4, verbosity=-1)
    model.fit(xtrain, target, group=groups)
    values = model.predict(xheld)
    result = np.zeros((len(held), 4), dtype=np.float32)
    for value, (local, members) in zip(values, locations):
        result[local, members] = value
    return result


def group_fold(embeddings, scalar, official, votes, text, labels, valid, hard, fitting, held, seed):
    inner = np.array_split(np.random.default_rng(seed + 8100).permutation(len(fitting)), 2)
    oof = np.zeros((len(fitting), 4, 3), dtype=np.float32)
    for local_held in inner:
        mask = np.ones(len(fitting), dtype=bool)
        mask[local_held] = False
        training, validation = fitting[mask], fitting[local_held]
        base = train_base(training, embeddings, text, votes, labels, valid, hard)
        oof[local_held] = base_components(embeddings[validation], text[validation],
                                          votes[validation], base)
    base = train_base(fitting, embeddings, text, votes, labels, valid, hard)
    held_base = base_components(embeddings[held], text[held], votes[held], base)
    return fit_group_scores(embeddings, scalar, official, labels, fitting, held, oof, held_base, seed)


def peer_bottom_scores(bank, embeddings, scalar, labels, valid, hard, fitting, held, seed):
    raw = np.concatenate([embeddings, scalar], axis=-1)
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(raw[fitting].reshape(-1, raw.shape[-1]))
    xheld = scaler.transform(raw[held].reshape(-1, raw.shape[-1]))
    linear_weights = fit_pairwise_logistic(xtrain, None, labels[fitting], valid[fitting], hard[fitting])
    linear = score_pairwise(xheld, None, linear_weights)

    candidate_ids, group_sizes = [], []
    for local in np.flatnonzero(hard[fitting]):
        valid_ids = np.flatnonzero(valid[fitting[local]])
        if not len(valid_ids):
            raise ValueError('Disagreement question has no valid candidates')
        candidate_ids.extend(4 * local + valid_ids)
        group_sizes.append(len(valid_ids))
    tree = lgb.LGBMRanker(objective='lambdarank', n_estimators=150, num_leaves=7,
                          learning_rate=0.05, random_state=seed, n_jobs=4, verbosity=-1)
    tree.fit(xtrain[candidate_ids], labels[fitting].reshape(-1)[candidate_ids], group=group_sizes)
    tree_scores = tree.predict(xheld).reshape(-1, 4)

    solutions = [candidate['text'] for row in bank for candidate in row['candidates']]
    training_ids = np.arange(len(bank) * 4).reshape(-1, 4)[fitting].reshape(-1)
    held_ids = np.arange(len(bank) * 4).reshape(-1, 4)[held].reshape(-1)
    words, train_words = fit_numeric_vectorizer([solutions[i] for i in training_ids], 20000)
    held_words = words.transform([solutions[i] for i in held_ids])
    scalar_scaler = StandardScaler()
    train_scalar = scalar_scaler.fit_transform(scalar[fitting].reshape(-1, 65))
    held_scalar = scalar_scaler.transform(scalar[held].reshape(-1, 65))
    text_weights = fit_pairwise_logistic(train_scalar, train_words,
                                         labels[fitting], valid[fitting], hard[fitting])
    textual = score_pairwise(held_scalar, held_words, text_weights)
    return np.stack([linear, tree_scores, textual], axis=-1)


def group_fold_peer(bank, embeddings, scalar, official, labels, valid, hard, fitting, held, seed):
    inner = np.array_split(np.random.default_rng(seed + 8100).permutation(len(fitting)), 2)
    oof = np.zeros((len(fitting), 4, 3), dtype=np.float32)
    for local_held in inner:
        mask = np.ones(len(fitting), dtype=bool)
        mask[local_held] = False
        training, validation = fitting[mask], fitting[local_held]
        oof[local_held] = peer_bottom_scores(bank, embeddings, scalar, labels, valid, hard,
                                             training, validation, seed)
    held_base = peer_bottom_scores(bank, embeddings, scalar, labels, valid, hard, fitting, held, seed)
    return fit_group_scores(embeddings, scalar, official, labels, fitting, held, oof, held_base, seed)


def training_comparison(output_path, seeds, group_mode):
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
    official, votes, text = candidate_features(bank)
    reasoning = reasoning_features(bank, official)
    valid = np.array([[value is not None for value in row] for row in official], dtype=bool)
    hard = hard_mask(official)
    numeric = numeric_features(bank, official)
    normalized = normalize_bank(bank)
    _, answers, normalized_votes, normalized_text = normalized_candidate_features(bank, normalized)
    scalar = scalar_features(normalized, answers, normalized_votes, normalized_text)
    n = len(bank)
    runs = []
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / 'fold.npz'
        for seed in seeds:
            folds = np.array_split(np.random.default_rng(seed).permutation(n), 3)
            names = ('baseline', 'group', 'kernel', 'text', 'peer_structure',
                     'baseline_75_peer_25', 'baseline_50_peer_50')
            choices = {name: np.zeros(n, dtype=np.int64) for name in names}
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

                if group_mode == 'peer-bottom':
                    group_result = group_fold_peer(normalized, embeddings, scalar, official,
                                                   labels, valid, hard, fitting, held, seed + fold_id)
                else:
                    group_result = group_fold(embeddings, scalar, official, normalized_votes,
                                              normalized_text, labels, valid, hard, fitting, held, seed + fold_id)
                print(f'Seed {seed} fold {fold_id + 1}: group complete', flush=True)
                kernel_result = kernel_fold(embeddings, scalar, labels, valid, hard, fitting, held, seed + fold_id)
                print(f'Seed {seed} fold {fold_id + 1}: kernel complete', flush=True)
                text_result = text_fold(normalized, embeddings, scalar, labels, valid, hard,
                                        fitting, held, seed + fold_id)
                print(f'Seed {seed} fold {fold_id + 1}: text complete', flush=True)
                for name, result in (('group', group_result), ('kernel', kernel_result), ('text', text_result)):
                    choices[name][held] = select(result, held_official)
                peer = hierarchical_scores(group_result, kernel_result, text_result, valid[held])['hierarchical']
                choices['peer_structure'][held] = select(peer, held_official)
                old_relative = relative_scores(old, valid[held])
                peer_relative = relative_scores(peer, valid[held])
                for weight in (0.25, 0.5):
                    name = 'baseline_75_peer_25' if weight == 0.25 else 'baseline_50_peer_50'
                    choices[name][held] = select((1 - weight) * old_relative + weight * peer_relative, held_official)

            baseline = evaluate(choices['baseline'], labels, official)
            correct_old = labels[np.arange(n), choices['baseline']] > 0
            results = []
            for name in names[1:]:
                result = evaluate(choices[name], labels, official)
                correct = labels[np.arange(n), choices[name]] > 0
                results.append({'variant': name, **result,
                                'fold_improvements': [int(correct[fold].sum() - correct_old[fold].sum())
                                                      for fold in folds],
                                'baseline_only_correct': int((correct_old & ~correct).sum()),
                                'variant_only_correct': int((~correct_old & correct).sum())})
            runs.append({'seed': seed, 'baseline': baseline, 'variants': results})
    summary = {
        'data': 'training split only; matched question-level nested 3-fold/2-fold cross-fitting',
        'method': 'peer-inspired group LambdaRank, Nyström pairwise logistic, TF-IDF/SVD interaction pairwise logistic',
        'normalization': 'questions and candidates normalized in memory; vocabulary reserves each distinct numeric unigram in the fitting fold; unseen numeric tokens can be out of vocabulary; embeddings unchanged',
        'scalar_features': '65 formatting, equation-checking and answer-group attributes; no new question-answer numeric-matching features',
        'group': ('300-tree, 7-leaf LambdaRank; 16-dimensional PCA and out-of-fold pairwise linear, candidate LambdaRank and TF-IDF pairwise scores'
                  if group_mode == 'peer-bottom' else
                  '300-tree, 7-leaf LambdaRank; 16-dimensional PCA and three out-of-fold component scores (linear, neural, pairwise)'),
        'group_mode': group_mode,
        'kernel': '512 landmarks, RBF gamma 4/1024, feature weight 2, pairwise logistic C=0.1',
        'text': 'question TF-IDF up to 6000, candidate word/bigram TF-IDF up to 20000; SVD 32/64/64; 4096 interactions; pairwise logistic C=0.1',
        'fusion': 'group 25% + kernel 75%, renormalized, then 75% of this + 25% text; baseline hybrids also assessed',
        'caveat': 'not an exact reproduction: peer source, checkpoint and full feature definitions were not supplied; seeds reuse training data',
        'runs': runs, 'elapsed_seconds': time.perf_counter() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def validation_comparison(output_path, seed, group_mode):
    if output_path.exists():
        raise FileExistsError(f'Experiment output already exists: {output_path}')
    started = time.perf_counter()
    check_data(DATA, ('train', 'validation'))
    train_bank, train_arrays = load_inputs('train')
    train_labels = labels_for(train_bank, 'train')
    validation_bank, validation_arrays = load_inputs('validation')
    bank = train_bank + validation_bank
    embeddings = np.concatenate([train_arrays['embeddings'], validation_arrays['embeddings']])
    labels = np.concatenate([train_labels, np.zeros((len(validation_bank), 4), dtype=np.float32)])
    fitting = np.arange(len(train_bank))
    held = np.arange(len(train_bank), len(bank))
    official, _, _ = candidate_features(bank)
    valid = np.array([[value is not None for value in row] for row in official], dtype=bool)
    hard = hard_mask(official)
    normalized = normalize_bank(bank)
    _, answers, votes, text = normalized_candidate_features(bank, normalized)
    scalar = scalar_features(normalized, answers, votes, text)
    with np.load('runs/numeric/selected.npz', allow_pickle=False) as baseline_model:
        old, held_official = normalized_scores(validation_bank, embeddings[held], baseline_model)
    baseline = select(old, held_official)
    if baseline != predict_normalized(validation_bank, {'embeddings': embeddings[held]},
                                      Path('runs/numeric/selected.npz')):
        raise ValueError('Baseline scores differ from normalized inference')
    if group_mode == 'peer-bottom':
        group = group_fold_peer(normalized, embeddings, scalar, official, labels, valid, hard, fitting, held, seed)
    else:
        group = group_fold(embeddings, scalar, official, votes, text, labels, valid, hard, fitting, held, seed)
    print('Validation: group complete', flush=True)
    kernel = kernel_fold(embeddings, scalar, labels, valid, hard, fitting, held, seed)
    print('Validation: kernel complete', flush=True)
    textual = text_fold(normalized, embeddings, scalar, labels, valid, hard, fitting, held, seed)
    print('Validation: text complete', flush=True)
    peer = hierarchical_scores(group, kernel, textual, valid[held])['hierarchical']
    old_relative = relative_scores(old, valid[held])
    peer_relative = relative_scores(peer, valid[held])
    choices = {
        'baseline': baseline,
        'peer_structure': select(peer, held_official),
        'baseline_75_peer_25': select(0.75 * old_relative + 0.25 * peer_relative, held_official),
        'baseline_50_peer_50': select(0.50 * old_relative + 0.50 * peer_relative, held_official),
    }
    validation_labels = labels_for(validation_bank, 'validation')
    baseline_correct = validation_labels[np.arange(len(held)), choices['baseline']] > 0
    results = []
    for name, selected in choices.items():
        correct = validation_labels[np.arange(len(held)), selected] > 0
        metrics = evaluate(selected, validation_labels, held_official)
        official_metrics = score_choices('validation', [row['id'] for row in validation_bank], selected)
        if any(metrics[key] != official_metrics[key] for key in ('correct', 'n', 'accuracy')):
            raise ValueError(f'Official validation score mismatch for {name}')
        results.append({'variant': name, **metrics,
                        'baseline_only_correct': int((baseline_correct & ~correct).sum()),
                        'variant_only_correct': int((~baseline_correct & correct).sum())})
    summary = {
        'data': 'fit on training split only; validation labels loaded after prediction',
        'seed': seed,
        'group_mode': group_mode,
        'caveat': 'validation and training folds have previously been used for development; this is exploratory, not an independent test',
        'method': 'numeric-preserving peer-inspired ensemble; baseline checkpoint unchanged',
        'scoring': 'common.score_choices verified against development-label metrics',
        'results': results, 'elapsed_seconds': time.perf_counter() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--group-mode', choices=('existing', 'peer-bottom'), default='existing')
    args = parser.parse_args()
    if args.validation:
        if len(args.seeds) != 1:
            parser.error('--validation requires exactly one --seeds value')
        output = args.output or Path('runs/normalization/peer-validation-summary.json')
        validation_comparison(output, args.seeds[0], args.group_mode)
    else:
        output = args.output or Path('runs/normalization/peer-numeric-summary.json')
        training_comparison(output, args.seeds, args.group_mode)


if __name__ == '__main__':
    main()
