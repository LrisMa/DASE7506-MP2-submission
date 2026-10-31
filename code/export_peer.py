"""Fit and export both preselected peer-inspired models using training data only."""
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler

from common import DATA, check_data, load_inputs
from submission import candidate_features
from submission_normalized import normalize_bank, normalized_candidate_features
from train_experiments import labels_for
from train_improved import hard_mask
from train_meta import train_base
from train_peer import (base_components, fit_numeric_vectorizer, fit_pairwise_logistic,
                        group_rows, peer_bottom_scores, scalar_features)


SEED = 7507


def bytes_array(text):
    return np.frombuffer(text.encode('utf-8'), dtype=np.uint8).copy()


def group_tree(embeddings, scalar, official, labels, fitting, oof, pca):
    reduced = pca.transform(embeddings.reshape(-1, embeddings.shape[-1])).reshape(len(embeddings), 4, -1)
    features, targets, groups, _ = group_rows(fitting, official, scalar, reduced, oof, labels)
    model = lgb.LGBMRanker(objective='lambdarank', n_estimators=300, num_leaves=7,
                           learning_rate=0.05, random_state=SEED, n_jobs=4, verbosity=-1)
    model.fit(features, targets, group=groups)
    return bytes_array(model.booster_.model_to_string())


def fit_bottom(embeddings, scalar, labels, valid, hard, solution_tfidf):
    raw = np.concatenate([embeddings, scalar], axis=-1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw.reshape(-1, raw.shape[-1]))
    linear = fit_pairwise_logistic(scaled, None, labels, valid, hard)
    candidate_ids, group_sizes = [], []
    for i in np.flatnonzero(hard):
        ids = np.flatnonzero(valid[i])
        if not len(ids):
            raise ValueError('Disagreement question has no valid candidates')
        candidate_ids.extend(4 * i + ids)
        group_sizes.append(len(ids))
    tree = lgb.LGBMRanker(objective='lambdarank', n_estimators=150, num_leaves=7,
                          learning_rate=0.05, random_state=SEED, n_jobs=4, verbosity=-1)
    tree.fit(scaled[candidate_ids], labels.reshape(-1)[candidate_ids], group=group_sizes)
    scalar_scaler = StandardScaler()
    scaled_scalar = scalar_scaler.fit_transform(scalar.reshape(-1, 65))
    text_weights = fit_pairwise_logistic(scaled_scalar, solution_tfidf, labels, valid, hard)
    return {'bottom_mean': scaler.mean_, 'bottom_scale': scaler.scale_,
            'bottom_linear': linear, 'bottom_tree': bytes_array(tree.booster_.model_to_string()),
            'bottom_scalar_mean': scalar_scaler.mean_, 'bottom_scalar_scale': scalar_scaler.scale_,
            'bottom_text_dense': text_weights[:65], 'bottom_text_sparse': text_weights[65:]}


def fit_kernel(embeddings, scalar, labels, valid, hard):
    raw = np.concatenate([embeddings, scalar], axis=-1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw.reshape(-1, raw.shape[-1]))
    kernel = Nystroem(kernel='rbf', gamma=4 / 1024, n_components=512,
                      random_state=SEED, n_jobs=4)
    basis = kernel.fit_transform(scaled)
    features = np.concatenate([scaled, 2 * basis], axis=1)
    weights = fit_pairwise_logistic(features, None, labels, valid, hard)
    dimension = raw.shape[-1]
    return {'kernel_mean': scaler.mean_, 'kernel_scale': scaler.scale_,
            'kernel_centers': kernel.components_, 'kernel_linear': weights[:dimension],
            'kernel_nonlinear': 2 * (kernel.normalization_.T @ weights[dimension:])}


def fit_text(questions, solutions, embeddings, scalar, labels, valid, hard):
    qvector, qtrain = fit_numeric_vectorizer(questions, 6000)
    svector, strain = fit_numeric_vectorizer(solutions, 20000)
    tails = svector.transform([text[-512:] for text in solutions])
    qsvd = TruncatedSVD(n_components=32, random_state=SEED)
    ssvd = TruncatedSVD(n_components=64, random_state=SEED)
    tsvd = TruncatedSVD(n_components=64, random_state=SEED)
    q = qsvd.fit_transform(qtrain)
    s = ssvd.fit_transform(strain)
    t = tsvd.fit_transform(tails)
    raw = np.concatenate([embeddings, scalar], axis=-1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw.reshape(-1, raw.shape[-1]))
    candidates = np.concatenate([s, t], axis=-1).reshape(len(questions), 4, 128)
    interactions = np.einsum('ni,njk->njik', q, candidates).reshape(-1, 4096)
    features = np.concatenate([scaled, interactions], axis=1)
    weights = fit_pairwise_logistic(features, strain, labels, valid, hard)
    dimension = raw.shape[-1]
    interaction_weights = weights[dimension:dimension + 4096].reshape(32, 128)
    return ({
        'question_vocabulary': bytes_array(json.dumps(qvector.vocabulary_, ensure_ascii=False)),
        'solution_vocabulary': bytes_array(json.dumps(svector.vocabulary_, ensure_ascii=False)),
        'question_idf': qvector.idf_, 'solution_idf': svector.idf_,
        'question_projection': qsvd.components_.T,
        'solution_projection': (ssvd.components_.T @ interaction_weights[:, :64].T).astype(np.float32),
        'tail_projection': (tsvd.components_.T @ interaction_weights[:, 64:].T).astype(np.float32),
        'text_mean': scaler.mean_, 'text_scale': scaler.scale_,
        'text_linear': weights[:dimension], 'text_sparse': weights[dimension + 4096:],
    }, strain)


def main():
    output = Path('runs/peer-test')
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    check_data(DATA, ('train',))
    bank, arrays = load_inputs('train')
    labels = labels_for(bank, 'train')
    embeddings = arrays['embeddings']
    official, _, _ = candidate_features(bank)
    valid = np.array([[value is not None for value in row] for row in official], dtype=bool)
    hard = hard_mask(official)
    normalized = normalize_bank(bank)
    _, answers, votes, text = normalized_candidate_features(bank, normalized)
    scalar = scalar_features(normalized, answers, votes, text)
    fitting = np.arange(len(bank))
    inner = np.array_split(np.random.default_rng(SEED + 8100).permutation(len(bank)), 2)
    existing_oof = np.zeros((len(bank), 4, 3), dtype=np.float32)
    peer_oof = np.zeros_like(existing_oof)
    for local_held in inner:
        mask = np.ones(len(bank), dtype=bool)
        mask[local_held] = False
        training = fitting[mask]
        base = train_base(training, embeddings, text, votes, labels, valid, hard)
        existing_oof[local_held] = base_components(embeddings[local_held], text[local_held],
                                                    votes[local_held], base)
        peer_oof[local_held] = peer_bottom_scores(normalized, embeddings, scalar, labels, valid,
                                                  hard, training, local_held, SEED)
    print('Both out-of-fold bottom score sets complete', flush=True)

    pca = PCA(n_components=16, svd_solver='randomized', random_state=SEED)
    pca.fit(embeddings.reshape(-1, embeddings.shape[-1]))
    existing_tree = group_tree(embeddings, scalar, official, labels, fitting, existing_oof, pca)
    peer_tree = group_tree(embeddings, scalar, official, labels, fitting, peer_oof, pca)
    existing_base = train_base(fitting, embeddings, text, votes, labels, valid, hard)
    questions = [row['question'] for row in normalized]
    solutions = [candidate['text'] for row in normalized for candidate in row['candidates']]
    text_assets, solution_tfidf = fit_text(questions, solutions, embeddings, scalar, labels, valid, hard)
    print('Shared text ranker complete', flush=True)
    bottom_assets = fit_bottom(embeddings, scalar, labels, valid, hard, solution_tfidf)
    print('New bottom rankers complete', flush=True)
    kernel_assets = fit_kernel(embeddings, scalar, labels, valid, hard)
    print('Shared kernel ranker complete', flush=True)
    with np.load('runs/numeric/selected.npz', allow_pickle=False) as checkpoint:
        old_assets = {'old_' + key: checkpoint[key] for key in checkpoint.files}
    common = {**old_assets, **text_assets, **kernel_assets,
              'pca_mean': pca.mean_, 'pca_components': pca.components_}
    existing = {**common, **{'a_base_' + key: value for key, value in existing_base.items()},
                'group_mode': np.array('existing'), 'group_tree': existing_tree}
    peer = {**common, **bottom_assets,
            'group_mode': np.array('peer-bottom'), 'group_tree': peer_tree}
    for name, assets in (('validated', existing), ('peer-bottom', peer)):
        size = sum(np.asarray(value).nbytes for value in assets.values())
        if size > 16 * 1024 * 1024:
            raise ValueError(f'{name} inference assets exceed 16 MiB: {size} bytes')
        np.savez_compressed(output / f'{name}.npz', **assets)
        print(f'{name}: {size} uncompressed asset bytes', flush=True)
    print(f'Both models trained from train data in {time.perf_counter() - started:.2f} seconds', flush=True)


if __name__ == '__main__':
    main()
