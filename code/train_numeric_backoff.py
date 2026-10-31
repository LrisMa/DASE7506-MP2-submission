"""Training-only numeric character backoff for the peer text ranker."""
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler

from train_peer import fit_numeric_vectorizer, fit_pairwise_logistic, score_pairwise, tokens


ALPHABET = '0123456789-+./%stndrhe'
POSITIONS = 6
LENGTH_BUCKETS = 13
NUMERIC_DIMENSIONS = 2 * POSITIONS * len(ALPHABET) + LENGTH_BUCKETS
BACKOFF_WEIGHT = 0.25
NUMBER = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?'
NUMERIC_TOKEN = re.compile(
    rf'(?<![\w.])[-+]?(?:{NUMBER}|\.\d+(?:[eE][+-]?\d+)?)(?:/{NUMBER})?'
    r'(?:%|st|nd|rd|th)?(?![\w%]|,\d|\.\d|/\d)',
    re.IGNORECASE,
)


def numeric_counts(documents):
    rows, columns = [], []
    for row, document in enumerate(documents):
        for match in NUMERIC_TOKEN.finditer(document):
            token = match.group().replace(',', '').lower()
            for position in range(min(POSITIONS, len(token))):
                for side, character in enumerate((token[position], token[-position - 1])):
                    index = ALPHABET.find(character)
                    if index < 0:
                        raise ValueError(f'Unsupported numeric token character: {character!r}')
                    rows.append(row)
                    columns.append((side * POSITIONS + position) * len(ALPHABET) + index)
            rows.append(row)
            columns.append(2 * POSITIONS * len(ALPHABET) + min(len(token), LENGTH_BUCKETS) - 1)
    values = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((values, (rows, columns)), shape=(len(documents), NUMERIC_DIMENSIONS))


def unseen_numeric_tokens(text, vocabulary):
    return any(re.search(r'\d', token) and token not in vocabulary for token in tokens(text))


def text_fold_numeric(bank, embeddings, scalar, labels, valid, hard, fitting, held, seed):
    questions = [row['question'] for row in bank]
    candidates = [candidate['text'] for row in bank for candidate in row['candidates']]
    tails = [text[-512:] for text in candidates]
    train_candidates = np.arange(len(bank) * 4).reshape(-1, 4)[fitting].reshape(-1)
    held_candidates = np.arange(len(bank) * 4).reshape(-1, 4)[held].reshape(-1)
    train_questions = [questions[i] for i in fitting]
    held_questions = [questions[i] for i in held]
    train_solutions = [candidates[i] for i in train_candidates]
    held_solutions = [candidates[i] for i in held_candidates]

    qvector, qtrain = fit_numeric_vectorizer(train_questions, 6000)
    svector, strain = fit_numeric_vectorizer(train_solutions, 20000)
    qheld = qvector.transform(held_questions)
    sheld = svector.transform(held_solutions)
    tail_train = svector.transform([tails[i] for i in train_candidates])
    tail_held = svector.transform([tails[i] for i in held_candidates])

    question_backoff = TfidfTransformer(sublinear_tf=True)
    candidate_backoff = TfidfTransformer(sublinear_tf=True)
    qtrain_extra = question_backoff.fit_transform(numeric_counts(train_questions))
    qheld_extra = question_backoff.transform(numeric_counts(held_questions))
    strain_extra = candidate_backoff.fit_transform(numeric_counts(train_solutions))
    sheld_extra = candidate_backoff.transform(numeric_counts(held_solutions))
    tail_train_extra = candidate_backoff.transform(numeric_counts([tails[i] for i in train_candidates]))
    tail_held_extra = candidate_backoff.transform(numeric_counts([tails[i] for i in held_candidates]))
    qtrain = hstack([qtrain, BACKOFF_WEIGHT * qtrain_extra], format='csr')
    qheld = hstack([qheld, BACKOFF_WEIGHT * qheld_extra], format='csr')
    strain = hstack([strain, BACKOFF_WEIGHT * strain_extra], format='csr')
    sheld = hstack([sheld, BACKOFF_WEIGHT * sheld_extra], format='csr')
    tail_train = hstack([tail_train, BACKOFF_WEIGHT * tail_train_extra], format='csr')
    tail_held = hstack([tail_held, BACKOFF_WEIGHT * tail_held_extra], format='csr')

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

    def interactions(question, solution, tail):
        candidate = np.concatenate([solution, tail], axis=-1).reshape(len(question), 4, 128)
        return np.einsum('ni,njk->njik', question, candidate).reshape(-1, 4096)

    xtrain = np.concatenate([xtrain, interactions(qtrain_reduced, strain_reduced, tail_train_reduced)], axis=1)
    xheld = np.concatenate([xheld, interactions(qheld_reduced, sheld_reduced, tail_held_reduced)], axis=1)
    weights = fit_pairwise_logistic(xtrain, strain, labels[fitting], valid[fitting], hard[fitting])
    unseen = np.array([
        unseen_numeric_tokens(questions[i], qvector.vocabulary_)
        or any(unseen_numeric_tokens(candidate['text'], svector.vocabulary_)
               for candidate in bank[i]['candidates'])
        for i in held
    ], dtype=bool)
    return score_pairwise(xheld, sheld, weights), unseen
