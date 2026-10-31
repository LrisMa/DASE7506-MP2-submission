"""Inference for a frozen normalized baseline plus peer-inspired rankers."""
import json

import lightgbm as lgb
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import rbf_kernel

from submission import select
from submission_normalized import normalize_bank, normalized_candidate_features
from train_complement import hierarchical_scores, normalized_scores, relative_scores
from train_peer import base_components, scalar_features, tokens


def decode_bytes(value):
    return value.tobytes().decode('utf-8')


def transform_text(documents, vocabulary, idf):
    counter = CountVectorizer(vocabulary=vocabulary, tokenizer=tokens, token_pattern=None,
                              ngram_range=(1, 2), dtype=np.float32)
    transformer = TfidfTransformer(sublinear_tf=True)
    transformer.idf_ = idf
    return transformer.transform(counter.transform(documents))


def group_scores(official, scalar, embeddings, bottom, model):
    reduced = ((embeddings - model['pca_mean']) @ model['pca_components'].T)
    rows, locations = [], []
    for i, answers in enumerate(official):
        groups = {}
        for j, answer in enumerate(answers):
            if answer is not None:
                groups.setdefault(answer, []).append(j)
        if len(groups) < 2:
            continue
        for members in groups.values():
            members = np.asarray(members)
            member_features = np.concatenate([scalar[i, members], reduced[i, members]], axis=1)
            member_scores = bottom[i, members]
            rows.append(np.concatenate([member_features.mean(axis=0), member_features.min(axis=0),
                                        member_features.max(axis=0), [len(members)],
                                        member_scores.mean(axis=0), member_scores.min(axis=0),
                                        member_scores.max(axis=0)]))
            locations.append((i, members))
    result = np.zeros((len(official), 4), dtype=np.float32)
    if rows:
        tree = lgb.Booster(model_str=decode_bytes(model['group_tree']))
        values = tree.predict(np.asarray(rows, dtype=np.float32), num_threads=4)
        for value, (i, members) in zip(values, locations):
            result[i, members] = value
    return result


def predict_scores(bank, embeddings, model):
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
    tail_tfidf = transform_text([text[-512:] for text in solutions], vocabulary, model['solution_idf'])
    question_tfidf = transform_text(questions, question_vocabulary, model['question_idf'])

    mode = str(model['group_mode'].item())
    if mode == 'existing':
        base = {key[7:]: value for key, value in model.items() if key.startswith('a_base_')}
        bottom = base_components(embeddings, text, votes, base)
    elif mode == 'peer-bottom':
        scaled = (raw - model['bottom_mean']) / model['bottom_scale']
        linear = (scaled @ model['bottom_linear']).reshape(-1, 4)
        tree = lgb.Booster(model_str=decode_bytes(model['bottom_tree']))
        tree_values = tree.predict(scaled.reshape(-1, scaled.shape[-1]), num_threads=4).reshape(-1, 4)
        scaled_scalar = (scalar - model['bottom_scalar_mean']) / model['bottom_scalar_scale']
        textual = (scaled_scalar.reshape(-1, 65) @ model['bottom_text_dense']
                   + solution_tfidf @ model['bottom_text_sparse']).reshape(-1, 4)
        bottom = np.stack([linear, tree_values, textual], axis=-1)
    else:
        raise ValueError(f'Unsupported group mode: {mode}')
    group = group_scores(official, scalar, embeddings, bottom, model)

    kernel_input = ((raw - model['kernel_mean']) / model['kernel_scale']).reshape(-1, raw.shape[-1])
    basis = rbf_kernel(kernel_input, model['kernel_centers'], gamma=4 / 1024)
    kernel = (kernel_input @ model['kernel_linear'] + basis @ model['kernel_nonlinear']).reshape(-1, 4)

    text_input = ((raw - model['text_mean']) / model['text_scale']).reshape(-1, raw.shape[-1])
    question_projection = question_tfidf @ model['question_projection']
    candidate_projection = (solution_tfidf @ model['solution_projection']
                            + tail_tfidf @ model['tail_projection'])
    interaction = np.sum(np.repeat(question_projection, 4, axis=0) * candidate_projection, axis=1)
    textual = (text_input @ model['text_linear'] + solution_tfidf @ model['text_sparse']
               + interaction).reshape(-1, 4)

    old_model = {key[4:]: value for key, value in model.items() if key.startswith('old_')}
    old, original = normalized_scores(bank, embeddings, old_model)
    if original != official:
        raise ValueError('Official answer parsing differs between ensemble branches')
    peer = hierarchical_scores(group, kernel, textual, valid)['hierarchical']
    scores = 0.75 * relative_scores(old, valid) + 0.25 * relative_scores(peer, valid)
    return scores, official


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('The peer-inspired selector requires a checkpoint')
    if not bank:
        return []
    with np.load(checkpoint, allow_pickle=False) as checkpoint_data:
        model = {key: checkpoint_data[key] for key in checkpoint_data.files}
    scores, official = predict_scores(bank, arrays['embeddings'], model)
    return select(scores, official)
