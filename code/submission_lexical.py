"""Pairwise embedding ranker combined with a compact lexical ranker."""
import re
import zlib

import numpy as np

from submission import candidate_features, select


TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)?|[+\-*/=%]", re.IGNORECASE)
DIMENSION = 768


def lexical_features(bank):
    features = np.zeros((len(bank), 4, DIMENSION), dtype=np.float32)
    for i, row in enumerate(bank):
        for j, candidate in enumerate(row['candidates']):
            tokens = [('#' if token[0].isdigit() else token.lower())
                      for token in TOKEN.findall(candidate['text'])]
            for token in tokens:
                index = zlib.crc32(token.encode('utf-8')) % DIMENSION
                features[i, j, index] += 1
            np.log1p(features[i, j], out=features[i, j])
    return features


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('The lexical selector requires a trained checkpoint')
    if not bank:
        return []
    answers, votes, _ = candidate_features(bank)
    text = lexical_features(bank)
    with np.load(checkpoint, allow_pickle=False) as model:
        embedding = (arrays['embeddings'] - model['embedding_mean']) / model['embedding_scale']
        lexical = (text - model['lexical_mean']) / model['lexical_scale']
        scores = ((embedding @ model['embedding_w']).squeeze(-1) + 0.5 * votes
                  + float(model['blend']) * (lexical @ model['lexical_w']).squeeze(-1))
    return select(scores, answers)
