from __future__ import annotations

from collections import Counter
from hashlib import blake2b
import math

import numpy as np

from .text import STOPWORDS, Occurrence


def build_vocab(
    sentences: list[list[str]],
    max_vocab: int = 5000,
    min_count: int = 1,
) -> tuple[dict[str, int], Counter[str]]:
    counts: Counter[str] = Counter()
    for sentence in sentences:
        counts.update(token for token in sentence if token not in STOPWORDS)

    most_common = [
        token
        for token, count in counts.most_common(max_vocab)
        if count >= min_count and len(token) > 1
    ]
    return {token: i for i, token in enumerate(most_common)}, counts


def cooccurrence_matrix(
    sentences: list[list[str]],
    vocab: dict[str, int],
    window: int = 5,
) -> np.ndarray:
    size = len(vocab)
    matrix = np.zeros((size, size), dtype=np.float64)

    for sentence in sentences:
        indices = [vocab.get(token) for token in sentence]
        for pos, center in enumerate(indices):
            if center is None:
                continue

            start = max(0, pos - window)
            end = min(len(indices), pos + window + 1)
            for ctx_pos in range(start, end):
                if ctx_pos == pos:
                    continue
                context = indices[ctx_pos]
                if context is None:
                    continue
                distance = abs(ctx_pos - pos)
                matrix[center, context] += 1.0 / distance

    return matrix


def ppmi(matrix: np.ndarray) -> np.ndarray:
    total = matrix.sum()
    if total == 0:
        return matrix

    row_sum = matrix.sum(axis=1, keepdims=True)
    col_sum = matrix.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((matrix * total) / expected)
    pmi[~np.isfinite(pmi)] = 0.0
    pmi[pmi < 0.0] = 0.0
    return pmi


def train_word_embeddings(
    sentences: list[list[str]],
    dim: int = 64,
    max_vocab: int = 5000,
    min_count: int = 1,
    window: int = 5,
) -> tuple[dict[str, int], np.ndarray]:
    vocab, _ = build_vocab(sentences, max_vocab=max_vocab, min_count=min_count)
    if not vocab:
        raise ValueError("The corpus produced an empty vocabulary.")

    matrix = ppmi(cooccurrence_matrix(sentences, vocab, window=window))
    effective_dim = max(2, min(dim, min(matrix.shape) - 1))
    if effective_dim <= 0:
        return vocab, np.zeros((len(vocab), dim), dtype=np.float64)

    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    embeddings = u[:, :effective_dim] * np.sqrt(singular_values[:effective_dim])

    if effective_dim < dim:
        pad = np.zeros((embeddings.shape[0], dim - effective_dim), dtype=np.float64)
        embeddings = np.hstack([embeddings, pad])

    return vocab, l2_normalize(embeddings)


def occurrence_vectors(
    occurrences: list[Occurrence],
    vocab: dict[str, int],
    word_embeddings: np.ndarray,
    dim: int,
) -> np.ndarray:
    vectors = np.zeros((len(occurrences), dim), dtype=np.float64)

    for row, occurrence in enumerate(occurrences):
        pieces: list[np.ndarray] = []
        weights: list[float] = []
        lexical = np.zeros(dim, dtype=np.float64)

        for side, tokens in (("left", occurrence.left), ("right", occurrence.right)):
            token_count = len(tokens)
            for offset, token in enumerate(tokens):
                index = vocab.get(token)

                if side == "left":
                    distance = token_count - offset
                else:
                    distance = offset + 1
                weight = 1.0 / math.sqrt(distance)

                if index is not None:
                    pieces.append(word_embeddings[index])
                    weights.append(weight)
                if token not in STOPWORDS and len(token) > 1:
                    feature_index, sign = hashed_feature(token, dim)
                    lexical[feature_index] += sign * weight

        if pieces:
            stacked = np.vstack(pieces)
            semantic = np.average(stacked, axis=0, weights=np.array(weights))
        else:
            semantic = np.zeros(dim, dtype=np.float64)

        semantic = normalize_vector(semantic)
        lexical = normalize_vector(lexical)
        vectors[row] = 0.72 * semantic + 0.48 * lexical

    return l2_normalize(vectors)


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def normalize_vector(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return vector
    return vector / norm


def hashed_feature(token: str, dim: int) -> tuple[int, float]:
    digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little", signed=False)
    index = value % dim
    sign = 1.0 if (value >> 8) & 1 else -1.0
    return index, sign
