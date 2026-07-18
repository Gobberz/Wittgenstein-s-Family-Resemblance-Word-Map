from __future__ import annotations

import numpy as np

from .reduction import pairwise_distances


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def target_metrics(vectors: np.ndarray, coords: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    n = len(vectors)
    if n == 0:
        return empty_metrics()
    if n == 1:
        result = empty_metrics()
        result.update({"occurrences": 1, "medoid_dominance": 1.0})
        return result

    cosine = cosine_matrix(vectors)
    mean_similarity = (cosine.sum(axis=1) - 1.0) / max(n - 1, 1)
    medoid_index = int(np.argmax(mean_similarity))
    medoid_dominance = float(mean_similarity[medoid_index])

    distances = pairwise_distances(coords)
    upper = distances[np.triu_indices(n, k=1)]
    semantic_diameter = float(np.max(upper)) if len(upper) else 0.0
    median_pairwise = float(np.median(upper)) if len(upper) else 0.0
    nearest = np.partition(distances + np.eye(n) * 1e9, 1, axis=1)[:, 0]
    median_nearest = float(np.median(nearest))
    dispersion_ratio = median_pairwise / max(median_nearest, 1e-9)

    cluster_labels = set(labels.tolist()) - {-1}
    cluster_count = len(cluster_labels)
    noise_ratio = float(np.count_nonzero(labels == -1) / n)
    entropy = cluster_entropy(labels)

    anti_essence = (
        0.38 * clamp01((1.0 - medoid_dominance) / 1.2)
        + 0.24 * clamp01(cluster_count / 4.0)
        + 0.24 * clamp01(dispersion_ratio / 5.0)
        + 0.14 * clamp01(noise_ratio)
    )

    return {
        "occurrences": int(n),
        "cluster_count": int(cluster_count),
        "noise_ratio": round(noise_ratio, 4),
        "cluster_entropy": round(float(entropy), 4),
        "medoid_index": medoid_index,
        "medoid_dominance": round(medoid_dominance, 4),
        "semantic_diameter": round(semantic_diameter, 4),
        "dispersion_ratio": round(float(dispersion_ratio), 4),
        "anti_essence_score": round(float(clamp01(anti_essence)), 4),
    }


def cluster_entropy(labels: np.ndarray) -> float:
    usable = labels[labels >= 0]
    if len(usable) == 0:
        return 0.0
    counts = np.array([np.count_nonzero(usable == label) for label in sorted(set(usable.tolist()))])
    probs = counts / counts.sum()
    return float(-(probs * np.log2(np.maximum(probs, 1e-12))).sum())


def empty_metrics() -> dict[str, float | int]:
    return {
        "occurrences": 0,
        "cluster_count": 0,
        "noise_ratio": 0.0,
        "cluster_entropy": 0.0,
        "medoid_index": -1,
        "medoid_dominance": 0.0,
        "semantic_diameter": 0.0,
        "dispersion_ratio": 0.0,
        "anti_essence_score": 0.0,
    }


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
