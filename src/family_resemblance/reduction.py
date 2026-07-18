from __future__ import annotations

import numpy as np


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))


def pca_2d(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if len(points) == 1:
        return np.zeros((1, 2), dtype=np.float64)

    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((len(coords), 1), dtype=np.float64)])
    return scale_unit(coords)


def spectral_umap_like(points: np.ndarray, n_neighbors: int = 8) -> np.ndarray:
    n = len(points)
    if n < 4:
        return pca_2d(points)

    distances = pairwise_distances(points)
    k = max(2, min(n_neighbors, n - 1))
    neighbor_order = np.argsort(distances, axis=1)[:, 1:k + 1]
    kth_distances = np.take_along_axis(distances, neighbor_order[:, -1:], axis=1)
    sigma = np.maximum(kth_distances, 1e-9)

    weights = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in neighbor_order[i]:
            weights[i, j] = np.exp(-distances[i, j] / sigma[i, 0])

    weights = np.maximum(weights, weights.T)
    degrees = weights.sum(axis=1)
    if np.count_nonzero(degrees) < 3:
        return pca_2d(points)

    inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    normalized = np.eye(n) - (inv_sqrt[:, None] * weights * inv_sqrt[None, :])
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    order = np.argsort(eigenvalues)
    coords = eigenvectors[:, order[1:3]]
    if coords.shape[1] < 2:
        coords = pca_2d(points)
    return scale_unit(coords)


def scale_unit(coords: np.ndarray) -> np.ndarray:
    if len(coords) == 0:
        return coords
    coords = coords - coords.mean(axis=0, keepdims=True)
    span = np.max(np.abs(coords), axis=0, keepdims=True)
    coords = coords / np.maximum(span, 1e-9)
    return coords
