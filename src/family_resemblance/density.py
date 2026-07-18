from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reduction import pairwise_distances


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    threshold: float
    core_distances: np.ndarray
    mst_edges: list[tuple[int, int, float]]


def hdbscan_like(
    points: np.ndarray,
    min_cluster_size: int = 4,
    min_samples: int = 3,
) -> ClusterResult:
    n = len(points)
    if n == 0:
        return ClusterResult(np.array([], dtype=int), 0.0, np.array([]), [])
    if n < min_cluster_size:
        return ClusterResult(np.full(n, -1, dtype=int), 0.0, np.zeros(n), [])

    distances = pairwise_distances(points)
    order = np.sort(distances, axis=1)
    sample_index = min(max(1, min_samples), n - 1)
    core = order[:, sample_index]
    mutual = np.maximum(np.maximum(distances, core[:, None]), core[None, :])
    mst = prim_mst(mutual)
    edge_weights = [edge[2] for edge in mst]
    threshold = choose_density_threshold(edge_weights, min_cluster_size)
    labels = choose_cluster_cut(n, mst, edge_weights, threshold, min_cluster_size)
    if len(set(labels.tolist()) - {-1}) < 2:
        labels = dbscan(points, eps=float(np.quantile(core, 0.72)), min_samples=min_samples)
        labels = prune_small_clusters(labels, min_cluster_size)

    return ClusterResult(labels=labels, threshold=float(threshold), core_distances=core, mst_edges=mst)


def prim_mst(weights: np.ndarray) -> list[tuple[int, int, float]]:
    n = weights.shape[0]
    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    edges: list[tuple[int, int, float]] = []

    for _ in range(n - 1):
        best_i = -1
        best_j = -1
        best_weight = float("inf")
        for i in np.where(selected)[0]:
            candidates = np.where(~selected)[0]
            if len(candidates) == 0:
                break
            j = candidates[np.argmin(weights[i, candidates])]
            weight = float(weights[i, j])
            if weight < best_weight:
                best_i, best_j, best_weight = int(i), int(j), weight

        if best_j == -1:
            break
        selected[best_j] = True
        edges.append((best_i, best_j, best_weight))

    return edges


def choose_density_threshold(weights: list[float], min_cluster_size: int) -> float:
    if not weights:
        return 0.0
    ordered = np.sort(np.array(weights, dtype=np.float64))
    if len(ordered) < 4:
        return float(np.quantile(ordered, 0.70))

    gaps = np.diff(ordered)
    start = max(1, min_cluster_size // 2)
    if start >= len(gaps):
        return float(np.quantile(ordered, 0.72))

    gap_index = int(np.argmax(gaps[start:]) + start)
    if gaps[gap_index] <= 1e-12:
        return float(np.quantile(ordered, 0.55))
    return float(min(ordered[gap_index], np.quantile(ordered, 0.58)))


def choose_cluster_cut(
    n: int,
    mst: list[tuple[int, int, float]],
    weights: list[float],
    preferred_threshold: float,
    min_cluster_size: int,
) -> np.ndarray:
    if not weights:
        return np.full(n, -1, dtype=int)

    ordered = np.array(weights, dtype=np.float64)
    candidates = [preferred_threshold]
    candidates.extend(float(np.quantile(ordered, q)) for q in (0.58, 0.50, 0.42, 0.34, 0.26))

    best_labels = components_from_edges(n, mst, preferred_threshold, min_cluster_size)
    best_score = score_cut(best_labels)
    for threshold in candidates:
        labels = components_from_edges(n, mst, threshold, min_cluster_size)
        score = score_cut(labels)
        if score > best_score:
            best_labels = labels
            best_score = score
    return best_labels


def score_cut(labels: np.ndarray) -> float:
    n = len(labels)
    cluster_labels = set(labels.tolist()) - {-1}
    cluster_count = len(cluster_labels)
    coverage = float(np.count_nonzero(labels >= 0) / max(n, 1))
    if cluster_count == 0:
        return -1.0
    count_score = min(cluster_count, 5) / 5.0
    single_penalty = 0.42 if cluster_count == 1 else 0.0
    noise_penalty = abs((1.0 - coverage) - 0.22) * 0.35
    return count_score + coverage * 0.65 - single_penalty - noise_penalty


def components_from_edges(
    n: int,
    edges: list[tuple[int, int, float]],
    threshold: float,
    min_cluster_size: int,
) -> np.ndarray:
    parent = list(range(n))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i, j, weight in edges:
        if weight <= threshold:
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    labels = np.full(n, -1, dtype=int)
    next_label = 0
    for members in sorted(groups.values(), key=lambda group: (-len(group), group[0])):
        if len(members) < min_cluster_size:
            continue
        for index in members:
            labels[index] = next_label
        next_label += 1
    return labels


def dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    n = len(points)
    distances = pairwise_distances(points)
    labels = np.full(n, -99, dtype=int)
    cluster_id = 0

    for point in range(n):
        if labels[point] != -99:
            continue
        neighbors = np.where(distances[point] <= eps)[0].tolist()
        if len(neighbors) < min_samples:
            labels[point] = -1
            continue

        labels[point] = cluster_id
        seeds = [neighbor for neighbor in neighbors if neighbor != point]
        while seeds:
            current = seeds.pop()
            if labels[current] == -1:
                labels[current] = cluster_id
            if labels[current] != -99:
                continue
            labels[current] = cluster_id
            current_neighbors = np.where(distances[current] <= eps)[0].tolist()
            if len(current_neighbors) >= min_samples:
                for neighbor in current_neighbors:
                    if labels[neighbor] in (-99, -1):
                        seeds.append(neighbor)
        cluster_id += 1

    labels[labels == -99] = -1
    return labels


def prune_small_clusters(labels: np.ndarray, min_cluster_size: int) -> np.ndarray:
    pruned = labels.copy()
    for label in set(labels.tolist()) - {-1}:
        if np.count_nonzero(labels == label) < min_cluster_size:
            pruned[labels == label] = -1

    remap = {}
    next_label = 0
    for label in sorted(set(pruned.tolist()) - {-1}):
        remap[label] = next_label
        next_label += 1
    for old, new in remap.items():
        pruned[pruned == old] = new
    return pruned
