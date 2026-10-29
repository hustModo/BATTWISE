from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .contracts import All200Dataset, ClusterResult


def _validated_finite(
    values: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    value_array = np.asarray(values, dtype=float).reshape(-1)
    label_array = np.asarray(labels).reshape(-1)
    if len(value_array) != len(label_array):
        raise ValueError("values and labels must have the same length")
    valid = np.isfinite(value_array)
    return value_array[valid], label_array[valid]


def mean_cluster_variance(values: np.ndarray, labels: np.ndarray) -> float:
    clean_values, clean_labels = _validated_finite(values, labels)
    variances = [
        float(np.var(clean_values[clean_labels == label], ddof=1))
        for label in np.unique(clean_labels)
        if np.sum(clean_labels == label) >= 2
    ]
    return float(np.mean(variances)) if variances else float("nan")


def pooled_within_variance(values: np.ndarray, labels: np.ndarray) -> float:
    clean_values, clean_labels = _validated_finite(values, labels)
    numerator = 0.0
    denominator = 0
    for label in np.unique(clean_labels):
        group = clean_values[clean_labels == label]
        if len(group) >= 2:
            numerator += float((len(group) - 1) * np.var(group, ddof=1))
            denominator += len(group) - 1
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def normalized_pooled_within_variance(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    clean_values, clean_labels = _validated_finite(values, labels)
    if len(clean_values) < 2:
        return float("nan")
    global_variance = float(np.var(clean_values, ddof=1))
    if not np.isfinite(global_variance) or global_variance <= 0:
        return float("nan")
    return pooled_within_variance(clean_values, clean_labels) / global_variance


def _sse(matrix: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for label in np.unique(labels):
        group = matrix[labels == label]
        centroid = group.mean(axis=0)
        total += float(np.sum((group - centroid) ** 2))
    return total


def compute_partition_metrics(
    dataset: All200Dataset,
    cluster: ClusterResult,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    if not np.array_equal(dataset.cell_ids, cluster.cell_ids):
        raise ValueError("Cell_num order differs between dataset and cluster result")
    labels = np.asarray(cluster.labels, dtype=int)
    matrix = np.asarray(cluster.cluster_matrix, dtype=float)
    unique = np.unique(labels)
    if len(unique) < 2:
        raise ValueError("Partition metrics require at least two clusters")

    electro_rows: list[dict[str, float | int | str]] = []
    for feature in dataset.electrochemical_feature_names:
        values = pd.to_numeric(
            dataset.electrochemical[feature],
            errors="coerce",
        ).to_numpy(dtype=float)
        electro_rows.append(
            {
                "method": cluster.method,
                "seed": cluster.seed,
                "feature": feature,
                "n_valid": int(np.isfinite(values).sum()),
                "mean_cluster_variance": mean_cluster_variance(values, labels),
                "pooled_within_variance": pooled_within_variance(values, labels),
                "npwv": normalized_pooled_within_variance(values, labels),
            }
        )
    per_feature = pd.DataFrame(electro_rows)
    valid_npwv = pd.to_numeric(per_feature["npwv"], errors="coerce").dropna()

    summary: dict[str, float | int | str] = {
        "method": cluster.method,
        "seed": int(cluster.seed),
        "n_cells": int(len(dataset.cell_ids)),
        "k": int(cluster.k),
        "capacity_mean_cluster_variance": mean_cluster_variance(
            dataset.capacity,
            labels,
        ),
        "capacity_pooled_within_variance": pooled_within_variance(
            dataset.capacity,
            labels,
        ),
        "capacity_npwv": normalized_pooled_within_variance(
            dataset.capacity,
            labels,
        ),
        "electrochemical_valid_features": int(len(valid_npwv)),
        "electrochemical_mean_npwv": float(valid_npwv.mean()),
        "electrochemical_median_npwv": float(valid_npwv.median()),
        "dbi": float(davies_bouldin_score(matrix, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
        "silhouette": float(silhouette_score(matrix, labels)),
        "sse": _sse(matrix, labels),
        "ga_fitness": float(cluster.fitness),
        "ga_compactness": float(cluster.compactness),
        "ga_elapsed_s": float(cluster.elapsed_s),
    }
    return summary, per_feature
