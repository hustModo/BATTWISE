from __future__ import annotations

import numpy as np

from .contracts import (
    All200Dataset,
    ClusterResult,
    RepresentationResult,
)
from .preprocessing import build_cluster_matrix


def cluster_representation(
    dataset: All200Dataset,
    representation: RepresentationResult,
    *,
    k: int = 6,
    seed: int = 42,
    pop_size: int = 200,
    max_iter: int = 80,
    use_z: bool = True,
) -> ClusterResult:
    if not np.array_equal(dataset.cell_ids, representation.cell_ids):
        raise ValueError("Cell_num order differs between dataset and representation")
    if k < 2 or k > len(dataset.cell_ids):
        raise ValueError(f"k must be between 2 and {len(dataset.cell_ids)}, got {k}")

    matrix = build_cluster_matrix(
        dataset,
        z1=representation.z1 if use_z else None,
    )

    try:
        from app.backend.services.clustering_runner import run_adaptive_kmeans
    except ModuleNotFoundError:
        # The local isolated staging snapshot stores ``app/backend`` as
        # ``app_backend``. Alias it only for local tests; the remote project
        # uses the canonical ``app.backend`` package.
        import importlib
        import sys

        local_backend = importlib.import_module("app_backend")
        sys.modules.setdefault("app", local_backend)
        sys.modules.setdefault("app.backend", local_backend)
        from app_backend.services.clustering_runner import run_adaptive_kmeans

    result = run_adaptive_kmeans(
        matrix,
        np.zeros(len(dataset.cell_ids), dtype=float),
        pop_size=pop_size,
        max_iter=max_iter,
        min_k=k,
        max_k=k,
        use_capacity=False,
        seed=seed,
        feedback_metric="combined",
    )
    labels = np.asarray(result.labels, dtype=int)
    if len(np.unique(labels)) != k:
        raise RuntimeError(
            f"GA-KMeans returned {len(np.unique(labels))} non-empty clusters; expected {k}"
        )
    return ClusterResult(
        method=representation.method,
        seed=seed,
        cell_ids=dataset.cell_ids.copy(),
        labels=labels,
        cluster_matrix=matrix,
        k=int(result.k),
        fitness=float(result.fitness),
        dbi=float(result.dbi),
        compactness=float(result.compactness),
        elapsed_s=float(result.elapsed_s),
    )
