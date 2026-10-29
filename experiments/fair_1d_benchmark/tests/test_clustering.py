from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.fair_1d_benchmark.clustering import cluster_representation
from experiments.fair_1d_benchmark.contracts import (
    All200Dataset,
    RepresentationResult,
)
from experiments.fair_1d_benchmark.preprocessing import (
    build_cluster_matrix,
    minmax_1d,
)


def _dataset(n: int = 12) -> All200Dataset:
    ids = np.arange(1, n + 1)
    return All200Dataset(
        cell_ids=ids,
        curves=np.zeros((n, 8)),
        cell_curves_raw=np.zeros((n, 4, 2)),
        model_curve_raw=np.zeros((4, 2)),
        ocv_ir=np.column_stack(
            (
                np.linspace(3.0, 4.0, n),
                np.r_[np.linspace(0.01, 0.02, n // 2), np.linspace(0.08, 0.1, n - n // 2)],
            )
        ),
        train_targets=np.zeros((n, 2)),
        capacity=np.linspace(1.0, 2.0, n),
        electrochemical=pd.DataFrame(
            {"state": np.linspace(0.0, 1.0, n)},
            index=ids,
        ),
        curve_feature_names=("a", "b"),
        electrochemical_feature_names=("state",),
        channel_mean=np.zeros(2),
        channel_std=np.ones(2),
    )


def test_minmax_1d_rejects_non_finite_and_handles_constant() -> None:
    assert np.allclose(minmax_1d(np.array([5.0, 5.0])), [0.5, 0.5])
    with pytest.raises(ValueError, match="finite"):
        minmax_1d(np.array([1.0, np.nan]))


def test_build_cluster_matrix_uses_equal_minmax_scaled_coordinates() -> None:
    dataset = _dataset()
    z1 = np.linspace(100.0, 200.0, len(dataset.cell_ids))

    matrix = build_cluster_matrix(dataset, z1=z1)

    assert matrix.shape == (12, 3)
    assert np.allclose(matrix.min(axis=0), 0.0)
    assert np.allclose(matrix.max(axis=0), 1.0)


def test_cluster_representation_uses_fixed_k_and_preserves_cell_order() -> None:
    dataset = _dataset()
    representation = RepresentationResult(
        method="dummy",
        seed=7,
        cell_ids=dataset.cell_ids.copy(),
        z1=np.r_[np.zeros(6), np.ones(6)],
    )

    result = cluster_representation(
        dataset,
        representation,
        k=2,
        seed=7,
        pop_size=12,
        max_iter=4,
    )

    assert result.method == "dummy"
    assert result.cell_ids.tolist() == dataset.cell_ids.tolist()
    assert result.cluster_matrix.shape == (12, 3)
    assert len(np.unique(result.labels)) == 2
    assert result.k == 2


def test_cluster_representation_rejects_reordered_cells() -> None:
    dataset = _dataset()
    representation = RepresentationResult(
        method="dummy",
        seed=1,
        cell_ids=dataset.cell_ids[::-1],
        z1=np.arange(len(dataset.cell_ids), dtype=float),
    )

    with pytest.raises(ValueError, match="Cell_num order"):
        cluster_representation(
            dataset,
            representation,
            k=2,
            seed=1,
            pop_size=8,
            max_iter=2,
        )
