from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from experiments.fair_1d_benchmark.contracts import All200Dataset
from experiments.fair_1d_benchmark.methods.kpca_1d import KPCA1DReducer


def _dataset(curves: np.ndarray | None = None) -> All200Dataset:
    cell_ids = np.array([41, 7, 99, 12], dtype=int)
    if curves is None:
        curves = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 1.0],
            ],
            dtype=float,
        )
    curves = np.asarray(curves, dtype=float)
    n_cells = curves.shape[0]
    if n_cells != len(cell_ids):
        cell_ids = np.arange(100, 100 + n_cells, dtype=int)

    return All200Dataset(
        cell_ids=cell_ids,
        curves=curves,
        cell_curves_raw=np.full((n_cells, 2, 2), 11.0),
        model_curve_raw=np.full((2, 2), 13.0),
        ocv_ir=np.full((n_cells, 2), 17.0),
        train_targets=np.full((n_cells, 3), 19.0),
        capacity=np.full(n_cells, 23.0),
        electrochemical=pd.DataFrame(
            {"state_a": np.full(n_cells, 29.0)},
            index=cell_ids,
        ),
        curve_feature_names=("curve_a", "curve_b", "curve_c"),
        electrochemical_feature_names=("state_a",),
        channel_mean=np.array([31.0, 37.0]),
        channel_std=np.array([41.0, 43.0]),
    )


def test_kpca_1d_returns_finite_ordered_one_dimensional_representation() -> None:
    dataset = _dataset()
    reducer = KPCA1DReducer()

    result = reducer.fit_transform(dataset, seed=5)

    assert reducer.name == "KPCA-1D"
    assert reducer.requires_targets is False
    assert result.method == "KPCA-1D"
    assert result.seed == 5
    assert result.cell_ids.tolist() == [41, 7, 99, 12]
    assert result.z1.shape == (4,)
    assert np.isfinite(result.z1).all()
    assert result.diagnostics["gamma"] == pytest.approx(0.1)
    assert result.diagnostics["n_components"] == 1
    assert result.diagnostics["kernel"] == "rbf"
    assert result.diagnostics["elapsed_s"] >= 0.0
    assert np.isfinite(result.diagnostics["first_eigenvalue"])
    assert result.diagnostics["first_eigenvalue"] > 0.0


def test_kpca_1d_is_deterministic_for_the_same_curves() -> None:
    dataset = _dataset()
    reducer = KPCA1DReducer()

    first = reducer.fit_transform(dataset, seed=1)
    second = reducer.fit_transform(dataset, seed=999)

    np.testing.assert_allclose(first.z1, second.z1, rtol=0.0, atol=1e-12)


def test_kpca_1d_ignores_targets_capacity_and_electrochemical_fields() -> None:
    dataset = _dataset()
    changed = replace(
        dataset,
        cell_curves_raw=np.full_like(dataset.cell_curves_raw, -101.0),
        model_curve_raw=np.full_like(dataset.model_curve_raw, -103.0),
        ocv_ir=np.full_like(dataset.ocv_ir, -107.0),
        train_targets=np.full_like(dataset.train_targets, -109.0),
        capacity=np.full_like(dataset.capacity, -113.0),
        electrochemical=pd.DataFrame(
            {"state_a": np.full(len(dataset.cell_ids), -127.0)},
            index=dataset.cell_ids,
        ),
        channel_mean=np.array([-131.0, -137.0]),
        channel_std=np.array([-139.0, -149.0]),
    )

    baseline = KPCA1DReducer().fit_transform(dataset, seed=3)
    altered = KPCA1DReducer().fit_transform(changed, seed=3)

    np.testing.assert_allclose(baseline.z1, altered.z1, rtol=0.0, atol=1e-12)


def test_kpca_1d_excludes_zero_distances_from_partial_duplicates() -> None:
    dataset = _dataset(
        np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [2.0, 0.0],
            ]
        )
    )

    result = KPCA1DReducer().fit_transform(dataset, seed=0)

    assert result.z1.shape == (3,)
    assert np.isfinite(result.z1).all()
    assert result.diagnostics["gamma"] == pytest.approx(0.125)


@pytest.mark.parametrize(
    "curves",
    [
        np.zeros((4, 3), dtype=float),
        np.tile(np.array([[1.0, 2.0, 3.0]]), (4, 1)),
    ],
    ids=["constant", "all-repeated"],
)
def test_kpca_1d_rejects_curves_without_positive_pairwise_distance(
    curves: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="positive pairwise squared distance"):
        KPCA1DReducer().fit_transform(_dataset(curves), seed=0)
