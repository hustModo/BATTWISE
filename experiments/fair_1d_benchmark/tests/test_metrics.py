from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.fair_1d_benchmark.contracts import (
    All200Dataset,
    ClusterResult,
)
from experiments.fair_1d_benchmark.metrics import (
    compute_partition_metrics,
    mean_cluster_variance,
    normalized_pooled_within_variance,
    pooled_within_variance,
)


def test_variance_metrics_match_hand_calculation() -> None:
    values = np.array([1.0, 2.0, 10.0, 12.0])
    labels = np.array([0, 0, 1, 1])

    assert np.isclose(mean_cluster_variance(values, labels), 1.25)
    assert np.isclose(pooled_within_variance(values, labels), 1.25)
    expected = 1.25 / np.var(values, ddof=1)
    assert np.isclose(
        normalized_pooled_within_variance(values, labels),
        expected,
    )


def test_compute_partition_metrics_uses_capacity_and_external_features() -> None:
    ids = np.arange(1, 9)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    matrix = np.column_stack(
        (
            np.r_[np.linspace(0.0, 0.2, 4), np.linspace(0.8, 1.0, 4)],
            np.r_[np.linspace(0.0, 0.1, 4), np.linspace(0.9, 1.0, 4)],
            np.r_[np.zeros(4), np.ones(4)],
        )
    )
    dataset = All200Dataset(
        cell_ids=ids,
        curves=np.zeros((8, 8)),
        cell_curves_raw=np.zeros((8, 4, 2)),
        model_curve_raw=np.zeros((4, 2)),
        ocv_ir=matrix[:, :2],
        train_targets=np.zeros((8, 2)),
        capacity=np.array([1.0, 1.1, 0.9, 1.0, 2.0, 2.1, 1.9, 2.0]),
        electrochemical=pd.DataFrame(
            {
                "state_a": [0.0, 0.1, -0.1, 0.0, 5.0, 5.1, 4.9, 5.0],
                "state_b": [1.0, 1.2, 0.8, 1.0, 3.0, 3.2, 2.8, 3.0],
                "with_nan": [0.0, np.nan, 0.2, 0.1, 4.0, 4.1, 3.9, 4.0],
            },
            index=ids,
        ),
        curve_feature_names=("a", "b"),
        electrochemical_feature_names=("state_a", "state_b", "with_nan"),
        channel_mean=np.zeros(2),
        channel_std=np.ones(2),
    )
    cluster = ClusterResult(
        method="dummy",
        seed=4,
        cell_ids=ids,
        labels=labels,
        cluster_matrix=matrix,
        k=2,
        fitness=0.2,
        dbi=0.1,
        compactness=0.05,
        elapsed_s=0.01,
    )

    summary, per_feature = compute_partition_metrics(dataset, cluster)

    assert summary["method"] == "dummy"
    assert summary["seed"] == 4
    assert summary["k"] == 2
    assert summary["electrochemical_valid_features"] == 3
    assert 0.0 <= summary["silhouette"] <= 1.0
    assert summary["capacity_npwv"] < 0.1
    assert per_feature["feature"].tolist() == [
        "state_a",
        "state_b",
        "with_nan",
    ]
    assert per_feature["n_valid"].tolist() == [8, 8, 7]
