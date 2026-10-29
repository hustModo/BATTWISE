from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.fair_1d_benchmark.contracts import All200Dataset
from experiments.fair_1d_benchmark.methods.battwise_cddi import (
    BATTWISECDDReducer,
)


def _dataset() -> All200Dataset:
    ids = np.array([2, 4])
    model = np.zeros((3, 2), dtype=float)
    raw = np.stack(
        (
            np.ones((3, 2), dtype=float),
            np.full((3, 2), 2.0, dtype=float),
        )
    )
    return All200Dataset(
        cell_ids=ids,
        curves=np.zeros((2, 6)),
        cell_curves_raw=raw,
        model_curve_raw=model,
        ocv_ir=np.ones((2, 2)),
        train_targets=np.zeros((2, 2)),
        capacity=np.ones(2),
        electrochemical=pd.DataFrame({"state": [1.0, 2.0]}, index=ids),
        curve_feature_names=("a", "b"),
        electrochemical_feature_names=("state",),
        channel_mean=np.zeros(2),
        channel_std=np.ones(2),
    )


def _write_generated_scorer(path: Path, *, with_function: bool = True) -> None:
    code = "value = 1\n"
    if with_function:
        code += """
import numpy as np

def calculate_mmd_v2(xs, xt, fix_sigma=0, kernel_mul=2.0, kernel_num=5):
    return float(np.mean((xs - xt) ** 2))
"""
    path.write_text(
        json.dumps(
            {
                "algorithm": "synthetic scorer",
                "code": code,
                "objective": -0.9,
                "operator": "test",
            }
        ),
        encoding="utf-8",
    )


def test_battwise_cddi_loads_generated_function_and_scores_every_cell(
    tmp_path: Path,
) -> None:
    source = tmp_path / "best.json"
    _write_generated_scorer(source)
    reducer = BATTWISECDDReducer(source_json=source, output_scale=1e5)

    result = reducer.fit_transform(_dataset(), seed=3)

    assert result.method == "BATTWISE-CDDI"
    assert result.seed == 3
    assert result.cell_ids.tolist() == [2, 4]
    assert np.allclose(result.z1, [1e5, 4e5])
    assert result.diagnostics["objective"] == -0.9
    assert len(result.diagnostics["code_sha256"]) == 64
    assert result.diagnostics["function_name"] == "calculate_mmd_v2"


def test_battwise_cddi_rejects_json_without_calculate_function(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bad.json"
    _write_generated_scorer(source, with_function=False)

    with pytest.raises(ValueError, match="calculate_mmd"):
        BATTWISECDDReducer(source_json=source).fit_transform(_dataset(), seed=0)
