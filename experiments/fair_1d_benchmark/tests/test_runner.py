from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.fair_1d_benchmark.registry import load_reducer
from experiments.fair_1d_benchmark.run import run_benchmark


def _write_dataset(root: Path) -> None:
    index = np.arange(4, dtype=float)
    model = pd.DataFrame(
        {
            "Index": index,
            "Current": np.full(4, 2.0),
            "Voltage": np.linspace(3.0, 3.3, 4),
        }
    )
    model.to_excel(root / "Model_200.xlsx", index=False)
    for cell_id, offset in ((1, 0.1), (2, 0.2), (3, 0.8), (4, 0.9)):
        pd.DataFrame(
            {
                "Index": index,
                "Current": model["Current"] + offset,
                "Voltage": model["Voltage"] + offset,
            }
        ).to_excel(root / f"Cell{cell_id}_200.xlsx", index=False)
    pd.DataFrame(
        {
            "Cell_num": [1, 2, 3, 4],
            "OCV": [3.1, 3.2, 3.8, 3.9],
            "IR": [0.01, 0.02, 0.08, 0.09],
            "Capacity": [1.0, 1.1, 2.0, 2.1],
        }
    ).to_excel(root / "Statis data.xlsx", index=False)
    pd.DataFrame(
        {
            "Cell_num": [1, 2, 3, 4],
            "curve_a": [0.0, 0.1, 0.9, 1.0],
            "curve_b": [0.0, 0.2, 1.8, 2.0],
        }
    ).to_excel(root / "Charging_curve_characteristics.xlsx", index=False)
    pd.DataFrame(
        {
            "Cell_num": [1, 2, 3, 4],
            "state_a": [0.0, 0.1, 1.0, 1.1],
            "state_b": [0.0, 0.2, 2.0, 2.2],
        }
    ).to_excel(root / "Electrochemical_indicators.xlsx", index=False)


def _write_scorer(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "algorithm": "test residual score",
                "objective": -0.5,
                "code": """
import numpy as np
def calculate_mmd_v2(xs, xt, fix_sigma=0, kernel_mul=2.0, kernel_num=5):
    return float(np.mean((xs - xt) ** 2))
""",
            }
        ),
        encoding="utf-8",
    )


def test_registry_loads_reducer_from_dotted_class_path(tmp_path: Path) -> None:
    scorer = tmp_path / "scorer.json"
    _write_scorer(scorer)

    reducer = load_reducer(
        {
            "class_path": (
                "experiments.fair_1d_benchmark.methods.battwise_cddi:"
                "BATTWISECDDReducer"
            ),
            "kwargs": {"source_json": "scorer.json"},
        },
        project_root=tmp_path,
    )

    assert reducer.name == "BATTWISE-CDDI"
    assert reducer.source_json == scorer


def test_runner_writes_common_artifacts_for_registered_methods(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "all200"
    data_dir.mkdir()
    _write_dataset(data_dir)
    scorer = tmp_path / "scorer.json"
    _write_scorer(scorer)
    config = {
        "project_root": str(tmp_path),
        "dataset": {
            "data_dir": "all200",
            "expected_cells": 4,
            "expected_points": 4,
        },
        "clustering": {
            "k": 2,
            "pop_size": 10,
            "max_iter": 3,
            "seeds": [0],
        },
        "output_dir": "results",
        "methods": {
            "battwise_cddi": {
                "class_path": (
                    "experiments.fair_1d_benchmark.methods.battwise_cddi:"
                    "BATTWISECDDReducer"
                ),
                "kwargs": {"source_json": "scorer.json"},
            },
            "ocv_ir": {
                "class_path": (
                    "experiments.fair_1d_benchmark.methods.ocv_ir_control:"
                    "OCVIRControl"
                ),
                "kwargs": {},
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_benchmark(config_path)

    assert result["methods"] == ["battwise_cddi", "ocv_ir"]
    output_dir = tmp_path / "results"
    for filename in (
        "embeddings.csv",
        "assignments.csv",
        "metrics_long.csv",
        "electrochemical_metrics.csv",
        "summary.xlsx",
        "run_manifest.json",
    ):
        assert (output_dir / filename).is_file(), filename

    metrics = pd.read_csv(output_dir / "metrics_long.csv")
    assert set(metrics["method"]) == {"BATTWISE-CDDI", "OCV-IR"}
    assert set(metrics["k"]) == {2}
    assignments = pd.read_csv(output_dir / "assignments.csv")
    assert len(assignments) == 8
    assert set(assignments["Cell_num"]) == {1, 2, 3, 4}
