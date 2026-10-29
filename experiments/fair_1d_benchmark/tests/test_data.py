from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.fair_1d_benchmark.data import load_all200_dataset


def _write_synthetic_all200(root: Path, *, bad_index: bool = False) -> None:
    index = np.arange(4, dtype=float)
    model = pd.DataFrame(
        {
            "Index": index,
            "Current": np.full(4, 2.0),
            "Voltage": np.array([3.0, 3.1, 3.2, 3.3]),
        }
    )
    model.to_excel(root / "Model_200.xlsx", index=False)

    for cell_id, offset in ((10, 0.1), (20, 0.2), (30, 0.3)):
        cell_index = index.copy()
        if bad_index and cell_id == 20:
            cell_index[-1] = 99.0
        pd.DataFrame(
            {
                "Index": cell_index,
                "Current": model["Current"] + offset,
                "Voltage": model["Voltage"] + 2.0 * offset,
            }
        ).to_excel(root / f"Cell{cell_id}_200.xlsx", index=False)

    pd.DataFrame(
        {
            "Cell_num": [30, 10, 20],
            "OCV ": [3.3, 3.1, 3.2],
            "IR": [0.03, 0.01, 0.02],
            "Capacity": [2.3, 2.1, 2.2],
        }
    ).to_excel(root / "Statis data.xlsx", index=False)
    pd.DataFrame(
        {
            "Cell_num": [20, 30, 10],
            "curve_a": [2.0, 3.0, 1.0],
            "curve_b": [20.0, 30.0, 10.0],
        }
    ).to_excel(root / "Charging_curve_characteristics.xlsx", index=False)
    pd.DataFrame(
        {
            "Cell_num": [10, 20, 30],
            "electro_a": [0.1, 0.2, 0.3],
            "electro_b": [1.0, 2.0, 3.0],
            "empty": [np.nan, np.nan, np.nan],
            "Unnamed: 99": [1.0, np.nan, 1.0],
        }
    ).to_excel(root / "Electrochemical_indicators.xlsx", index=False)


def test_load_all200_builds_sorted_reference_residual_matrix(tmp_path: Path) -> None:
    _write_synthetic_all200(tmp_path)

    dataset = load_all200_dataset(
        tmp_path,
        expected_cells=3,
        expected_points=4,
    )

    assert dataset.cell_ids.tolist() == [10, 20, 30]
    assert dataset.curves.shape == (3, 8)
    assert dataset.cell_curves_raw.shape == (3, 4, 2)
    assert dataset.model_curve_raw.shape == (4, 2)
    assert dataset.ocv_ir.shape == (3, 2)
    assert dataset.train_targets.shape == (3, 2)
    assert dataset.capacity.tolist() == [2.1, 2.2, 2.3]
    assert dataset.electrochemical.columns.tolist() == [
        "electro_a",
        "electro_b",
    ]
    assert dataset.curve_feature_names == ("curve_a", "curve_b")
    assert dataset.electrochemical_feature_names == (
        "electro_a",
        "electro_b",
    )

    # Each channel is standardised globally across every cell and time point.
    current_block = dataset.curves[:, :4]
    voltage_block = dataset.curves[:, 4:]
    assert np.isclose(current_block.mean(), 0.0)
    assert np.isclose(current_block.std(ddof=0), 1.0)
    assert np.isclose(voltage_block.mean(), 0.0)
    assert np.isclose(voltage_block.std(ddof=0), 1.0)

    # A per-cell normaliser would make every row identical/zero. The public
    # preprocessing must preserve the between-cell ageing offsets.
    assert not np.allclose(dataset.curves[0], dataset.curves[1])


def test_load_all200_rejects_misaligned_curve_index(tmp_path: Path) -> None:
    _write_synthetic_all200(tmp_path, bad_index=True)

    with pytest.raises(ValueError, match="Index mismatch.*Cell20"):
        load_all200_dataset(
            tmp_path,
            expected_cells=3,
            expected_points=4,
        )


def test_load_all200_rejects_cell_id_mismatch_across_tables(
    tmp_path: Path,
) -> None:
    _write_synthetic_all200(tmp_path)
    statis = pd.read_excel(tmp_path / "Statis data.xlsx")
    statis.loc[0, "Cell_num"] = 999
    statis.to_excel(tmp_path / "Statis data.xlsx", index=False)

    with pytest.raises(ValueError, match="Cell_num mismatch.*Statis data"):
        load_all200_dataset(
            tmp_path,
            expected_cells=3,
            expected_points=4,
        )
