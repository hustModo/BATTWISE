from __future__ import annotations

import numpy as np

from .contracts import All200Dataset


def minmax_1d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Values must be a non-empty finite one-dimensional array")
    low = float(array.min())
    high = float(array.max())
    if np.isclose(high, low):
        return np.full(array.shape, 0.5, dtype=float)
    return (array - low) / (high - low)


def build_cluster_matrix(
    dataset: All200Dataset,
    *,
    z1: np.ndarray | None,
) -> np.ndarray:
    ocv_ir = np.asarray(dataset.ocv_ir, dtype=float)
    if ocv_ir.shape != (len(dataset.cell_ids), 2):
        raise ValueError(
            f"OCV/IR must have shape ({len(dataset.cell_ids)}, 2), got {ocv_ir.shape}"
        )
    columns = [minmax_1d(ocv_ir[:, 0]), minmax_1d(ocv_ir[:, 1])]
    if z1 is not None:
        z_array = np.asarray(z1, dtype=float).reshape(-1)
        if len(z_array) != len(dataset.cell_ids):
            raise ValueError(
                f"z1 has {len(z_array)} rows; expected {len(dataset.cell_ids)}"
            )
        columns.append(minmax_1d(z_array))
    return np.column_stack(columns)
