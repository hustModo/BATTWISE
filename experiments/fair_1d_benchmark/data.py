from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .contracts import All200Dataset


_CELL_RE = re.compile(r"Cell(\d+)", re.IGNORECASE)


def _read_excel(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=0)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _find_data_file(data_dir: Path, stem: str) -> Path:
    matches = sorted(
        path
        for path in data_dir.glob(f"{stem}.*")
        if path.suffix.lower() in {".xls", ".xlsx"}
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {stem}.xls/.xlsx in {data_dir}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _extract_cell_id(path: Path) -> int:
    match = _CELL_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot extract Cell_num from {path.name}")
    return int(match.group(1))


def _clean_cell_table(path: Path) -> pd.DataFrame:
    frame = _read_excel(path)
    if "Cell_num" not in frame.columns:
        raise ValueError(f"{path.name} is missing Cell_num")
    frame["Cell_num"] = pd.to_numeric(frame["Cell_num"], errors="coerce")
    if frame["Cell_num"].isna().any():
        raise ValueError(f"{path.name} contains invalid Cell_num values")
    frame["Cell_num"] = frame["Cell_num"].astype(int)
    if frame["Cell_num"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["Cell_num"].duplicated(), "Cell_num"].unique()
        )
        raise ValueError(f"{path.name} contains duplicate Cell_num: {duplicates}")
    frame = frame.dropna(axis=1, how="all")
    frame = frame.loc[
        :,
        [
            column
            for column in frame.columns
            if not str(column).strip().lower().startswith("unnamed:")
        ],
    ]
    return frame


def _align_table(
    frame: pd.DataFrame,
    expected_ids: np.ndarray,
    *,
    source_name: str,
) -> pd.DataFrame:
    actual = set(frame["Cell_num"].tolist())
    expected = set(int(value) for value in expected_ids)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Cell_num mismatch in {source_name}: missing={missing}, extra={extra}"
        )
    return frame.set_index("Cell_num").loc[expected_ids].reset_index()


def _numeric_features(
    frame: pd.DataFrame,
    *,
    source_name: str,
    allow_nan: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    columns = tuple(column for column in frame.columns if column != "Cell_num")
    if not columns:
        raise ValueError(f"{source_name} contains no feature columns")
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    columns = tuple(str(column) for column in numeric.columns)
    if not columns:
        raise ValueError(f"{source_name} contains no numeric feature columns")
    if not allow_nan and numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"{source_name} contains missing values in {bad}")
    return numeric.to_numpy(dtype=float), columns


def _standardize_columns(values: np.ndarray, source_name: str) -> np.ndarray:
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=0)
    if np.any(~np.isfinite(std)) or np.any(std <= 0):
        bad = np.where((~np.isfinite(std)) | (std <= 0))[0].tolist()
        raise ValueError(f"{source_name} has constant/invalid columns: {bad}")
    return (values - mean) / std


def load_all200_dataset(
    data_dir: Path,
    *,
    expected_cells: int = 71,
    expected_points: int = 200,
) -> All200Dataset:
    """Load the common All_200 benchmark without leaking evaluation variables."""

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"All_200 data directory does not exist: {data_dir}")

    model_path = _find_data_file(data_dir, "Model_200")
    model = _read_excel(model_path)
    required_curve_columns = {"Index", "Current", "Voltage"}
    if not required_curve_columns.issubset(model.columns):
        raise ValueError(f"{model_path.name} must contain Index, Current, Voltage")
    if len(model) != expected_points:
        raise ValueError(
            f"{model_path.name} has {len(model)} points; expected {expected_points}"
        )
    model_index = pd.to_numeric(model["Index"], errors="coerce").to_numpy(dtype=float)
    model_values = (
        model.loc[:, ["Current", "Voltage"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    if not np.isfinite(model_index).all() or not np.isfinite(model_values).all():
        raise ValueError(f"{model_path.name} contains non-numeric curve values")

    cell_paths = sorted(
        (
            path
            for path in data_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".xls", ".xlsx"}
            and _CELL_RE.search(path.name)
        ),
        key=_extract_cell_id,
    )
    cell_ids = np.asarray([_extract_cell_id(path) for path in cell_paths], dtype=int)
    if len(cell_ids) != expected_cells:
        raise ValueError(
            f"All_200 has {len(cell_ids)} cell curves; expected {expected_cells}"
        )
    if len(set(cell_ids.tolist())) != len(cell_ids):
        raise ValueError("All_200 contains duplicate Cell_num curve files")

    residuals: list[np.ndarray] = []
    raw_cell_curves: list[np.ndarray] = []
    for cell_id, path in zip(cell_ids, cell_paths, strict=True):
        frame = _read_excel(path)
        if not required_curve_columns.issubset(frame.columns):
            raise ValueError(f"{path.name} must contain Index, Current, Voltage")
        if len(frame) != expected_points:
            raise ValueError(
                f"{path.name} has {len(frame)} points; expected {expected_points}"
            )
        index = pd.to_numeric(frame["Index"], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(index, model_index, rtol=0.0, atol=1e-12):
            raise ValueError(f"Index mismatch between Cell{cell_id} and Model_200")
        values = (
            frame.loc[:, ["Current", "Voltage"]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name} contains non-numeric curve values")
        raw_cell_curves.append(values)
        residuals.append(values - model_values)

    residual_array = np.stack(residuals, axis=0)
    raw_cell_array = np.stack(raw_cell_curves, axis=0)
    channel_mean = residual_array.mean(axis=(0, 1))
    channel_std = residual_array.std(axis=(0, 1), ddof=0)
    if np.any(~np.isfinite(channel_std)) or np.any(channel_std <= 0):
        raise ValueError(
            "Current/Voltage residual channels must have finite non-zero variance"
        )
    standard = (residual_array - channel_mean) / channel_std
    curves = np.concatenate((standard[:, :, 0], standard[:, :, 1]), axis=1)

    statis_path = _find_data_file(data_dir, "Statis data")
    statis = _align_table(
        _clean_cell_table(statis_path),
        cell_ids,
        source_name="Statis data",
    )
    required_statis = {"OCV", "IR", "Capacity"}
    missing_statis = required_statis - set(statis.columns)
    if missing_statis:
        raise ValueError(f"Statis data is missing columns: {sorted(missing_statis)}")
    statis_values = (
        statis.loc[:, ["OCV", "IR", "Capacity"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    if not np.isfinite(statis_values).all():
        raise ValueError("Statis data contains missing/non-numeric values")

    curve_features_path = _find_data_file(
        data_dir,
        "Charging_curve_characteristics",
    )
    curve_features = _align_table(
        _clean_cell_table(curve_features_path),
        cell_ids,
        source_name="Charging_curve_characteristics",
    )
    raw_targets, curve_feature_names = _numeric_features(
        curve_features,
        source_name="Charging_curve_characteristics",
        allow_nan=False,
    )
    train_targets = _standardize_columns(
        raw_targets,
        "Charging_curve_characteristics",
    )

    electro_path = _find_data_file(data_dir, "Electrochemical_indicators")
    electro = _align_table(
        _clean_cell_table(electro_path),
        cell_ids,
        source_name="Electrochemical_indicators",
    )
    electro_values, electro_names = _numeric_features(
        electro,
        source_name="Electrochemical_indicators",
        allow_nan=True,
    )
    electro_frame = pd.DataFrame(
        electro_values,
        index=cell_ids,
        columns=electro_names,
    )
    electro_frame.index.name = "Cell_num"

    return All200Dataset(
        cell_ids=cell_ids,
        curves=curves,
        cell_curves_raw=raw_cell_array,
        model_curve_raw=model_values,
        ocv_ir=statis_values[:, :2],
        train_targets=train_targets,
        capacity=statis_values[:, 2],
        electrochemical=electro_frame,
        curve_feature_names=curve_feature_names,
        electrochemical_feature_names=electro_names,
        channel_mean=channel_mean,
        channel_std=channel_std,
    )
