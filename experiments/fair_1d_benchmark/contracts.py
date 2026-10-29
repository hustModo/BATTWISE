from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class All200Dataset:
    """Aligned inputs, training targets, and held-out evaluation variables."""

    cell_ids: np.ndarray
    curves: np.ndarray
    cell_curves_raw: np.ndarray
    model_curve_raw: np.ndarray
    ocv_ir: np.ndarray
    train_targets: np.ndarray
    capacity: np.ndarray
    electrochemical: pd.DataFrame
    curve_feature_names: tuple[str, ...]
    electrochemical_feature_names: tuple[str, ...]
    channel_mean: np.ndarray
    channel_std: np.ndarray


@dataclass
class RepresentationResult:
    method: str
    seed: int
    cell_ids: np.ndarray
    z1: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterResult:
    method: str
    seed: int
    cell_ids: np.ndarray
    labels: np.ndarray
    cluster_matrix: np.ndarray
    k: int
    fitness: float
    dbi: float
    compactness: float
    elapsed_s: float


@runtime_checkable
class Reducer1D(Protocol):
    name: str
    requires_targets: bool

    def fit_transform(
        self,
        dataset: All200Dataset,
        seed: int,
    ) -> RepresentationResult:
        ...
