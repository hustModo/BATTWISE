from __future__ import annotations

import time

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import KernelPCA

from ..contracts import All200Dataset, RepresentationResult


class KPCA1DReducer:
    name = "KPCA-1D"
    requires_targets = False

    def fit_transform(
        self,
        dataset: All200Dataset,
        seed: int,
    ) -> RepresentationResult:
        started = time.perf_counter()
        curves = np.asarray(dataset.curves, dtype=float)
        cell_ids = np.asarray(dataset.cell_ids)

        if curves.ndim != 2:
            raise ValueError("dataset.curves must have shape (cells, features)")
        if curves.shape[0] != len(cell_ids):
            raise ValueError(
                "dataset.curves row count must match the number of Cell_num values"
            )
        if curves.shape[1] == 0:
            raise ValueError("dataset.curves must contain at least one feature")
        if not np.isfinite(curves).all():
            raise ValueError("dataset.curves must contain only finite values")

        squared_distances = pdist(curves, metric="sqeuclidean")
        if not np.isfinite(squared_distances).all():
            raise ValueError("Pairwise squared distances must be finite")
        positive_distances = squared_distances[squared_distances > 0.0]
        if positive_distances.size == 0:
            raise ValueError(
                "KPCA-1D requires a positive pairwise squared distance "
                "to compute the median-distance gamma"
            )

        median_d2 = float(np.median(positive_distances))
        if not np.isfinite(median_d2) or median_d2 <= 0.0:
            raise ValueError(
                "Median positive pairwise squared distance is not usable "
                "for KPCA-1D gamma"
            )
        gamma = 1.0 / (2.0 * median_d2)

        kpca = KernelPCA(
            n_components=1,
            kernel="rbf",
            gamma=gamma,
            eigen_solver="dense",
        )
        z1 = np.asarray(kpca.fit_transform(curves), dtype=float).reshape(-1)
        if z1.shape != (curves.shape[0],) or not np.isfinite(z1).all():
            raise ValueError("KPCA-1D returned an invalid one-dimensional output")

        first_eigenvalue = float(kpca.eigenvalues_[0])
        diagnostics = {
            "gamma": float(gamma),
            "median_d2": median_d2,
            "elapsed_s": time.perf_counter() - started,
            "n_components": 1,
            "kernel": "rbf",
            "first_eigenvalue": first_eigenvalue,
        }
        return RepresentationResult(
            method=self.name,
            seed=int(seed),
            cell_ids=cell_ids.copy(),
            z1=z1,
            diagnostics=diagnostics,
        )
