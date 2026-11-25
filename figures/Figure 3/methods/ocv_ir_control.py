from __future__ import annotations

import numpy as np

from ..contracts import All200Dataset, RepresentationResult


class OCVIRControl:
    name = "OCV-IR"
    requires_targets = False

    def fit_transform(
        self,
        dataset: All200Dataset,
        seed: int,
    ) -> RepresentationResult:
        return RepresentationResult(
            method=self.name,
            seed=int(seed),
            cell_ids=dataset.cell_ids.copy(),
            z1=np.zeros(len(dataset.cell_ids), dtype=float),
            diagnostics={"use_z": False, "elapsed_s": 0.0},
        )
