from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import types
from typing import Callable

import numpy as np

from ..contracts import All200Dataset, RepresentationResult


class BATTWISECDDReducer:
    name = "BATTWISE-CDDI"
    requires_targets = False

    def __init__(
        self,
        *,
        source_json: str | Path,
        output_scale: float = 1e5,
        fix_sigma: float = 0.0,
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
    ) -> None:
        self.source_json = Path(source_json)
        self.output_scale = float(output_scale)
        self.fix_sigma = float(fix_sigma)
        self.kernel_mul = float(kernel_mul)
        self.kernel_num = int(kernel_num)

    def _load_generated_function(
        self,
    ) -> tuple[Callable[..., float], dict, str]:
        if not self.source_json.is_file():
            raise FileNotFoundError(
                f"BATTWISE-CDDI source JSON does not exist: {self.source_json}"
            )
        metadata = json.loads(self.source_json.read_text(encoding="utf-8"))
        code = metadata.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"{self.source_json.name} has no generated code")

        module = types.ModuleType("battwise_cddi_generated")
        exec(compile(code, str(self.source_json), "exec"), module.__dict__)
        candidates = sorted(
            name
            for name, value in module.__dict__.items()
            if name.startswith("calculate_mmd") and callable(value)
        )
        if not candidates:
            raise ValueError(
                f"{self.source_json.name} defines no calculate_mmd* function"
            )
        function_name = candidates[0]
        return module.__dict__[function_name], metadata, function_name

    def fit_transform(
        self,
        dataset: All200Dataset,
        seed: int,
    ) -> RepresentationResult:
        scorer, metadata, function_name = self._load_generated_function()
        if dataset.cell_curves_raw.ndim != 3:
            raise ValueError("cell_curves_raw must have shape (cells, points, 2)")
        if dataset.model_curve_raw.shape != dataset.cell_curves_raw.shape[1:]:
            raise ValueError(
                "Model curve shape does not match individual cell curve shape"
            )

        started = time.perf_counter()
        values: list[float] = []
        for cell_id, curve in zip(
            dataset.cell_ids,
            dataset.cell_curves_raw,
            strict=True,
        ):
            score = scorer(
                curve,
                dataset.model_curve_raw,
                fix_sigma=self.fix_sigma,
                kernel_mul=self.kernel_mul,
                kernel_num=self.kernel_num,
            )
            scalar = float(np.asarray(score).squeeze()) * self.output_scale
            if not np.isfinite(scalar):
                raise ValueError(
                    f"BATTWISE-CDDI returned a non-finite score for Cell{cell_id}"
                )
            values.append(scalar)

        code = str(metadata["code"])
        diagnostics = {
            "source_json": str(self.source_json),
            "objective": metadata.get("objective"),
            "operator": metadata.get("operator"),
            "algorithm": metadata.get("algorithm"),
            "function_name": function_name,
            "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "output_scale": self.output_scale,
            "elapsed_s": time.perf_counter() - started,
        }
        return RepresentationResult(
            method=self.name,
            seed=int(seed),
            cell_ids=dataset.cell_ids.copy(),
            z1=np.asarray(values, dtype=float),
            diagnostics=diagnostics,
        )
