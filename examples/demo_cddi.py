"""Deterministic synthetic demo; no measured battery data or LLM calls."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import time

import numpy as np


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    program = root / "programs/cddi/calculate_mmd_v2.py"
    expected = "f18815885200043f9f14371d169b6bc600757d8b19e70f262fd175d2d6bbfda7"
    code_hash = hashlib.sha256(program.read_text(encoding="utf-8").strip().encode("utf-8")).hexdigest()
    if code_hash != expected:
        raise RuntimeError("Frozen program hash mismatch")
    spec = importlib.util.spec_from_file_location("demo_cddi_program", program)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    np.random.seed(20261029)
    t = np.linspace(0.0, 1.0, 200)
    reference = np.column_stack((2.0 - 0.6 * t + 0.04 * np.sin(8 * t),
                                 3.1 + 0.8 * t + 0.02 * np.cos(9 * t)))
    started = time.perf_counter()
    for cell, offset in enumerate((0.02, 0.05, 0.10), 1):
        curve = reference + np.column_stack((offset * (1 + t), -offset * (0.3 + t ** 2)))
        score = float(module.calculate_mmd_v2(curve, reference))
        if not np.isfinite(score):
            raise RuntimeError("Synthetic demo produced a non-finite value")
        print(f"Synthetic cell {cell}: CDDI={score:.8f}")
    print(f"PASS: 3 synthetic curves, 200 points, 2 channels; elapsed={time.perf_counter()-started:.3f}s")
    print("This interface demonstration is not a manuscript experimental result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
