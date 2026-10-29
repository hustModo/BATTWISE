"""Offline test for mmd_runner: inject good MMD_cal_new + run on builtin-first-200s.

Uses the GOOD_CODE constant from test_mmd_designer so no LLM call is needed.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd

from app.backend.services import mmd_runner
from app.backend.services.mmd_runner import MMDProgress
from app.backend.settings import DIANCHI_MMD_TOOLS_NEW
from app.backend.services import data_resolver
from app.backend.tests.test_mmd_designer import GOOD_CODE


def t_inject_and_reload() -> None:
    original_text = (
        DIANCHI_MMD_TOOLS_NEW.read_text(encoding="utf-8") if DIANCHI_MMD_TOOLS_NEW.exists() else None
    )
    try:
        code_only = GOOD_CODE.strip().strip("`")
        code_only = code_only.replace("python\n", "", 1)
        path = mmd_runner.inject_mmd_code(code_only)
        assert path == DIANCHI_MMD_TOOLS_NEW, "wrong target path"
        text = path.read_text(encoding="utf-8")
        assert "MMD_cal_new" in text, "injection did not rename function"
        bak = path.with_suffix(path.suffix + ".bak")
        if original_text is not None:
            assert bak.exists(), "expected .bak backup"

        fn = mmd_runner.load_injected_function()
        assert callable(fn), "load_injected_function did not return callable"
        print("  [OK] inject + .bak backup + reload")

        ds = data_resolver.resolve("builtin-first-200s")
        with tempfile.TemporaryDirectory() as td:
            out_csv = Path(td) / "mmd_results.csv"
            progresses = []

            def cb(p: MMDProgress) -> None:
                progresses.append(p)

            summary = mmd_runner.run_full_pipeline(
                data_folder=ds.folder,
                model_filename=ds.model_file,
                char_filename=ds.char_file,
                out_csv=out_csv,
                mmd_fn=fn,
                on_progress=cb,
            )

            assert out_csv.exists(), "CSV not written"
            df = pd.read_csv(out_csv)
            expected_cols = {"Filename", "Cell_num", "MMD_Loss", "Sample_count"}
            assert expected_cols.issubset(set(df.columns)), f"missing cols, got {df.columns}"
            assert len(df) >= 60, f"expected ≥60 cells, got {len(df)}"
            losses = df["MMD_Loss"].values
            assert all(map(lambda v: v == v and v != float("inf"), losses)), "non-finite losses"
            assert len(set(round(float(v), 8) for v in losses)) > 5, "losses too similar"
            assert summary["total_cells"] == ds.cell_count
            assert summary["correlation"] is not None
            corr = summary["correlation"]
            print(f"  [OK] computed {len(df)} cells in {summary['elapsed_s']:.1f}s, "
                  f"saved to {out_csv.name}")
            print(f"  [OK] correlation: matched={corr['matched']}, "
                  f"|r| mean={corr['all_mean_abs']:.3f}, "
                  f"mean_sig={corr['mean_sig']:.3f} ({corr['num_significant']} sig features)")

            phases = [p.phase for p in progresses]
            assert phases[0] == "start"
            assert "model_loaded" in phases
            assert phases.count("cell") >= 60
            assert phases[-1] == "done"
            print(f"  [OK] {len(progresses)} progress events emitted")
    finally:
        if original_text is not None:
            DIANCHI_MMD_TOOLS_NEW.write_text(original_text, encoding="utf-8")
        bak = DIANCHI_MMD_TOOLS_NEW.with_suffix(DIANCHI_MMD_TOOLS_NEW.suffix + ".bak")
        if bak.exists():
            bak.unlink()


def main() -> int:
    print("== test_mmd_runner ==")
    tests = [t_inject_and_reload]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as exc:
            failures.append(f"{t.__name__}: {exc}")
            import traceback
            traceback.print_exc()
            print(f"  [FAIL] {t.__name__}: {exc}")
    if failures:
        print(f"FAILED: {len(failures)}/{len(tests)}")
        return 1
    print(f"PASSED: {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
