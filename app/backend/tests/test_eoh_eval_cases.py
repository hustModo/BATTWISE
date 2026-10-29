"""Regression test: Evo eval.py case ids point at the intended data folders."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVO = ROOT / "Evo_Textgrad_battery_eoh_deepseek"
EVAL = EVO / "problems" / "battery_mmd" / "eval.py"


def _load_eval_for_case(case: int):
    old_argv = list(sys.argv)
    try:
        sys.argv = [str(EVAL), str(EVO), "train", str(case), "0"]
        spec = importlib.util.spec_from_file_location(f"battery_mmd_eval_case_{case}", EVAL)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = old_argv


def main() -> int:
    failed = 0
    checks = {
        2: "first_200s",
        4: "AII_200_new",
    }
    for case, folder_name in checks.items():
        mod = _load_eval_for_case(case)
        folder = Path(mod.DATA_FOLDER)
        ok = folder.name == folder_name and folder.exists()
        if not ok:
            print(f"  [FAIL] case {case}: DATA_FOLDER={folder} (want existing */{folder_name})")
            failed += 1
        else:
            n_cells = sum(1 for p in folder.iterdir() if p.is_file() and "Cell" in p.name)
            print(f"  [OK] case {case}: {folder.name}, cells={n_cells}")
            if n_cells != 71:
                print(f"  [FAIL] case {case}: expected 71 cell files, got {n_cells}")
                failed += 1
    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
