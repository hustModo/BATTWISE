"""Unit test: backend dataset ids map to the correct Evo eval.py case ids."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.agents.designer_agent import _dataset_to_eoh_case


def main() -> int:
    cases = {
        "builtin-first-200s": 2,
        "builtin-all-200": 4,
        "unknown-custom": 2,
    }
    failed = 0
    for ds_id, want in cases.items():
        got = _dataset_to_eoh_case(ds_id)
        if got != want:
            print(f"  [FAIL] {ds_id} -> {got}, want {want}")
            failed += 1
        else:
            print(f"  [OK] {ds_id} -> case {got}")
    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
