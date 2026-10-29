"""Smoke test: eoh_seeds.build_seeds() produces a non-empty JSON file
with `algorithm` + `code` for each candidate, drawn from the curated
historical run directories.

Soft-asserts: we tolerate missing history dirs (CI environments may not
have them) but require the baseline fallback to keep the file usable.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.services import eoh_seeds


def main() -> int:
    failed = 0
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "seeds.json"
        path = eoh_seeds.build_seeds(out_path=out)
        if not path.exists():
            print(f"  [FAIL] seeds file not written: {path}")
            return 1
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            print(f"  [FAIL] seeds file is empty or not a list: {type(data).__name__}")
            failed += 1
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"  [FAIL] seeds[{i}] not a dict")
                failed += 1
                continue
            if "algorithm" not in item or "code" not in item:
                print(f"  [FAIL] seeds[{i}] missing keys: {list(item)}")
                failed += 1
            if not item.get("code", "").strip():
                print(f"  [FAIL] seeds[{i}].code is empty")
                failed += 1
        print(f"  [OK] {len(data)} seed(s) written to {path}")
        # show first signatures so we can eyeball the pool
        for i, item in enumerate(data[:3]):
            sig = item.get("algorithm", "")[:80]
            print(f"        - [{i}] {sig!r}")

    # Now also check the production location works.
    real_path = eoh_seeds.build_seeds()
    data = json.loads(real_path.read_text(encoding="utf-8"))
    if not data:
        print(f"  [FAIL] real seed file is empty: {real_path}")
        failed += 1
    else:
        print(f"  [OK] production seed file has {len(data)} candidates at {real_path}")

    # Case-specific seed pools: first_200s (case 2) and All-200/AII_200
    # (case 4) must not collapse to the same generic file.
    with tempfile.TemporaryDirectory() as td:
        p2 = eoh_seeds.build_seeds(case_num=2, out_path=Path(td) / "case2.json")
        p4 = eoh_seeds.build_seeds(case_num=4, out_path=Path(td) / "case4.json")
        d2 = json.loads(p2.read_text(encoding="utf-8"))
        d4 = json.loads(p4.read_text(encoding="utf-8"))
        if not d2 or not d4:
            print("  [FAIL] case-specific seed file is empty")
            failed += 1
        elif d2 == d4:
            print("  [FAIL] case 2 and case 4 seed pools are identical")
            failed += 1
        else:
            print(f"  [OK] case-specific seed pools differ (case2={len(d2)}, case4={len(d4)})")

    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
