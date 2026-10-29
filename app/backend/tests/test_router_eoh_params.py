"""Unit test: RouterAgent extracts EOH params (max_fe / pop_size / batch_size)
from natural-language utterances, in both English and Chinese.

The decision payload should:
  * route to ``design_mmd`` when the message includes a Full-EOH trigger
    ("run full EOH", "跑一次 DDIA", ...)
  * carry an ``eoh_params`` dict with whichever knobs the user mentioned
  * NOT extract params when the EOH context is absent (so "design MMD with
    3 attempts" doesn't get misread as max_fe=3).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.agents.router_agent import (
    _extract_designer_params,
    _extract_eoh_params,
    _heuristic,
)


# (text, want_intent, want_params)
CASES: list[tuple[str, str | None, dict]] = [
    (
        "Run full EOH with 5 generations and pop_size=6.",
        "design_mmd",
        {"max_fe": 5, "pop_size": 6},
    ),
    (
        "Start DDIA, max_fe=3 pop=4 batch=4",
        "design_mmd",
        {"max_fe": 3, "pop_size": 4, "batch_size": 4},
    ),
    (
        "再跑一次 EOH 5 代 每代 6 个体, 批 4",
        "design_mmd",
        {"max_fe": 5, "pop_size": 6, "batch_size": 4},
    ),
    (
        "Run Full EOH",
        "design_mmd",
        {},  # no params mentioned
    ),
    # context guard: a "design MMD with 3 attempts" message shouldn't be
    # misread as max_fe=3, because no EOH context keyword is present.
    (
        "design an MMD function with 3 attempts please",
        "design_mmd",
        {},
    ),
]


def main() -> int:
    failed = 0
    for text, want_intent, want_params in CASES:
        h = _heuristic(text)
        got_intent = h.get("intent")
        got_params = h.get("eoh_params") or {}
        intent_ok = got_intent == want_intent
        # subset check: every key the test demands must be present and equal
        params_ok = all(got_params.get(k) == v for k, v in want_params.items())
        # also: when test wants NO params, ensure none were attached
        if not want_params:
            params_ok = params_ok and not got_params
        ok = intent_ok and params_ok
        flag = "OK" if ok else "FAIL"
        print(
            f"  [{flag}] {text!r}\n"
            f"        intent={got_intent} (want {want_intent}), "
            f"eoh_params={got_params} (want {want_params})"
        )
        if not ok:
            failed += 1

    # direct extractor sanity
    direct = _extract_eoh_params("跑 3 代 4 个体 批次 4")
    if direct != {"max_fe": 3, "pop_size": 4, "batch_size": 4}:
        print(f"  [FAIL] direct CN extract → {direct}")
        failed += 1
    else:
        print(f"  [OK] direct CN extract → {direct}")

    # rejection: implausible value
    if _extract_eoh_params("跑 500 代"):
        print("  [FAIL] should have rejected max_fe=500 as out-of-range")
        failed += 1
    else:
        print("  [OK] rejected max_fe=500 (out of range)")

    light = _extract_designer_params("轻量化 repair_attempts=6 sandbox_cells=5")
    if light != {"repair_attempts": 6, "sandbox_cells": 5}:
        print(f"  [FAIL] lightweight params extract → {light}")
        failed += 1
    else:
        print(f"  [OK] lightweight params extract → {light}")

    h = _heuristic("开始 DDIA lightweight repair_attempts=6 sandbox=5")
    if h.get("intent") != "design_mmd" or h.get("designer_params") != {
        "repair_attempts": 6,
        "sandbox_cells": 5,
    }:
        print(f"  [FAIL] heuristic lightweight params → {h}")
        failed += 1
    else:
        print(f"  [OK] heuristic lightweight params → {h.get('designer_params')}")

    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
