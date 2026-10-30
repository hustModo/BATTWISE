"""Offline tests for eoh_runner.parse_log_line + load_best_code_from_json.

Uses the real ``battwise_cddi_search/outputs/output/<ts>/main.log``
files already on disk so no subprocess is spawned.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.services.eoh_runner import (  # noqa: E402
    EOHEvent,
    load_best_code_from_json,
    parse_lines,
    parse_log_line,
)
from app.backend.settings import EVO_DIR  # noqa: E402


def test_individual_lines() -> None:
    cases = [
        (
            "[2025-12-13 23:28:38,039][root][INFO] - Generation 0 (Initial) - BEST SOLUTION:",
            {"phase": "generation", "generation": 0, "total_generations": None},
        ),
        (
            "[2025-12-13 23:42:14,229][root][INFO] - Generation 5/20 - BEST SOLUTION:",
            {"phase": "generation", "generation": 5, "total_generations": 20},
        ),
        (
            "[2025-12-13 23:42:14,229][root][INFO] -   Objective Value: inf",
            {"phase": "objective", "objective": float("inf")},
        ),
        (
            "[2025-12-13 23:42:14,229][root][INFO] -   Objective Value: 0.837",
            {"phase": "objective", "objective": 0.837},
        ),
        (
            "[2025-12-13 23:42:14,229][root][INFO] -   Saved to: /tmp/best_population_generation_5.json",
            {"phase": "best_path", "generation": 5},
        ),
        (
            "[2025-12-13 23:28:35,422][root][INFO] - Iteration 0, response_id 17: Objective value: 0.84",
            {"phase": "iter_eval", "iter_idx": 0, "response_id": 17, "objective": 0.84},
        ),
        (
            "[2025-12-14 00:22:50,623][root][INFO] - Best Code Overall: import numpy as np",
            {"phase": "best_code"},
        ),
        (
            "[2025-12-14 00:22:50,623][root][INFO] - Best Objective Overall: 0.84",
            {"phase": "finished", "objective": 0.84},
        ),
        (
            "[2025-12-13 23:05:09,987][root][INFO] - Starting EOH-Textgrad evolution...",
            {"phase": "startup"},
        ),
        (
            "[2026-07-23 12:00:00,000][root][INFO] - Starting BATTWISE Reflective Metric Search...",
            {"phase": "startup"},
        ),
        (
            "[2026-07-23 12:00:01,000][root][INFO] - BATTWISE Reflective Metric Search completed",
            {"phase": "finished"},
        ),
        (
            "[2025-12-13 23:05:09,951][root][INFO] - Case num: [1]",
            {"phase": "case", "case_num": [1]},
        ),
        (
            "[ts][httpx][INFO] HTTP Request: POST ...",
            None,
        ),
    ]
    for line, expected in cases:
        ev = parse_log_line(line)
        if expected is None:
            assert ev is None, f"expected None for noise line, got {ev}"
            continue
        assert ev is not None, f"failed to parse: {line!r}"
        for k, v in expected.items():
            assert getattr(ev, k) == v, f"{k!r} mismatch on {line!r}: got {getattr(ev, k)!r}, want {v!r}"
    print(f"[1] {len(cases)} canonical lines parsed correctly")


def test_real_log_walk() -> None:
    """Walk the longest sample main.log we have on disk; ensure parser
    recognises at least the expected high-level events."""
    root = EVO_DIR / "outputs" / "output"
    if not root.exists():
        print("[skip] no outputs/output dir")
        return
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "main.log").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        print("[skip] no main.log on disk")
        return
    # pick the largest log we have
    log_path = max(
        (p / "main.log" for p in candidates),
        key=lambda p: p.stat().st_size,
    )
    print(f"     parsing {log_path}")
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    events = parse_lines(lines)
    phases = [e.phase for e in events]
    summary = {p: phases.count(p) for p in set(phases)}
    print(f"     parsed {len(events)} events: {summary}")
    assert "generation" in summary, f"no generation events in {summary}"
    assert "objective" in summary or "iter_eval" in summary, summary
    # At least the first generation should be 0
    first_gen = next((e for e in events if e.phase == "generation"), None)
    assert first_gen is not None and first_gen.generation == 0, first_gen


def test_load_best_code_from_json() -> None:
    root = EVO_DIR / "outputs" / "output"
    if not root.exists():
        print("[skip] no outputs/output dir")
        return
    candidates: list[Path] = []
    for d in root.iterdir():
        if d.is_dir():
            for p in d.glob("best_population_generation_*.json"):
                candidates.append(p)
    if not candidates:
        print("[skip] no best_population_generation json")
        return
    sample = candidates[0]
    code = load_best_code_from_json(sample)
    assert code is not None, f"no code field in {sample}"
    assert "def " in code or "import " in code, f"loaded code looks empty: {code[:100]}"
    print(f"[3] loaded {len(code)} chars of code from {sample.name}")


if __name__ == "__main__":
    test_individual_lines()
    test_real_log_walk()
    test_load_best_code_from_json()
    print("All EOH parser tests passed.")
