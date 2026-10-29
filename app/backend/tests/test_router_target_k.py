"""Unit test: RouterAgent extracts a cluster `target_k` from natural language.

Verifies that phrases like:
  - "Run clustering with k=6"
  - "Cluster into 6 groups."
  - "Split the cells into 4 categories."
  - "Run kmeans clustering with 6 clusters."
all route to intent="cluster" AND attach decision["target_k"] = 6 (or 4),
while a plain "Run clustering." stays adaptive (no target_k key).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.agents.router_agent import _heuristic, _extract_target_k


CASES_WITH_K: list[tuple[str, int]] = [
    ("Run clustering with k=6.", 6),
    ("run kmeans clustering with 6 clusters", 6),
    ("Cluster into 6 groups.", 6),
    ("Group cells into 4 clusters.", 4),
    ("Re-run clustering into 8 clusters please.", 8),
    ("k = 3", 3),
]

CASES_NO_K = [
    "Run clustering.",
    "Ready to run clustering.",
    "Do clustering now.",
]


def main() -> int:
    failed = 0
    for text, want_k in CASES_WITH_K:
        h = _heuristic(text)
        # Plain k-only ("k = 3") doesn't trigger the `cluster` keyword pattern,
        # so we cover it via _extract_target_k directly. The other cases must
        # land on intent='cluster' AND carry target_k.
        if text.strip().lower().startswith("k"):
            got = _extract_target_k(text)
            ok = got == want_k
        else:
            ok = h.get("intent") == "cluster" and h.get("target_k") == want_k
            got = h.get("target_k")
        flag = "OK" if ok else "FAIL"
        print(f"  [{flag}] {text!r} → intent={h.get('intent')}, target_k={got} (want {want_k})")
        if not ok:
            failed += 1

    for text in CASES_NO_K:
        h = _heuristic(text)
        ok = h.get("intent") == "cluster" and "target_k" not in h
        flag = "OK" if ok else "FAIL"
        print(f"  [{flag}] {text!r} → intent={h.get('intent')}, target_k={h.get('target_k')} (want absent)")
        if not ok:
            failed += 1

    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
