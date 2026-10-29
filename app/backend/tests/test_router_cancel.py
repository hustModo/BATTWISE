"""Unit test: RouterAgent recognises conversational cancel/stop requests
and routes them to the special `cancel` intent, BEFORE any other keyword
match.

If this regresses, "stop the clustering" will re-trigger a fresh
clustering run instead of halting the running one.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.agents.router_agent import _heuristic, _next_agent


CANCEL_CASES = [
    "stop",
    "stop the clustering",
    "abort please",
    "cancel",
    "nevermind",
    "kill it",
    "enough",
    "that's enough",
    "停",
    "停止",
    "终止",
    "算了",
    "够了",
    "不跑了",
    "取消",
    "请停止",
    "帮我停止计算",
    "麻烦取消一下",
    "中止聚类",
]

# Non-cancel: must NOT be misread as cancel.
NON_CANCEL_CASES = [
    "run clustering",
    "design mmd function",
    "compute mmd",
    "what is dbi?",
    "cluster into 6 groups",
    "Run Full EOH with 5 generations",
]


def main() -> int:
    failed = 0
    for text in CANCEL_CASES:
        h = _heuristic(text)
        ok = h.get("intent") == "cancel" and _next_agent("cancel") == "cancel"
        flag = "OK" if ok else "FAIL"
        print(f"  [{flag}] {text!r} → intent={h.get('intent')}")
        if not ok:
            failed += 1

    for text in NON_CANCEL_CASES:
        h = _heuristic(text)
        ok = h.get("intent") != "cancel"
        flag = "OK" if ok else "FAIL"
        print(f"  [{flag}] {text!r} → intent={h.get('intent')} (must not be cancel)")
        if not ok:
            failed += 1

    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
