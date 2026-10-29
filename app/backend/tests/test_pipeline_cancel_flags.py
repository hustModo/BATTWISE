"""Regression test: conversational cancel only flips the active agent's flag.

Old behavior set cancel_compute/cancel_cluster/cancel_eoh all at once, so a
later unrelated run could be aborted by a stale flag. This test exercises the
pipeline branch directly.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.orchestrator.pipeline import handle_message
from app.backend.orchestrator.session_store import Session


async def _case(active_agent: str | None, want: dict[str, bool]) -> int:
    session = Session(session_id=f"test_cancel_flags_{active_agent or 'none'}")
    session.active_agent = active_agent
    # Simulate stale flags from a prior request; the cancel branch must clear
    # them before setting the currently active agent's flag.
    session.context.update({
        "cancel_compute": True,
        "cancel_cluster": True,
        "cancel_eoh": True,
    })
    result = await handle_message(session, "请停止")
    failed = 0
    if result.get("routed_to") != "cancel":
        print(f"  [FAIL] active={active_agent}: routed_to={result.get('routed_to')}")
        failed += 1
    for key, expected in want.items():
        got = bool(session.context.get(key))
        if got != expected:
            print(f"  [FAIL] active={active_agent}: {key}={got} expected {expected}")
            failed += 1
    if failed == 0:
        flags = {k: session.context.get(k) for k in ("cancel_compute", "cancel_cluster", "cancel_eoh")}
        print(f"  [OK] active={active_agent}: {flags}")
    return failed


async def _main() -> int:
    failed = 0
    failed += await _case("compute", {
        "cancel_compute": True,
        "cancel_cluster": False,
        "cancel_eoh": False,
    })
    failed += await _case("cluster", {
        "cancel_compute": False,
        "cancel_cluster": True,
        "cancel_eoh": False,
    })
    failed += await _case("designer", {
        "cancel_compute": False,
        "cancel_cluster": False,
        "cancel_eoh": True,
    })
    failed += await _case(None, {
        "cancel_compute": False,
        "cancel_cluster": False,
        "cancel_eoh": False,
    })
    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
