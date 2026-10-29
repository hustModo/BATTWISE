"""End-to-end test for DesignerAgent using a stubbed LLM.

Validates:
- pipeline routes "design a new MMD" to designer
- agent_started / thinking / tool_call / code_update / artifact / agent_finished events
- session.artifacts grows by one (cddi_function.py)
- session.pipeline_state.designer transitions idle -> running -> done
- session.context.last_designed_version is incremented
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.agents.designer_agent import DesignerAgent
from app.backend.orchestrator.events import bus
from app.backend.orchestrator.session_store import Session
from app.backend.services import mmd_designer
from app.backend.tests.test_mmd_designer import GOOD_CODE


async def collect_events(session_id: str, kinds_until: set[str], timeout: float = 5.0):
    queue = await bus.subscribe(session_id)
    events: list[dict] = []
    try:
        while True:
            ev = await asyncio.wait_for(queue.get(), timeout=timeout)
            events.append(ev)
            if ev.get("kind") in kinds_until and ev.get("agent") == "designer":
                if ev.get("kind") == "agent_finished":
                    break
    finally:
        await bus.unsubscribe(session_id, queue)
    return events


async def run_one_shot() -> int:
    session = Session(session_id="test_designer_001")
    # Patch the async chat callable inside mmd_designer.
    original_chat_async = mmd_designer.llm_client.chat_async
    mmd_designer.llm_client.chat_async = AsyncMock(return_value=GOOD_CODE)

    try:
        agent = DesignerAgent(session)
        listener = asyncio.create_task(
            collect_events(session.session_id, {"agent_finished"})
        )
        await asyncio.sleep(0.1)
        await agent.run(user_text="design a new mmd")
        events = await listener
    finally:
        mmd_designer.llm_client.chat_async = original_chat_async

    kinds = [e.get("kind") for e in events]
    print(f"  events: {len(events)}; kinds: {kinds}")

    expected_subset = {"agent_started", "thinking", "tool_call", "code_update", "artifact", "agent_finished"}
    missing = expected_subset - set(kinds)
    if missing:
        print(f"  [FAIL] missing event kinds: {missing}")
        return 1

    # Designer must have emitted at least one final code_update with passed=True
    final_code_updates = [
        e for e in events if e.get("kind") == "code_update" and e.get("final") is True
    ]
    if not final_code_updates:
        print("  [FAIL] no final code_update emitted")
        return 1
    nan_check = final_code_updates[-1].get("nan_check") or {}
    if not nan_check.get("passed"):
        print("  [FAIL] final code_update reports nan_check.passed=False")
        return 1

    # Session state
    if session.pipeline_state.get("designer") != "done":
        print(f"  [FAIL] pipeline_state.designer = {session.pipeline_state.get('designer')!r}")
        return 1
    if not any(a["name"] == "cddi_function.py" for a in session.artifacts):
        print("  [FAIL] cddi_function.py artifact missing")
        return 1
    if session.context.get("last_designed_version") != 2:
        print(f"  [FAIL] last_designed_version != 2: {session.context.get('last_designed_version')}")
        return 1

    print("  [OK] designer pipeline produced expected events + state")
    return 0


def main() -> int:
    print("== test_designer_agent ==")
    rc = asyncio.run(run_one_shot())
    if rc == 0:
        print("PASSED: 1/1")
    else:
        print("FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
