"""Conversational cancel must also clear pending confirmation gates.

Before this fix, "取消" / "stop" / "停" while waiting on the design or
cluster parameter gate hit the conversational-cancel branch, posted
"No active computation is running right now.", and returned — but the
``pending_*_confirmation`` flag stayed armed. The next user message
then ran into the gate's "you're still waiting on confirmation" nudge
and the user couldn't actually escape.

The cancel branch now pops both gates and acknowledges what was
cancelled. These tests pin that behaviour.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.orchestrator import pipeline  # noqa: E402
from app.backend.orchestrator.session_store import Session  # noqa: E402


def _cancel_decision() -> Dict[str, Any]:
    return {
        "intent": "cancel",
        "next_agent": "cancel",
        "source": "llm-classify",
        "confidence": 0.92,
    }


class _RouterStub:
    def __init__(self, decision: Dict[str, Any]):
        self._d = decision

    def __call__(self, session: Session):
        return self

    async def run(self, user_text: str) -> Dict[str, Any]:  # noqa: D401
        return dict(self._d)


def _patches():
    """Common patches: router stub, store/bus silenced, summary noop."""
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch(
        "app.backend.orchestrator.pipeline.RouterAgent",
        _RouterStub(_cancel_decision()),
    ))
    stack.enter_context(patch(
        "app.backend.orchestrator.pipeline.store.update",
        new=AsyncMock(),
    ))
    stack.enter_context(patch(
        "app.backend.orchestrator.pipeline.bus.publish",
        new=AsyncMock(),
    ))
    stack.enter_context(patch(
        "app.backend.orchestrator.pipeline.conversation_memory."
        "refresh_in_background",
        new=lambda *a, **kw: None,
    ))
    return stack


def test_cancel_clears_pending_design_gate() -> int:
    s = Session(session_id="t-cancel-gate-1")
    s.context["pending_design_confirmation"] = True
    with _patches():
        result = asyncio.run(pipeline.handle_message(s, "取消"))
    assert result["routed_to"] == "cancel"
    assert "pending_design_confirmation" not in s.context, s.context
    assistant = [m for m in s.messages if m["role"] == "assistant"][-1]["text"]
    assert "design" in assistant
    return 0


def test_cancel_clears_pending_cluster_gate() -> int:
    s = Session(session_id="t-cancel-gate-2")
    s.context["pending_cluster_confirmation"] = True
    with _patches():
        result = asyncio.run(pipeline.handle_message(s, "stop"))
    assert result["routed_to"] == "cancel"
    assert "pending_cluster_confirmation" not in s.context
    last = [m for m in s.messages if m["role"] == "assistant"][-1]["text"]
    assert "cluster" in last
    return 0


def test_cancel_clears_both_gates_when_armed() -> int:
    s = Session(session_id="t-cancel-gate-3")
    s.context["pending_design_confirmation"] = True
    s.context["pending_cluster_confirmation"] = True
    with _patches():
        asyncio.run(pipeline.handle_message(s, "算了"))
    assert "pending_design_confirmation" not in s.context
    assert "pending_cluster_confirmation" not in s.context
    last = [m for m in s.messages if m["role"] == "assistant"][-1]["text"]
    assert "design" in last and "cluster" in last
    return 0


def test_cancel_sets_active_flag_AND_clears_gate() -> int:
    """The rare-but-possible 'designer running while cluster gate
    armed' case: cancel both at once."""
    s = Session(session_id="t-cancel-gate-4")
    s.active_agent = "designer"
    s.context["pending_cluster_confirmation"] = True
    with _patches():
        asyncio.run(pipeline.handle_message(s, "停"))
    assert s.context.get("cancel_eoh") is True, s.context
    assert "pending_cluster_confirmation" not in s.context
    last = [m for m in s.messages if m["role"] == "assistant"][-1]["text"]
    assert "stop on its next iteration" in last  # running-agent ack
    assert "cluster" in last                     # gate ack
    return 0


def test_cancel_with_no_gate_no_active_keeps_old_msg() -> int:
    s = Session(session_id="t-cancel-gate-5")
    with _patches():
        asyncio.run(pipeline.handle_message(s, "取消"))
    last = [m for m in s.messages if m["role"] == "assistant"][-1]["text"]
    assert "No active computation" in last
    return 0


def main() -> int:
    tests = [
        test_cancel_clears_pending_design_gate,
        test_cancel_clears_pending_cluster_gate,
        test_cancel_clears_both_gates_when_armed,
        test_cancel_sets_active_flag_AND_clears_gate,
        test_cancel_with_no_gate_no_active_keeps_old_msg,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  [OK] {fn.__name__}")
        except AssertionError as exc:
            print(f"  [FAIL] {fn.__name__}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"  [ERROR] {fn.__name__}: {exc!r}")
            traceback.print_exc()
            failed += 1
    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
