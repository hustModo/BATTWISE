"""Tests for the DDIA design parameter confirmation gate.

The UI prompt tells the user they may reply with "开始设计", but real
users often echo the visible parameters too, e.g. "开始设计 lightweight,
repair_attempts=5, sandbox=3". The confirmation gate must treat that as
approval and dispatch DesignerAgent instead of repeating the same prompt.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.orchestrator import pipeline  # noqa: E402
from app.backend.orchestrator.session_store import Session  # noqa: E402


class _RouterStub:
    def __init__(self, decision: Dict[str, Any]):
        self._decision = decision

    def __call__(self, session: Session):
        return self

    async def run(self, user_text: str) -> Dict[str, Any]:  # noqa: D401
        return dict(self._decision)


def _patch_pipeline(router_decision: Dict[str, Any]):
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch(
        "app.backend.orchestrator.pipeline.RouterAgent",
        _RouterStub(router_decision),
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
        "app.backend.orchestrator.pipeline.conversation_memory.refresh_in_background",
        new=lambda *a, **kw: None,
    ))
    stack.enter_context(patch(
        "app.backend.orchestrator.pipeline._narrate",
        new=AsyncMock(),
    ))

    designer_runs: List[Dict[str, Any]] = []

    class _DesignerStub:
        def __init__(self, session: Session):
            self.session = session

        async def run(self, *, user_text: str, **kwargs):  # noqa: D401
            designer_runs.append({"user_text": user_text, **kwargs})
            return {"passed": True, "stub": True}

    stack.enter_context(patch.dict(
        pipeline._AGENT_REGISTRY,
        {"designer": _DesignerStub},
    ))
    return stack, designer_runs


def _new_session(sid: str) -> Session:
    s = Session(session_id=sid)
    s.mode = "lightweight"
    s.active_dataset_id = "builtin-first-200"
    s.context["pending_design_confirmation"] = True
    return s


def test_natural_design_confirm_sentence_dispatches_designer() -> int:
    s = _new_session("t-design-gate-natural-confirm")
    # Even if the router thinks this is plain chat, the armed design gate
    # should recognize the confirmation phrase and run DesignerAgent.
    decision = {
        "intent": "chitchat",
        "next_agent": "chat",
        "source": "llm-classify",
        "confidence": 0.90,
    }
    stack, runs = _patch_pipeline(decision)
    with stack:
        result = asyncio.run(
            pipeline.handle_message(
                s,
                "开始设计 lightweight，repair_attempts=5，sandbox_cells=3，就按这个开始",
            )
        )
    assert result["routed_to"] == "designer", result
    assert "pending_design_confirmation" not in s.context
    assert len(runs) == 1, runs
    assert "开始设计" in runs[0]["user_text"]
    assert runs[0]["max_attempts"] == 5
    assert runs[0]["sandbox_cells"] == 3
    return 0


def test_full_eoh_param_tuning_keeps_gate_open() -> int:
    s = _new_session("t-design-gate-param-tuning")
    decision = {
        "intent": "design_mmd",
        "next_agent": "designer",
        "source": "llm-classify",
        "confidence": 0.90,
        "eoh_params": {"max_fe": 5},
    }
    stack, runs = _patch_pipeline(decision)
    with stack:
        result = asyncio.run(pipeline.handle_message(s, "切换 Full EOH，5 代"))
    assert result["routed_to"] == "designer", result
    assert result["result"]["awaiting"] == "design_parameter_confirmation"
    assert s.context.get("pending_design_confirmation") is True
    assert runs == [], runs
    assert s.context["eoh_user_params"]["max_fe"] == 5
    return 0


def test_lightweight_param_tuning_keeps_gate_open_even_if_router_says_chat() -> int:
    s = _new_session("t-design-gate-light-param-tuning")
    # Router can misclassify a bare parameter sentence as chat; the armed
    # design gate must still parse lightweight params locally.
    decision = {
        "intent": "chitchat",
        "next_agent": "chat",
        "source": "llm-classify",
        "confidence": 0.90,
    }
    stack, runs = _patch_pipeline(decision)
    with stack:
        result = asyncio.run(
            pipeline.handle_message(s, "轻量化 repair_attempts=6 sandbox_cells=5")
        )
    assert result["routed_to"] == "designer", result
    assert result["result"]["awaiting"] == "design_parameter_confirmation"
    assert s.context.get("pending_design_confirmation") is True
    assert runs == [], runs
    assert s.context["designer_user_params"]["repair_attempts"] == 6
    assert s.context["designer_user_params"]["sandbox_cells"] == 5
    assert "repair_attempts=6" in s.messages[-1]["text"]
    assert "sandbox_cells=5" in s.messages[-1]["text"]
    return 0


def test_sticky_lightweight_params_forward_to_designer_on_confirm() -> int:
    s = _new_session("t-design-gate-light-sticky-forward")
    s.context["designer_user_params"] = {
        "repair_attempts": 7,
        "sandbox_cells": 4,
    }
    decision = {
        "intent": "chitchat",
        "next_agent": "chat",
        "source": "llm-classify",
        "confidence": 0.90,
    }
    stack, runs = _patch_pipeline(decision)
    with stack:
        result = asyncio.run(pipeline.handle_message(s, "确认"))
    assert result["routed_to"] == "designer", result
    assert len(runs) == 1, runs
    assert runs[0]["max_attempts"] == 7
    assert runs[0]["sandbox_cells"] == 4
    return 0


def main() -> int:
    tests = [
        test_natural_design_confirm_sentence_dispatches_designer,
        test_full_eoh_param_tuning_keeps_gate_open,
        test_lightweight_param_tuning_keeps_gate_open_even_if_router_says_chat,
        test_sticky_lightweight_params_forward_to_designer_on_confirm,
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
