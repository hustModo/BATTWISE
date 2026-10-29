"""Integration-style tests for the RouterAgent's new smart-routing flow.

We mock llm_router.classify_async so the test doesn't need a network and runs
fast. The point of these tests is to verify that:

1. The cancel regex fast-path bypasses the LLM entirely.
2. Pure greetings ("hi" / "你好") bypass the LLM.
3. Obvious non-action questions go straight to ChatAgent (one Chat LLM,
   no Router LLM).
4. Simple "start working" phrases open Designer directly instead of
   being inflated into a plan by the LLM router.
5. Everything else calls the LLM router; when it returns a confident
   decision, that decision is what flows into the pipeline dict.
6. When the LLM returns low confidence, the legacy regex heuristic is
   used as a fallback so a wrong/missing LLM never deadlocks routing.
7. ``clarification`` round-trips on the router_decision payload.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.agents.router_agent import RouterAgent  # noqa: E402
from app.backend.orchestrator.session_store import Session  # noqa: E402
from app.backend.services import llm_router  # noqa: E402


def _make_session() -> Session:
    return Session(session_id="test-router-smart")


def _run_router(session: Session, text: str):
    async def _go():
        agent = RouterAgent(session)
        # bypass started/finished/emit network plumbing — the WS bus /
        # store would otherwise try to publish to subscribers we don't
        # have here. AsyncMock returns awaitables that resolve to None.
        with patch.object(agent, "started", new=AsyncMock()), \
             patch.object(agent, "thinking", new=AsyncMock()), \
             patch.object(agent, "log", new=AsyncMock()), \
             patch.object(agent, "emit", new=AsyncMock()), \
             patch.object(agent, "finished", new=AsyncMock()):
            return await agent.run(text)
    return asyncio.run(_go())


def test_cancel_fast_path_skips_llm() -> int:
    s = _make_session()
    with patch("app.backend.agents.router_agent.llm_router.classify_async") as mock_llm, \
         patch("app.backend.agents.router_agent.conversation_memory.get_summary", return_value=None):
        d = _run_router(s, "stop the clustering")
    assert d["intent"] == "cancel" and d["next_agent"] == "cancel", d
    assert d["source"] == "regex-cancel", d
    mock_llm.assert_not_called()
    return 0


def test_greeting_fast_path_skips_llm() -> int:
    s = _make_session()
    with patch("app.backend.agents.router_agent.llm_router.classify_async") as mock_llm, \
         patch("app.backend.agents.router_agent.conversation_memory.get_summary", return_value=None):
        d = _run_router(s, "你好")
    assert d["intent"] == "chitchat" and d["next_agent"] == "chat", d
    assert d["source"] == "greeting-fast-path", d
    mock_llm.assert_not_called()
    return 0


def test_obvious_chat_question_skips_router_llm() -> int:
    """A normal product question should still be answered by ChatAgent's LLM,
    but the Router LLM must not sit in front of it and double the latency.
    """
    s = _make_session()
    with patch("app.backend.agents.router_agent.llm_router.classify_async") as mock_llm, \
         patch("app.backend.agents.router_agent.conversation_memory.get_summary", return_value=None):
        d = _run_router(s, "你可以帮我做什么？")
    assert d["intent"] == "chitchat" and d["next_agent"] == "chat", d
    assert d["source"] == "chat-question-fast-path", d
    mock_llm.assert_not_called()
    return 0


def test_simple_start_work_opens_designer_without_router_llm() -> int:
    s = _make_session()
    with patch("app.backend.agents.router_agent.llm_router.classify_async") as mock_llm, \
         patch("app.backend.agents.router_agent.conversation_memory.get_summary", return_value=None):
        d = _run_router(s, "那你可以开始基于 First-200s 数据集开始干活")
    assert d["intent"] == "start", d
    assert d["next_agent"] == "designer", d
    assert d["source"] == "start-work-fast-path", d
    mock_llm.assert_not_called()
    return 0


def test_simple_start_work_keeps_lightweight_params() -> int:
    s = _make_session()
    with patch("app.backend.agents.router_agent.llm_router.classify_async") as mock_llm, \
         patch("app.backend.agents.router_agent.conversation_memory.get_summary", return_value=None):
        d = _run_router(s, "开始轻量化 DDIA repair_attempts=6 sandbox=5")
    assert d["next_agent"] == "designer", d
    assert d["proposed_mode"] == "lightweight", d
    assert d["designer_params"] == {"repair_attempts": 6, "sandbox_cells": 5}, d
    mock_llm.assert_not_called()
    return 0


def test_llm_route_dispatches_designer() -> int:
    s = _make_session()
    fake = llm_router.IntentDecision(
        tool="design_ddia",
        next_agent="designer",
        intent="design_mmd",
        params={"mode": "full_eoh", "max_fe": 5},
        confidence=0.9,
        needs_clarification=False,
        clarification=None,
        reasoning="user asked for Full EOH",
    )
    with patch("app.backend.agents.router_agent.llm_router.classify_async",
               new=AsyncMock(return_value=fake)), \
         patch("app.backend.agents.router_agent.conversation_memory.get_summary", return_value=None):
        d = _run_router(s, "do a full EOH with 5 generations")
    assert d["intent"] == "design_mmd", d
    assert d["next_agent"] == "designer", d
    assert d["source"] == "llm-router", d
    assert d["tool"] == "design_ddia"
    assert d["eoh_params"] == {"max_fe": 5}, d
    assert d["proposed_mode"] == "reflective_search"
    return 0


def test_llm_clarification_round_trip() -> int:
    s = _make_session()
    fake = llm_router.IntentDecision(
        tool="design_ddia",
        next_agent="designer",
        intent="design_mmd",
        params={},
        confidence=0.7,
        needs_clarification=True,
        clarification="你想用 Lightweight 还是 Full EOH 模式？",
        reasoning="ambiguous",
    )
    with patch("app.backend.agents.router_agent.llm_router.classify_async",
               new=AsyncMock(return_value=fake)), \
         patch("app.backend.agents.router_agent.conversation_memory.get_summary", return_value=None):
        d = _run_router(s, "再来一次")
    assert d["needs_clarification"] is True
    assert "Lightweight" in d["clarification"], d
    return 0


def test_low_confidence_falls_back_to_regex() -> int:
    s = _make_session()
    # LLM returns "answer_question" with very low confidence — caller
    # should use the regex heuristic, which sees "run clustering" and
    # picks `cluster`.
    fake = llm_router.IntentDecision(
        tool="answer_question",
        next_agent="chat",
        intent="chitchat",
        params={},
        confidence=0.2,
        needs_clarification=False,
        clarification=None,
        reasoning="not sure",
    )
    with patch("app.backend.agents.router_agent.llm_router.classify_async",
               new=AsyncMock(return_value=fake)), \
         patch("app.backend.agents.router_agent.conversation_memory.get_summary", return_value=None):
        d = _run_router(s, "run clustering with k=6")
    assert d["source"] == "heuristic-fallback", d
    assert d["intent"] == "cluster", d
    assert d["target_k"] == 6, d
    return 0


def test_llm_none_falls_back_to_regex_too() -> int:
    s = _make_session()
    with patch("app.backend.agents.router_agent.llm_router.classify_async",
               new=AsyncMock(return_value=None)), \
         patch("app.backend.agents.router_agent.conversation_memory.get_summary", return_value=None):
        d = _run_router(s, "design mmd function")
    assert d["source"] == "heuristic-fallback", d
    assert d["intent"] == "design_mmd", d
    return 0


def test_long_greeting_not_fast_pathed() -> int:
    """A long message that starts with 'hello' still goes through the LLM —
    it might contain a real request hidden inside the greeting."""
    s = _make_session()
    fake = llm_router.IntentDecision(
        tool="compute_mmd", next_agent="compute", intent="compute_mmd",
        params={}, confidence=0.8, needs_clarification=False,
        clarification=None, reasoning="user asked to compute",
    )
    long_text = "hello can you please run the compute step on the latest mmd function"
    mock_llm = AsyncMock(return_value=fake)
    with patch("app.backend.agents.router_agent.llm_router.classify_async",
               new=mock_llm), \
         patch("app.backend.agents.router_agent.conversation_memory.get_summary",
               return_value=None):
        d = _run_router(s, long_text)
    assert mock_llm.called, "long messages must go through the LLM router"
    assert d["next_agent"] == "compute", d
    return 0


def main() -> int:
    tests = [
        test_cancel_fast_path_skips_llm,
        test_greeting_fast_path_skips_llm,
        test_obvious_chat_question_skips_router_llm,
        test_simple_start_work_opens_designer_without_router_llm,
        test_simple_start_work_keeps_lightweight_params,
        test_llm_route_dispatches_designer,
        test_llm_clarification_round_trip,
        test_low_confidence_falls_back_to_regex,
        test_llm_none_falls_back_to_regex_too,
        test_long_greeting_not_fast_pathed,
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
            print(f"  [ERROR] {fn.__name__}: {exc!r}")
            failed += 1
    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
