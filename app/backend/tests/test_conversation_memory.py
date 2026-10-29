"""Unit tests for the running-summary service (``services.conversation_memory``).

Mocks the LLM call. Covers:
  * _parse_summary handles fenced / unfenced JSON and rejects garbage
  * refresh stores the summary on session.context and publishes the WS event
  * throttling: a second refresh inside SUMMARY_MIN_INTERVAL_S is skipped
  * force=True bypasses throttling
  * LLM exception is non-fatal — previous summary is preserved
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.orchestrator.session_store import Session  # noqa: E402
from app.backend.services import conversation_memory  # noqa: E402


def _make_session() -> Session:
    s = Session(session_id="test-conv-mem")
    s.messages = [
        {"id": "1", "role": "user", "agent": None, "text": "design ddia for first 200s", "t": 0},
        {"id": "2", "role": "assistant", "agent": "designer",
         "text": "DDIA quality check passed |r|=0.83", "t": 0},
        {"id": "3", "role": "user", "agent": None, "text": "cluster with k=4", "t": 0},
    ]
    return s


def test_parse_summary_plain() -> int:
    raw = (
        '{"what_user_wants": "Run DDIA + clustering on First-200s",'
        ' "what_was_done": ["DDIA designed (|r|=0.83)"],'
        ' "current_state": "Waiting for clustering output",'
        ' "user_preferences": {"feedback_metric": "variance"},'
        ' "open_questions": []}'
    )
    p = conversation_memory._parse_summary(raw)
    assert p is not None, "expected parsed summary"
    assert p["what_user_wants"].startswith("Run DDIA"), p
    assert "DDIA designed" in p["what_was_done"][0], p
    assert p["user_preferences"] == {"feedback_metric": "variance"}, p
    return 0


def test_parse_summary_fenced() -> int:
    raw = """```json
{"what_user_wants": "X", "what_was_done": [], "current_state": "", "user_preferences": {}, "open_questions": []}
```"""
    p = conversation_memory._parse_summary(raw)
    assert p is not None and p["what_user_wants"] == "X", p
    return 0


def test_parse_summary_rejects_garbage() -> int:
    assert conversation_memory._parse_summary("not JSON") is None
    assert conversation_memory._parse_summary("") is None
    assert conversation_memory._parse_summary("[1,2,3]") is None  # top-level array
    return 0


def test_refresh_stores_summary() -> int:
    reply = (
        '{"what_user_wants": "DDIA + cluster", "what_was_done": ["designed"],'
        ' "current_state": "clustering", "user_preferences": {"k": 4},'
        ' "open_questions": []}'
    )
    s = _make_session()

    with patch("app.backend.services.conversation_memory.llm_client.chat_async",
               new=AsyncMock(return_value=reply)), \
         patch("app.backend.services.conversation_memory.store.update", new=AsyncMock()), \
         patch("app.backend.services.conversation_memory.bus.publish", new=AsyncMock()):
        result = asyncio.run(conversation_memory.refresh(s, force=True))

    assert result is not None, "refresh should return a summary"
    assert result["what_user_wants"] == "DDIA + cluster", result
    assert s.context["conversation_summary"]["current_state"] == "clustering"
    assert s.context["conversation_summary_t"] > 0
    return 0


def test_refresh_throttle_skips() -> int:
    s = _make_session()
    s.context["conversation_summary"] = {"what_user_wants": "prev", "what_was_done": [],
                                          "current_state": "", "user_preferences": {},
                                          "open_questions": []}
    s.context["conversation_summary_t"] = time.time()  # just refreshed

    chat_calls = []

    async def _chat_spy(*a, **kw):
        chat_calls.append((a, kw))
        return '{"what_user_wants": "new"}'

    with patch("app.backend.services.conversation_memory.llm_client.chat_async",
               side_effect=_chat_spy):
        result = asyncio.run(conversation_memory.refresh(s))

    assert chat_calls == [], "throttled refresh must not call the LLM"
    assert result["what_user_wants"] == "prev", "previous summary should be returned"
    return 0


def test_refresh_force_bypasses_throttle() -> int:
    s = _make_session()
    s.context["conversation_summary_t"] = time.time()

    reply = ('{"what_user_wants": "fresh", "what_was_done": [], "current_state": "",'
             ' "user_preferences": {}, "open_questions": []}')

    with patch("app.backend.services.conversation_memory.llm_client.chat_async",
               new=AsyncMock(return_value=reply)), \
         patch("app.backend.services.conversation_memory.store.update", new=AsyncMock()), \
         patch("app.backend.services.conversation_memory.bus.publish", new=AsyncMock()):
        result = asyncio.run(conversation_memory.refresh(s, force=True))

    assert result["what_user_wants"] == "fresh", result
    return 0


def test_refresh_llm_error_preserves_previous() -> int:
    s = _make_session()
    prev = {"what_user_wants": "prev", "what_was_done": [], "current_state": "",
            "user_preferences": {}, "open_questions": []}
    s.context["conversation_summary"] = prev

    with patch(
        "app.backend.services.conversation_memory.llm_client.chat_async",
        new=AsyncMock(side_effect=RuntimeError("LLM down")),
    ):
        result = asyncio.run(conversation_memory.refresh(s, force=True))

    assert result == prev, "previous summary must be preserved on LLM failure"
    return 0


def main() -> int:
    tests = [
        test_parse_summary_plain,
        test_parse_summary_fenced,
        test_parse_summary_rejects_garbage,
        test_refresh_stores_summary,
        test_refresh_throttle_skips,
        test_refresh_force_bypasses_throttle,
        test_refresh_llm_error_preserves_previous,
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
