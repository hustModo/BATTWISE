"""ChatAgent latency + streaming guardrails.

The user explicitly wants intelligent LLM-backed chat, not hard-coded
responses, AND a fluid streaming reply. These tests pin the contract:

* non-trivial chat goes through ``llm_client.chat_stream`` with the
  short chat timeout + token cap + ``purpose="chat"``;
* the stream emits ``chat_message_started`` and ``chat_delta`` events;
* the persisted message uses the same id as the streamed bubble;
* trivial greetings keep the local quick path (no LLM, no streaming).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.agents.chat_agent import ChatAgent  # noqa: E402
from app.backend.orchestrator.session_store import Session  # noqa: E402
from app.backend.settings import settings  # noqa: E402


def _make_async_iter(deltas: List[str]) -> Any:
    """Build an object that ``async for`` consumes, yielding the deltas."""

    async def _gen() -> AsyncIterator[str]:
        for d in deltas:
            yield d

    return _gen()


def _run_chat(
    text: str,
    deltas: List[str] | None = None,
) -> Dict[str, Any]:
    deltas = deltas if deltas is not None else ["你好，", "我是助手", "。"]
    captured_kwargs: Dict[str, Any] = {}
    emitted: List[tuple[str, Dict[str, Any]]] = []
    say_calls: List[Dict[str, Any]] = []

    def fake_stream(*_args, **kwargs) -> Any:
        captured_kwargs.update(kwargs)
        return _make_async_iter(list(deltas))

    async def fake_emit(kind: str, **payload: Any) -> None:
        emitted.append((kind, payload))

    async def fake_say(text_: str, **kwargs: Any) -> Dict[str, Any]:
        say_calls.append({"text": text_, **kwargs})
        return {"id": kwargs.get("message_id"), "text": text_, "role": "assistant"}

    async def _go() -> Dict[str, Any]:
        session = Session(session_id="test-chat-stream")
        session.add_message("user", text, agent=None)
        agent = ChatAgent(session)
        with patch.object(agent, "started", new=AsyncMock()), \
             patch.object(agent, "say", side_effect=fake_say), \
             patch.object(agent, "emit", side_effect=fake_emit), \
             patch.object(agent, "thinking", new=AsyncMock()), \
             patch(
                 "app.backend.agents.chat_agent.llm_client.chat_stream",
                 side_effect=fake_stream,
             ) as mock_stream:
            result = await agent.run(text)
        return {
            "result": result,
            "captured": captured_kwargs,
            "emitted": emitted,
            "say_calls": say_calls,
            "stream_called": mock_stream.called,
        }

    return asyncio.run(_go())


def test_nontrivial_chat_streams_with_short_timeout() -> int:
    info = _run_chat("这个系统现在可以帮我做什么？")
    assert info["stream_called"] is True
    captured = info["captured"]
    assert captured["timeout"] == float(settings.llm_chat_timeout), captured
    assert captured["total_timeout"] >= 60.0, captured
    assert captured["max_tokens"] == 700, captured
    assert captured["purpose"] == "chat", captured
    # full reply matches concatenated deltas
    assert info["result"]["reply"] == "你好，我是助手。", info["result"]
    assert info["result"]["source"] == "stream"
    # emitted lifecycle: started, ≥1 delta, ended (or message)
    kinds = [k for k, _ in info["emitted"]]
    assert kinds[0] == "chat_message_started", kinds
    assert "chat_delta" in kinds, kinds
    assert "chat_message_ended" in kinds, kinds
    # the say call reuses the streaming message id
    started_payload = next(p for k, p in info["emitted"] if k == "chat_message_started")
    assert info["say_calls"][0]["message_id"] == started_payload["id"]
    return 0


def test_greeting_quick_reply_skips_llm_and_streaming() -> int:
    info = _run_chat("你好")
    assert info["stream_called"] is False
    assert info["result"]["source"] == "quick_reply", info["result"]
    # No streaming lifecycle events for the canned path.
    kinds = [k for k, _ in info["emitted"]]
    assert "chat_message_started" not in kinds, kinds
    assert "chat_delta" not in kinds, kinds
    return 0


def test_stream_failure_emits_aborted_end() -> int:
    def boom(*_args, **_kwargs):
        async def _gen():
            raise TimeoutError("simulated slow LLM")
            yield ""  # pragma: no cover

        return _gen()

    emitted: List[tuple[str, Dict[str, Any]]] = []

    async def fake_emit(kind: str, **payload: Any) -> None:
        emitted.append((kind, payload))

    async def _go() -> None:
        session = Session(session_id="test-chat-stream-fail")
        session.add_message("user", "讲讲你能做什么", agent=None)
        agent = ChatAgent(session)
        with patch.object(agent, "started", new=AsyncMock()), \
             patch.object(agent, "say", new=AsyncMock(return_value={"id": "fallback"})), \
             patch.object(agent, "emit", side_effect=fake_emit), \
             patch.object(agent, "thinking", new=AsyncMock()), \
             patch(
                 "app.backend.agents.chat_agent.llm_client.chat_stream",
                 side_effect=boom,
             ):
            await agent.run("讲讲你能做什么")

    asyncio.run(_go())
    kinds = [k for k, _ in emitted]
    ended = [p for k, p in emitted if k == "chat_message_ended"]
    assert ended and ended[-1].get("aborted") is True, ended
    assert kinds[-1] in {"agent_finished", "chat_message_ended"}, kinds
    return 0


def test_stream_timeout_after_partial_persists_partial() -> int:
    async def partial_then_timeout():
        yield "已经收到一部分回答"
        raise TimeoutError("simulated idle timeout after partial")

    emitted: List[tuple[str, Dict[str, Any]]] = []
    say_calls: List[Dict[str, Any]] = []

    def fake_stream(*_args, **_kwargs):
        return partial_then_timeout()

    async def fake_emit(kind: str, **payload: Any) -> None:
        emitted.append((kind, payload))

    async def fake_say(text_: str, **kwargs: Any) -> Dict[str, Any]:
        say_calls.append({"text": text_, **kwargs})
        return {"id": kwargs.get("message_id"), "text": text_, "role": "assistant"}

    async def _go() -> Dict[str, Any]:
        session = Session(session_id="test-chat-stream-partial")
        session.add_message("user", "详细介绍一下系统", agent=None)
        agent = ChatAgent(session)
        with patch.object(agent, "started", new=AsyncMock()), \
             patch.object(agent, "say", side_effect=fake_say), \
             patch.object(agent, "emit", side_effect=fake_emit), \
             patch.object(agent, "thinking", new=AsyncMock()), \
             patch(
                 "app.backend.agents.chat_agent.llm_client.chat_stream",
                 side_effect=fake_stream,
             ):
            return await agent.run("详细介绍一下系统")

    result = asyncio.run(_go())
    assert result["reply"] == "已经收到一部分回答", result
    assert result["source"] == "stream_partial", result
    assert say_calls and say_calls[0]["text"] == "已经收到一部分回答", say_calls
    ended = [p for k, p in emitted if k == "chat_message_ended"]
    assert ended and ended[-1].get("partial") is True, ended
    assert ended[-1].get("aborted") is False, ended
    return 0


def main() -> int:
    tests = [
        test_nontrivial_chat_streams_with_short_timeout,
        test_greeting_quick_reply_skips_llm_and_streaming,
        test_stream_failure_emits_aborted_end,
        test_stream_timeout_after_partial_persists_partial,
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
