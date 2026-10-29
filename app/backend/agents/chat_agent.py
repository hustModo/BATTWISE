"""Chat Agent: fallback Q&A + small-talk companion.

This agent runs whenever the router classifies the user's message as
`chitchat`. It has two jobs in roughly equal measure:

  1) Small-talk / chit-chat: greetings, jokes, off-topic chat. Reply
     naturally, in the user's language.
  2) Pipeline guidance: when the user asks how something works, refer
     to the real pipeline state pulled from the live session.

Replies are streamed token-by-token so the user sees the assistant
"thinking" instead of staring at a spinner for 30+ seconds. Identity
questions ("你是谁 / who are you") used to be answered with a hard-coded
canned string — per user feedback ("感觉是提前内置的") they now go
through the LLM so every reply feels alive.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from app.backend.agents.base import AgentBase
from app.backend.services import llm_client
from app.backend.settings import settings

logger = logging.getLogger(__name__)


_CHAT_SYSTEM = """You are the BATTWISE assistant — a friendly,
conversational helper embedded in an industrial multi-agent battery
sorting and recycling platform. You serve two roles:

  • Casual conversation: greetings, jokes, small talk, encouragement.
    Reply like a real person would — warm, brief, but not robotic.
    It's fine to be playful.
  • Pipeline guidance: when the user asks how the system works, refer
    to the actual pipeline state (provided below in `Current context`).

The system is BATTWISE and the discovered curve-discrepancy indicator is
CDDI. Refer to the combined method as BATTWISE-CDDI. Use CDDI for the
indicator in replies; MMD names the underlying mathematical concept when
explaining the implementation.

Pipeline stages (in order):
  1. Designer  — an LLM designs a CDDI scoring function for the
     battery charging-curve segment.
  2. Compute   — runs the CDDI function across all cell files →
     cddi_results.csv (legacy file may be named mmd_results.csv).
  3. Cluster   — adaptive GA-KMeans clusters cells using
     OCV + IR + CDDI → Cluster_label.xlsx.
  4. Refine    — user picks one or more clusters of interest; system
     requests All-200s data for those cells and re-clusters the
     subset → Sub_Cluster_*_label.xlsx.

LANGUAGE: Mirror the user's latest message. If the latest user message
is English, reply in English, even for a short greeting like "hi" or
"hello". If the latest user message is Chinese (Simplified or
Traditional), reply in Simplified Chinese. If the message mixes both,
use the dominant language and keep any technical terms unchanged.

LENGTH: Match the user's energy. One-line answers for greetings,
two or three sentences for "what is X" style questions, a short list
for "how do I…" walk-throughs. Avoid wall-of-text replies.

If the user is clearly confused about how to start: tell them to
upload (or pick) a First-200s dataset in the top bar, then send the
message "start" to wake up the Designer.
"""


# How many prior turns (each turn = one user msg or one assistant msg)
# we replay back to the LLM for context. Six covers ~3 turns of
# back-and-forth, which is enough for short threads ("…what about
# the last cluster?") without blowing the context window.
_CHAT_HISTORY_MAX = 8

# Only keep TRULY trivial greetings on the local fast path. Identity
# questions ("你是谁") and capability questions ("你能做什么") now go
# through the LLM so they feel intelligent. Single-token "hi" / "你好"
# stays canned because no amount of LLM polish improves it.
_EN_GREETING_RE = re.compile(r"^\s*(hi|hello|hey|yo)[\s!.?]*$", re.I)
_ZH_GREETING_RE = re.compile(r"^\s*(你好|您好|嗨|哈喽|哈囉)[。！？!?\s]*$")


class ChatAgent(AgentBase):
    agent_id = "chat"
    node_id = None

    async def run(self, user_text: str) -> Dict[str, Any]:  # type: ignore[override]
        await self.started()
        quick_reply = self._quick_reply(user_text)
        if quick_reply is not None:
            await self.say(quick_reply)
            await self.emit("agent_finished", result={"reply": quick_reply, "source": "quick_reply"})
            return {"reply": quick_reply, "source": "quick_reply"}

        await self.thinking("正在生成回答…")
        context = self._build_context_summary()
        history = self._recent_history()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": _CHAT_SYSTEM + "\n\nCurrent context:\n" + context},
            *history,
        ]
        if not history or history[-1].get("content") != user_text:
            messages.append({"role": "user", "content": user_text})

        reply, source, error = await self._stream_reply(messages)
        if not reply:
            reply = (
                "抱歉，暂时没拿到大模型的回答，请稍后重试或检查右上角设置。"
                + (f"\n(LLM error: {error})" if error else "")
            )
            source = "error"
            # Even on failure, ensure the user sees the bubble.
            await self.say(reply)

        await self.emit(
            "agent_finished",
            result={"reply": reply, "source": source},
        )
        return {"reply": reply, "source": source}

    async def _stream_reply(
        self,
        messages: List[Dict[str, str]],
    ) -> tuple[str, str, Optional[str]]:
        """Stream the chat completion and surface it to the UI live.

        Returns ``(reply_text, source, error)``. ``source`` is one of
        ``"stream"`` or ``"non_stream_fallback"`` on success, ``"error"``
        on failure. ``error`` is set only on failure.
        """
        message_id = uuid.uuid4().hex
        await self.emit(
            "chat_message_started",
            id=message_id,
            role="assistant",
            agent=self.agent_id,
        )
        accumulated = ""
        last_flush = time.time()
        pending = ""
        try:
            stream = llm_client.chat_stream(
                messages=messages,
                temperature=0.7,
                timeout=float(settings.llm_chat_timeout),
                total_timeout=max(60.0, float(settings.llm_chat_timeout) * 4.0),
                max_tokens=700,
                purpose="chat",
            )
            # Coalesce tiny chunks: flush either every ~120ms or every
            # ~40 chars so we don't drown the WebSocket in single-token
            # events. The user still sees fluid streaming.
            async for delta in stream:
                accumulated += delta
                pending += delta
                now = time.time()
                if len(pending) >= 40 or (now - last_flush) >= 0.12:
                    await self.emit(
                        "chat_delta",
                        id=message_id,
                        delta=pending,
                        text=accumulated,
                    )
                    pending = ""
                    last_flush = now
            if pending:
                await self.emit(
                    "chat_delta",
                    id=message_id,
                    delta=pending,
                    text=accumulated,
                )
            if not accumulated.strip():
                await self.emit(
                    "chat_message_ended",
                    id=message_id,
                    aborted=True,
                )
                return "", "error", "empty stream"
            await self.say(accumulated, message_id=message_id)
            await self.emit("chat_message_ended", id=message_id, aborted=False)
            return accumulated, "stream", None
        except TimeoutError as exc:
            logger.warning("ChatAgent stream timed out: %s", exc)
            error_text = f"timeout: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChatAgent stream failed: %s", exc)
            error_text = str(exc)
        # If the model already streamed useful content, keep it. The old
        # behaviour aborted the placeholder even with 100+ chars received,
        # so the timeline showed deltas but the message was not persisted.
        if pending:
            await self.emit(
                "chat_delta",
                id=message_id,
                delta=pending,
                text=accumulated,
            )
        if accumulated.strip():
            await self.say(accumulated, message_id=message_id)
            await self.emit(
                "chat_message_ended",
                id=message_id,
                aborted=False,
                partial=True,
                error=error_text,
            )
            return accumulated, "stream_partial", error_text

        # No useful text arrived; mark the placeholder as aborted so the UI
        # can clean it up, then fall through to the caller's error path.
        await self.emit("chat_message_ended", id=message_id, aborted=True)
        return "", "error", error_text

    def _quick_reply(self, user_text: str) -> str | None:
        """Handle TRULY trivial chat locally so basic greetings feel
        instantaneous. Everything else (including identity and capability
        questions) should go through the LLM so the reply feels real.
        """
        text = user_text.strip()
        if _EN_GREETING_RE.match(text):
            return "Hello! I'm here. Tell me what you'd like the system to do next."
        if _ZH_GREETING_RE.match(text):
            return "你好，我在。告诉我接下来想做什么，我会帮你调度对应的智能体。"
        return None

    def _recent_history(self) -> List[Dict[str, str]]:
        """Replay the last few user/assistant turns to the LLM so a
        chit-chat thread can stay coherent (e.g. user asks a follow-up
        like "and that one?" — we need the previous topic in scope).
        """
        out: List[Dict[str, str]] = []
        for msg in self.session.messages[-_CHAT_HISTORY_MAX:]:
            role = msg.get("role")
            content = msg.get("text")
            if role in ("user", "assistant") and content:
                out.append({"role": role, "content": content})
        return out

    def _build_context_summary(self) -> str:
        recent_artifacts = [a.get("name") for a in self.session.artifacts[-3:]]
        return json.dumps(
            {
                "pipeline_state": self.session.pipeline_state,
                "active_dataset": self.session.active_dataset_id,
                "mode": self.session.mode,
                "selected_cluster": self.session.selected_cluster,
                "selected_clusters": self.session.selected_clusters,
                "recent_artifacts": recent_artifacts,
            },
            ensure_ascii=False,
        )
