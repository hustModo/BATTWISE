"""Running conversation summary for BATTWISE.

Maintains a compact JSON summary of "what the user wants and what has been
done so far", refreshed periodically with a small LLM call. The summary is:

* fed into the smart router so it has cross-turn memory;
* surfaced on the WebSocket as ``conversation_summary`` so the UI can show
  the user what the assistant currently remembers.

We deliberately do NOT call the LLM on every message — that would double
the average per-message latency. Instead refresh is opportunistic:

* After each non-router agent finishes (Designer / Compute / Cluster /
  Refine), pipeline.handle_message schedules ``refresh_in_background``.
* Throttled by ``SUMMARY_MIN_INTERVAL_S`` so a burst of agent_finished
  events doesn't melt the model.
* Manual override via ``force=True`` for tests.

Failures are non-fatal — if the LLM is unreachable we keep the last good
summary and routing degrades to "no summary context".
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from app.backend.orchestrator.events import AgentEvent, bus, safe_json
from app.backend.orchestrator.session_store import Session, store
from app.backend.services import llm_client
from app.backend.settings import settings

logger = logging.getLogger(__name__)


SUMMARY_MODEL = "qwen3.5-flash"
SUMMARY_HISTORY_TURNS = 16
SUMMARY_MIN_INTERVAL_S = 8.0  # throttle: at most one refresh every 8s
SUMMARY_TIMEOUT_S = 12.0  # hard wall-clock budget per refresh

# Sessions with an in-flight refresh task. Guards against firing a
# second refresh before the first one finishes — without this, a burst
# of agent_finished events can spawn N concurrent LLM calls.
_IN_FLIGHT: Dict[str, asyncio.Task] = {}


_SUMMARY_SYSTEM = """You maintain a running summary of an ongoing conversation
between a user and the BATTWISE battery-sorting assistant. The summary
must stay short (≤ 220 words total across all fields) and capture only
durable facts that will help the router decide what to do next.

Output STRICT JSON (no markdown fences, no prose):

{
  "what_user_wants": "<1-2 sentences>",
  "what_was_done": ["<short bullet of completed step>", ...],
  "current_state": "<1 sentence — what is the user mid-flow on, if anything>",
  "user_preferences": {"<key>": "<value>", ...},
  "open_questions": ["<question the user is awaiting an answer on>", ...]
}

RULES:
- Use the user's language (mirror the dominant language of recent turns).
- Drop greetings, jokes, off-topic chat — they are not durable.
- For preferences prefer concrete values, e.g.
feedback_metric=variance, k_range=3..6, mode=reflective_search,
    interested_clusters=[0,3,5].
- "what_was_done" stays under 6 items, newest first.
- If a field has no information, return "" or [].
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.M)
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _truncate(s: Any, n: int = 240) -> str:
    text = str(s or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _build_messages_block(session: Session) -> str:
    lines: List[str] = ["RECENT TURNS:"]
    for msg in session.messages[-SUMMARY_HISTORY_TURNS:]:
        role = msg.get("role", "?")
        agent = msg.get("agent")
        tag = f"{role}/{agent}" if agent else role
        lines.append(f"  [{tag}] {_truncate(msg.get('text'), 320)}")
    arts = [a.get("name") for a in session.artifacts[-5:] if a.get("name")]
    if arts:
        lines.append(f"\nRECENT ARTIFACTS: {arts}")
    return "\n".join(lines)


def _parse_summary(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = _FENCE_RE.sub("", text.strip())
    match = _JSON_BLOCK_RE.search(cleaned)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    prefs_raw = data.get("user_preferences") or {}
    prefs: Dict[str, Any] = {}
    if isinstance(prefs_raw, dict):
        for k, v in prefs_raw.items():
            key = _truncate(k, 40)
            if isinstance(v, (str, int, float, bool)) or v is None:
                prefs[key] = v
            elif isinstance(v, (list, tuple)):
                prefs[key] = [str(x)[:60] for x in v][:8]
            else:
                prefs[key] = _truncate(v, 80)
    return {
        "what_user_wants": _truncate(data.get("what_user_wants"), 400),
        "what_was_done": [_truncate(x, 160) for x in (data.get("what_was_done") or [])[:6]],
        "current_state": _truncate(data.get("current_state"), 240),
        "user_preferences": prefs,
        "open_questions": [_truncate(x, 160) for x in (data.get("open_questions") or [])[:4]],
    }


def get_summary(session: Session) -> Optional[Dict[str, Any]]:
    """Return the last computed summary (dict) or None if never built."""
    summary = session.context.get("conversation_summary")
    return summary if isinstance(summary, dict) else None


def _should_refresh(session: Session, *, force: bool) -> bool:
    if force:
        return True
    if len(session.messages) < 2:
        return False
    last = session.context.get("conversation_summary_t", 0.0)
    try:
        last_f = float(last)
    except (TypeError, ValueError):
        last_f = 0.0
    return (time.time() - last_f) >= SUMMARY_MIN_INTERVAL_S


async def refresh(session: Session, *, force: bool = False) -> Optional[Dict[str, Any]]:
    """Recompute the running summary (blocking on the LLM call).

    Returns the new summary dict, or the previous summary on failure.
    Throttled by ``SUMMARY_MIN_INTERVAL_S`` unless ``force=True``.
    """
    if not _should_refresh(session, force=force):
        return get_summary(session)

    snapshot = {
        "pipeline_state": session.pipeline_state,
        "mode": session.mode,
        "active_dataset": session.active_dataset_id,
        "selected_clusters": session.selected_clusters,
        "sticky_eoh_params": session.context.get("eoh_user_params"),
        "sticky_cluster_params": session.context.get("cluster_user_params"),
    }
    user_block = (
        f"SESSION SNAPSHOT:\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}\n\n"
        f"{_build_messages_block(session)}"
    )
    try:
        raw = await llm_client.chat_async(
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user", "content": user_block},
            ],
            temperature=0.0,
            model=SUMMARY_MODEL,
            max_retries=1,
            timeout=SUMMARY_TIMEOUT_S,
            max_tokens=400,
            purpose="summary",
        )
    except Exception as exc:
        logger.warning("conversation_memory.refresh LLM call failed: %s", exc)
        return get_summary(session)

    parsed = _parse_summary(raw)
    if not parsed:
        logger.warning(
            "conversation_memory got unparseable response (first 240 chars): %r",
            (raw or "")[:240],
        )
        return get_summary(session)

    session.context["conversation_summary"] = parsed
    session.context["conversation_summary_t"] = time.time()
    await store.update(session)
    await bus.publish(
        AgentEvent(
            kind="conversation_summary",
            session_id=session.session_id,
            agent="router",
            payload={"summary": safe_json(parsed)},
        )
    )
    return parsed


def refresh_in_background(session: Session) -> Optional[asyncio.Task]:
    """Fire-and-forget refresh. Returns the scheduled Task (mostly for tests).

    Deduplicates: if a refresh task is already running for this session we
    return it as-is instead of spawning a second one. Without this, two
    user turns within a few seconds spawn 2 concurrent 100s+ LLM calls.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("refresh_in_background: no running loop, skipping")
        return None

    existing = _IN_FLIGHT.get(session.session_id)
    if existing is not None and not existing.done():
        logger.debug(
            "refresh_in_background: refresh already in flight for session %s",
            session.session_id,
        )
        return existing

    task = loop.create_task(refresh(session))
    _IN_FLIGHT[session.session_id] = task
    task.add_done_callback(
        lambda _t, sid=session.session_id: _IN_FLIGHT.pop(sid, None)
    )
    return task
