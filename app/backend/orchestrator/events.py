"""Unified agent event model with WebSocket broadcasting."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


AgentId = str  # router | designer | compute | cluster | refine | chat


@dataclass
class AgentEvent:
    """Single event emitted into the WebSocket stream.

    Mirrors the contract documented in docs/PLAN.md §五:
        agent_started / thinking / tool_call / artifact / chart_data /
        code_update / agent_finished / router_decision /
        user_prompt_required / log
    """

    kind: str
    session_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    agent: Optional[AgentId] = None
    t: float = field(default_factory=lambda: time.time())
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        flat: Dict[str, Any] = {}
        for k, v in d["payload"].items():
            # never let payload override the reserved envelope fields
            if k in {"kind", "t", "event_id"}:
                continue
            flat[k] = v
        flat["kind"] = d["kind"]
        flat["t"] = d["t"]
        flat["event_id"] = d["event_id"]
        if d["agent"] is not None:
            flat["agent"] = d["agent"]
        return flat


class EventBus:
    """In-process pub/sub keyed by session_id."""

    def __init__(self) -> None:
        self._queues: Dict[str, List[asyncio.Queue]] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, session_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        async with self._lock:
            self._queues.setdefault(session_id, []).append(queue)
            for hist in self._history.get(session_id, [])[-50:]:
                queue.put_nowait(hist)
        return queue

    async def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            queues = self._queues.get(session_id, [])
            if queue in queues:
                queues.remove(queue)
            if not queues and session_id in self._queues:
                del self._queues[session_id]

    async def publish(self, event: AgentEvent) -> None:
        message = event.to_dict()
        async with self._lock:
            self._history.setdefault(event.session_id, []).append(message)
            if len(self._history[event.session_id]) > 1000:
                self._history[event.session_id] = self._history[event.session_id][-1000:]
            targets = list(self._queues.get(event.session_id, []))
        for queue in targets:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning("Event queue full for session %s, dropping event", event.session_id)

    def history(self, session_id: str) -> List[Dict[str, Any]]:
        return list(self._history.get(session_id, []))


bus = EventBus()


def safe_json(obj: Any) -> Any:
    """Return JSON-serializable representation, stringifying unknowns."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if isinstance(obj, dict):
            return {k: safe_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [safe_json(v) for v in obj]
        return repr(obj)
