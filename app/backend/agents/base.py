"""AgentBase: shared event-emission and state plumbing for all agents."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, Optional

from app.backend.orchestrator.events import AgentEvent, bus, safe_json
from app.backend.orchestrator.session_store import Session, store

logger = logging.getLogger(__name__)


class AgentBase:
    """Subclasses set ``agent_id`` and implement ``run``."""

    agent_id: str = "base"
    node_id: Optional[str] = None  # one of pipeline_state keys, if applicable

    def __init__(self, session: Session) -> None:
        self.session = session

    async def emit(self, kind: str, /, **payload: Any) -> None:
        # `kind` is positional-only so payload may carry its own 'kind' key
        # (e.g. an artifact's MIME-like type).
        event = AgentEvent(
            kind=kind,
            session_id=self.session.session_id,
            agent=self.agent_id,
            payload={k: safe_json(v) for k, v in payload.items()},
        )
        await bus.publish(event)

    async def log(self, text: str, level: str = "info") -> None:
        await self.emit("log", level=level, text=text)
        getattr(logger, level if level in {"info", "warning", "error", "debug"} else "info")(
            "[%s] %s", self.agent_id, text
        )

    async def started(self) -> None:
        if self.node_id:
            self.session.set_node(self.node_id, "running")
            self.session.active_agent = self.agent_id
            await store.update(self.session)
        await self.emit("agent_started", node_id=self.node_id)

    async def finished(self, result: Optional[Dict[str, Any]] = None, status: str = "done") -> None:
        if self.node_id:
            self.session.set_node(self.node_id, status)
            self.session.active_agent = None
            await store.update(self.session)
        await self.emit("agent_finished", result=result or {}, status=status)

    async def failed(self, error: str) -> None:
        if self.node_id:
            self.session.set_node(self.node_id, "error")
            self.session.active_agent = None
            await store.update(self.session)
        await self.emit("agent_finished", error=error, status="error")

    async def thinking(self, text: str) -> None:
        await self.emit("thinking", text=text)

    async def say(
        self,
        text: str,
        *,
        role: str = "assistant",
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist a chat message and push it to the live UI.

        Agents used to call ``session.add_message`` directly, which updated
        the JSON state but did not notify the WebSocket client. The user only
        saw the message after a refresh/reconnect. Keep message persistence
        and the real-time event coupled here so every agent summary is visible.
        """
        agent = self.agent_id if role == "assistant" else None
        msg = self.session.add_message(
            role=role, text=text, agent=agent, message_id=message_id,
        )
        await store.update(self.session)
        await self.emit("message", message=msg)
        return msg

    async def tool_call(
        self,
        tool: str,
        args: Dict[str, Any],
        *,
        status: str = "running",
        duration_ms: Optional[float] = None,
        result: Any = None,
        error: Optional[str] = None,
    ) -> str:
        call_id = uuid.uuid4().hex
        await self.emit(
            "tool_call",
            tool=tool,
            args=args,
            status=status,
            duration_ms=duration_ms,
            result=result,
            error=error,
            call_id=call_id,
        )
        return call_id

    async def tool_done(
        self,
        call_id: str,
        tool: str,
        *,
        duration_ms: float,
        result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        status = "error" if error else "ok"
        await self.emit(
            "tool_call",
            tool=tool,
            status=status,
            duration_ms=duration_ms,
            result=result,
            error=error,
            call_id=call_id,
            update=True,
        )

    async def artifact(self, name: str, kind: str, path: str, **preview: Any) -> None:
        # Frontend uses `mime` (per PLAN.md contract) so it doesn't collide with the
        # outer event envelope's `kind`. Local session.artifacts keeps the `kind` field.
        await self.emit("artifact", name=name, mime=kind, path=path, preview=preview or None)

    async def run(self, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError


def now_ms() -> float:
    return time.time() * 1000.0
