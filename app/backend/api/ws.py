"""WebSocket endpoints for live agent-event and upload-progress streams."""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.backend.orchestrator.events import bus
from app.backend.orchestrator.session_store import store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/session/{session_id}")
async def ws_session(ws: WebSocket, session_id: str) -> None:
    await ws.accept()
    session = await store.get(session_id)
    if session is None:
        await ws.send_text(json.dumps({"kind": "error", "text": "session not found"}))
        await ws.close()
        return
    queue = await bus.subscribe(session_id)
    try:
        await ws.send_text(json.dumps({"kind": "hello", "session_id": session_id, "state": session.to_dict()}))
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                await ws.send_text(json.dumps(msg, ensure_ascii=False))
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"kind": "ping", "t": 0}))
    except WebSocketDisconnect:
        logger.info("client disconnected from session %s", session_id)
    except Exception as exc:
        logger.exception("ws session error: %s", exc)
    finally:
        await bus.unsubscribe(session_id, queue)
