"""End-to-end smoke test: POST message → Router → Chat → WS event stream.

Usage:
    python -m app.backend.tests.smoke_router

Assumes a uvicorn instance is running on http://127.0.0.1:8765.
"""
from __future__ import annotations

import asyncio
import json
import sys
import urllib.request

import websockets

BASE = "http://127.0.0.1:8765"
WS_BASE = "ws://127.0.0.1:8765"


def _post(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


async def listen(session_id: str, target_kinds: set[str], stop_after_seconds: float) -> list[dict]:
    events: list[dict] = []
    async with websockets.connect(f"{WS_BASE}/ws/session/{session_id}") as ws:
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=stop_after_seconds)
                msg = json.loads(raw)
                events.append(msg)
                if msg.get("kind") == "agent_finished" and msg.get("agent") in {"chat", "router"}:
                    # stop after both router_decision and chat finish observed
                    kinds_seen = {e.get("kind") for e in events}
                    if "router_decision" in kinds_seen and any(
                        e.get("kind") == "agent_finished" and e.get("agent") == "chat" for e in events
                    ):
                        break
        except asyncio.TimeoutError:
            pass
    return events


async def main(user_text: str = "Tell me how this system works.") -> int:
    session = _post("/api/session")
    sid = session["session_id"]
    print(f"[*] session: {sid}")

    listener = asyncio.create_task(listen(sid, {"router_decision", "agent_finished"}, 30))
    await asyncio.sleep(0.5)
    print("[*] posting message…")
    _post(f"/api/session/{sid}/message", {"text": user_text})

    events = await listener
    print(f"[*] received {len(events)} events")
    for ev in events:
        print(" -", ev.get("kind"), ev.get("agent"), {k: v for k, v in ev.items() if k in {"text", "intent", "next_agent", "reason"}})

    kinds = {e.get("kind") for e in events}
    if "router_decision" not in kinds:
        print("[!] no router_decision observed", file=sys.stderr)
        return 1
    print("[OK] router decision observed")
    return 0


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "Tell me how this system works."
    sys.exit(asyncio.run(main(text)))
