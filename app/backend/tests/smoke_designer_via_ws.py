"""Full HTTP+WS smoke test: post 'design a new mmd' → observe DesignerAgent events.

Requires uvicorn running on http://127.0.0.1:8765.
Uses a stubbed LLM (monkeypatch via env var) so we don't burn the live API.
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
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


async def listen_until_designer_finishes(session_id: str, timeout: float = 300.0):
    events = []
    async with websockets.connect(f"{WS_BASE}/ws/session/{session_id}") as ws:
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                msg = json.loads(raw)
                events.append(msg)
                kind = msg.get("kind")
                agent = msg.get("agent")
                if kind in {"thinking", "code_update", "tool_call"} and agent == "designer":
                    snippet = ""
                    if kind == "thinking":
                        snippet = (msg.get("text") or "")[:140]
                    elif kind == "code_update":
                        snippet = f"nan_check.passed={(msg.get('nan_check') or {}).get('passed')} final={msg.get('final')}"
                    elif kind == "tool_call":
                        snippet = f"tool={msg.get('tool')} status={msg.get('status')}"
                    print(f"  [{kind}] {snippet}")
                if kind == "agent_finished" and agent == "designer":
                    print(f"  [agent_finished] designer result={msg.get('result')}")
                    break
        except asyncio.TimeoutError:
            print(f"  [TIMEOUT] after {timeout}s, last kind={(events[-1] or {}).get('kind') if events else None}")
    return events


async def main() -> int:
    session = _post("/api/session")
    sid = session["session_id"]
    print(f"[*] session: {sid}")

    listener = asyncio.create_task(listen_until_designer_finishes(sid, timeout=300.0))
    await asyncio.sleep(0.4)
    _post(f"/api/session/{sid}/message", {"text": "design a new MMD function please"})
    events = await listener

    finished = [
        e for e in events
        if e.get("kind") == "agent_finished" and e.get("agent") == "designer"
    ]
    if not finished:
        print("[!] designer did not finish")
        return 1
    result = finished[-1].get("result") or {}
    print(f"[OK] designer finished — passed={result.get('passed')} attempts={result.get('attempts')}")
    print(f"     function_name={result.get('function_name')}")
    print(f"     sample_losses={result.get('sample_losses')}")
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
