"""End-to-end test: ComputeAgent runs on builtin-first-200s using a known good gpt.py."""
from __future__ import annotations

import asyncio
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.agents.compute_agent import ComputeAgent
from app.backend.orchestrator.events import bus
from app.backend.orchestrator.session_store import Session
from app.backend.settings import DIANCHI_MMD_TOOLS_NEW, EVO_GPT_PATH
from app.backend.tests.test_mmd_designer import GOOD_CODE


GOOD_CODE_RAW = re.sub(r"^```python\n|```$", "", GOOD_CODE.strip())


async def collect_until(session_id: str, kind: str, agent: str, timeout: float = 60.0):
    queue = await bus.subscribe(session_id)
    events = []
    try:
        while True:
            ev = await asyncio.wait_for(queue.get(), timeout=timeout)
            events.append(ev)
            if ev.get("kind") == kind and ev.get("agent") == agent:
                break
    finally:
        await bus.unsubscribe(session_id, queue)
    return events


async def run_one_shot() -> int:
    saved_gpt = EVO_GPT_PATH.read_text(encoding="utf-8") if EVO_GPT_PATH.exists() else None
    saved_tools = (
        DIANCHI_MMD_TOOLS_NEW.read_text(encoding="utf-8") if DIANCHI_MMD_TOOLS_NEW.exists() else None
    )
    EVO_GPT_PATH.write_text(GOOD_CODE_RAW, encoding="utf-8")

    session = Session(session_id="test_compute_001")
    agent = ComputeAgent(session)

    listener = asyncio.create_task(
        collect_until(session.session_id, "agent_finished", "compute", timeout=60.0)
    )
    await asyncio.sleep(0.1)
    try:
        result = await agent.run()
        events = await listener
    finally:
        if saved_gpt is not None:
            EVO_GPT_PATH.write_text(saved_gpt, encoding="utf-8")
        if saved_tools is not None:
            DIANCHI_MMD_TOOLS_NEW.write_text(saved_tools, encoding="utf-8")
        bak = DIANCHI_MMD_TOOLS_NEW.with_suffix(DIANCHI_MMD_TOOLS_NEW.suffix + ".bak")
        if bak.exists():
            bak.unlink()

    if not result.get("passed"):
        print(f"[FAIL] result not passed: {result}")
        return 1

    kinds = [e.get("kind") for e in events]
    print(f"  events: {len(events)}; counts: " + ", ".join(
        f"{k}={kinds.count(k)}" for k in sorted(set(kinds))
    ))

    required_event_kinds = {"agent_started", "thinking", "tool_call", "chart_data", "artifact", "agent_finished"}
    missing = required_event_kinds - set(kinds)
    if missing:
        print(f"  [FAIL] missing kinds: {missing}")
        return 1

    progress_charts = [e for e in events if e.get("kind") == "chart_data" and e.get("chart") == "mmd_progress"]
    final_chart = [e for e in events if e.get("kind") == "chart_data" and e.get("chart") == "mmd_hist"]
    corr_chart = [e for e in events if e.get("kind") == "chart_data" and e.get("chart") == "mmd_correlation"]
    if not progress_charts:
        print("  [FAIL] no mmd_progress live chart events")
        return 1
    if not final_chart:
        print("  [FAIL] no final mmd_hist chart event")
        return 1
    if not corr_chart:
        print("  [FAIL] no mmd_correlation chart event")
        return 1

    hist = final_chart[-1]["data"]["histogram"]
    print(f"  [OK] histogram stats: n={hist['n']}, "
          f"min={hist['min']:.4f}, max={hist['max']:.4f}, mean={hist['mean']:.4f}")
    corr = corr_chart[-1]["data"]
    print(f"  [OK] correlation: matched={corr['matched']}, mean_sig={corr['mean_sig']:.3f}")
    print(f"  [OK] {len(progress_charts)} live progress ticks emitted")

    # session updated
    if session.pipeline_state.get("compute") != "done":
        print(f"  [FAIL] pipeline_state.compute = {session.pipeline_state.get('compute')}")
        return 1
    if not any(a["name"] == "mmd_results.csv" for a in session.artifacts):
        print("  [FAIL] mmd_results.csv artifact missing")
        return 1
    csv_path = Path([a for a in session.artifacts if a["name"] == "mmd_results.csv"][0]["path"])
    if not csv_path.exists():
        print(f"  [FAIL] csv not on disk: {csv_path}")
        return 1
    print(f"  [OK] csv written: {csv_path} ({csv_path.stat().st_size} bytes)")
    return 0


def main() -> int:
    print("== test_compute_agent ==")
    rc = asyncio.run(run_one_shot())
    print("PASSED" if rc == 0 else "FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
