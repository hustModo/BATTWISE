"""Full backend pipeline smoke: Designer → Compute → Cluster.

Designer uses a stubbed LLM (no network) so the whole chain stays under 30s.
"""
from __future__ import annotations

import asyncio
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.agents.cluster_agent import ClusterAgent
from app.backend.agents.compute_agent import ComputeAgent
from app.backend.agents.designer_agent import DesignerAgent
from app.backend.orchestrator.events import bus
from app.backend.orchestrator.session_store import Session
from app.backend.services import mmd_designer
from app.backend.settings import DIANCHI_MMD_TOOLS_NEW, EVO_GPT_PATH
from app.backend.tests.test_mmd_designer import GOOD_CODE, make_scripted_chat


async def collect_until(session_id: str, agent: str, kind: str = "agent_finished", timeout: float = 240.0):
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


async def run() -> int:
    saved_gpt = EVO_GPT_PATH.read_text(encoding="utf-8") if EVO_GPT_PATH.exists() else None
    saved_tools = (
        DIANCHI_MMD_TOOLS_NEW.read_text(encoding="utf-8") if DIANCHI_MMD_TOOLS_NEW.exists() else None
    )

    original_chat = mmd_designer.llm_client.chat
    mmd_designer.llm_client.chat = make_scripted_chat([GOOD_CODE])

    try:
        session = Session(session_id="test_e2e_001")

        # --- Designer ---
        print("[1/3] Designer")
        designer = DesignerAgent(session)
        listener = asyncio.create_task(collect_until(session.session_id, "designer", timeout=30.0))
        await asyncio.sleep(0.05)
        d_result = await designer.run(user_text="design a new MMD")
        await listener
        if not d_result.get("passed"):
            print("  [FAIL] designer failed")
            return 1
        print(f"  -> designed {d_result['function_name']} in {d_result['attempts']} attempt(s)")

        # --- Compute ---
        print("[2/3] Compute")
        compute = ComputeAgent(session)
        listener = asyncio.create_task(collect_until(session.session_id, "compute", timeout=60.0))
        await asyncio.sleep(0.05)
        c_result = await compute.run()
        await listener
        if not c_result.get("passed"):
            print("  [FAIL] compute failed")
            return 1
        print(f"  -> computed {c_result['computed']} cells; csv={Path(c_result['csv']).name}")

        # --- Cluster ---
        print("[3/3] Cluster")
        cluster = ClusterAgent(session)
        listener = asyncio.create_task(collect_until(session.session_id, "cluster", timeout=180.0))
        await asyncio.sleep(0.05)
        cl_result = await cluster.run()
        await listener
        if not cl_result.get("passed"):
            print("  [FAIL] cluster failed")
            return 1
        print(f"  -> k={cl_result['k']}, cluster_xlsx={Path(cl_result['cluster_xlsx']).name}")

        # --- Sanity ---
        names = [a["name"] for a in session.artifacts]
        expected = {"gpt.py", "mmd_results.csv", "Cluster_label.xlsx"}
        missing = expected - set(names)
        if missing:
            print(f"[FAIL] missing artifacts: {missing} (got {names})")
            return 1
        for state, expected_state in [
            ("designer", "done"),
            ("compute", "done"),
            ("cluster", "done"),
        ]:
            if session.pipeline_state.get(state) != expected_state:
                print(f"[FAIL] pipeline_state.{state} = {session.pipeline_state.get(state)!r}")
                return 1
        print("\n[OK] Full pipeline succeeded.")
        print(f"     artifacts: {names}")
        print(f"     pipeline_state: {session.pipeline_state}")
        return 0
    finally:
        mmd_designer.llm_client.chat = original_chat
        if saved_gpt is not None:
            EVO_GPT_PATH.write_text(saved_gpt, encoding="utf-8")
        if saved_tools is not None:
            DIANCHI_MMD_TOOLS_NEW.write_text(saved_tools, encoding="utf-8")
        bak = DIANCHI_MMD_TOOLS_NEW.with_suffix(DIANCHI_MMD_TOOLS_NEW.suffix + ".bak")
        if bak.exists():
            bak.unlink()
        evo_bak = EVO_GPT_PATH.with_suffix(EVO_GPT_PATH.suffix + ".bak")
        if evo_bak.exists():
            evo_bak.unlink()


def main() -> int:
    print("== test_pipeline_end_to_end ==")
    rc = asyncio.run(run())
    return rc


if __name__ == "__main__":
    sys.exit(main())
