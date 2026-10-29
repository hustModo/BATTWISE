"""End-to-end refinement test: Designer (stub LLM) → Compute → Cluster → Refine.

Verifies that RefinementAgent:
    * blocks with `user_prompt_required` when all200 not yet confirmed
    * proceeds to sub-MMD + sub-cluster once `confirm_all200` is set, using the
      built-in `builtin-all-200` (AII_200_new) dataset
    * emits the expected `chart_data` sub-events and writes a sub-Cluster artefact
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.agents.cluster_agent import ClusterAgent  # noqa: E402
from app.backend.agents.compute_agent import ComputeAgent  # noqa: E402
from app.backend.agents.designer_agent import DesignerAgent  # noqa: E402
from app.backend.agents.refinement_agent import RefinementAgent  # noqa: E402
from app.backend.orchestrator.events import bus  # noqa: E402
from app.backend.orchestrator.session_store import Session  # noqa: E402
from app.backend.services import mmd_designer  # noqa: E402
from app.backend.settings import DIANCHI_MMD_TOOLS_NEW, EVO_GPT_PATH  # noqa: E402
from app.backend.tests.test_mmd_designer import GOOD_CODE, make_scripted_chat  # noqa: E402


async def collect_until(session_id, agent, kind="agent_finished", timeout=240.0):
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
        session = Session(session_id="test_refine_001")

        print("[1/4] Designer (stub LLM)")
        listener = asyncio.create_task(collect_until(session.session_id, "designer", timeout=30.0))
        await asyncio.sleep(0.05)
        d = await DesignerAgent(session).run(user_text="design a new MMD")
        await listener
        assert d.get("passed"), f"designer failed: {d}"
        print(f"  -> {d['function_name']} in {d['attempts']} attempt(s)")

        print("[2/4] Compute (first-200s)")
        listener = asyncio.create_task(collect_until(session.session_id, "compute", timeout=60.0))
        await asyncio.sleep(0.05)
        c = await ComputeAgent(session).run()
        await listener
        assert c.get("passed"), f"compute failed: {c}"
        print(f"  -> computed {c['computed']} cells")

        print("[3/4] Cluster (first-200s)")
        listener = asyncio.create_task(collect_until(session.session_id, "cluster", timeout=180.0))
        await asyncio.sleep(0.05)
        cl = await ClusterAgent(session).run()
        await listener
        assert cl.get("passed"), f"cluster failed: {cl}"
        k = cl["k"]
        sizes = session.context.get("last_cluster_per_cluster", [])
        print(f"  -> k={k}, cluster_xlsx={Path(cl['cluster_xlsx']).name}, sizes={[s['size'] for s in sizes]}")

        # pick largest cluster as target for refinement
        per = sorted(sizes, key=lambda s: -s["size"])
        target_cluster = per[0]["cluster"]
        target_size = per[0]["size"]
        print(f"     refinement target: cluster {target_cluster} (n={target_size})")

        print("[4/4] Refinement — first call should BLOCK awaiting confirm_all200")
        listener = asyncio.create_task(
            collect_until(session.session_id, "refine", timeout=15.0)
        )
        await asyncio.sleep(0.05)
        r1 = await RefinementAgent(session).run(
            user_text=f"I'm interested in Cluster {target_cluster}"
        )
        events1 = await listener
        assert r1.get("awaiting") == "confirm_all200", f"expected awaiting, got {r1}"
        prompt_evs = [e for e in events1 if e.get("kind") == "user_prompt_required"]
        assert prompt_evs, "expected a user_prompt_required event"
        assert prompt_evs[0].get("cluster") == target_cluster
        assert prompt_evs[0].get("suggested_dataset_id") == "builtin-all-200"
        print(f"  -> got user_prompt_required for cluster {target_cluster}; "
              f"suggested={prompt_evs[0]['suggested_dataset_id']}")

        # Confirm all-200, second call should run sub-pipeline
        session.context["all200_ready"] = True
        session.context["refine_dataset_id"] = "builtin-all-200"

        print("       Refinement — second call after confirm should run sub-MMD + sub-cluster")
        history_before = len(bus.history(session.session_id))
        r2 = await RefinementAgent(session).run(
            user_text=f"refine cluster {target_cluster}"
        )
        assert r2.get("passed"), f"refinement failed: {r2}"
        print(f"  -> parent_cluster={r2['parent_cluster']}, k={r2['k']}, sizes={r2['sizes']}")

        # Validate emitted events from this refine call by slicing history.
        full_history = bus.history(session.session_id)
        events2 = full_history[history_before:]
        refine_events = [e for e in events2 if e.get("agent") == "refine"]
        print(f"     refine events captured: {len(refine_events)} / new {len(events2)}")
        required_charts = {
            "sub_mmd_progress",
            "sub_cluster_iter",
            "sub_cluster_scatter",
            "sub_capacity_box",
            "sub_fingerprint",
            "sub_cluster_summary",
        }
        seen_charts = {
            e.get("chart") for e in refine_events if e.get("kind") == "chart_data"
        }
        missing = required_charts - seen_charts
        assert not missing, f"missing sub-charts: {missing}, got {seen_charts}"

        # Verify artefacts
        names = [a["name"] for a in session.artifacts]
        sub_label = f"Sub_Cluster{target_cluster}_label.xlsx"
        assert sub_label in names, f"missing artifact {sub_label}; got {names}"

        # Pipeline state
        assert session.pipeline_state.get("refine") == "done", session.pipeline_state

        print("\n[OK] Refinement E2E passed.")
        print(f"     emitted sub-charts: {sorted(seen_charts)}")
        print(f"     artefacts: {names}")
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
    print("== test_refinement_agent ==")
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
