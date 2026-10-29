"""Regression test: RefinementAgent runs a single All-200 sub-pipeline on the
union of cells from multiple parent clusters.

User scenario (verbatim):
    "我要对感兴趣，例如 0,3,5 类别进行 all_200 实验区分"

The flow:
    1. Run Designer / Compute / Cluster (k=6) to seed `Cluster_label.xlsx`.
    2. Pick three parent clusters of decent size.
    3. First Refine call (without `confirm_all200`) must block and emit a
       `user_prompt_required` event whose payload now contains both the
       multi-cluster `clusters` list and a per-cluster `cells_by_cluster`
       breakdown.
    4. After `confirm_all200`, second call must run sub-MMD on the UNION
       of cells and write a `Sub_Clusters_<a>-<b>-<c>_label.xlsx`
       artifact (NOT the single-cluster name).
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
        session = Session(session_id="test_refine_multi_001")

        print("[1/4] Designer + Compute + Cluster (k=4 to keep test fast)")
        await DesignerAgent(session).run(user_text="design a new MMD")
        await ComputeAgent(session).run()
        # target_k=4 because the built-in dataset is small; the test only
        # needs ≥3 clusters and ≥4 total cells across the selected ones.
        cl = await ClusterAgent(session).run(target_k=4)
        assert cl.get("passed"), f"cluster failed: {cl}"
        per = session.context.get("last_cluster_per_cluster", [])
        sizes = {p["cluster"]: p["size"] for p in per}
        print(f"  -> k={cl['k']}, cluster sizes={sizes}")

        # Pick the THREE largest clusters as our multi-selection target.
        ranked = sorted(per, key=lambda s: -s["size"])
        picks = [int(p["cluster"]) for p in ranked[:3]]
        assert len(picks) >= 2, f"need at least 2 clusters to test multi, got {picks}"
        # Sort for deterministic file naming downstream.
        picks = sorted(picks)
        total_cells_in_picks = sum(sizes[c] for c in picks)
        print(f"     multi-cluster target: {picks} (n={total_cells_in_picks})")

        print("[2/4] Refinement (multi) — first call should BLOCK")
        listener = asyncio.create_task(
            collect_until(session.session_id, "refine", timeout=15.0)
        )
        await asyncio.sleep(0.05)
        r1 = await RefinementAgent(session).run(cluster_ids=picks)
        events1 = await listener
        assert r1.get("awaiting") == "confirm_all200", f"expected awaiting, got {r1}"

        prompt_evs = [e for e in events1 if e.get("kind") == "user_prompt_required"]
        assert prompt_evs, "expected a user_prompt_required event"
        prompt = prompt_evs[0]
        assert prompt.get("clusters") == picks, (
            f"expected clusters={picks} in event, got {prompt.get('clusters')}"
        )
        # cells_by_cluster keys are strings (JSON-safe) — verify shape.
        cbc = prompt.get("cells_by_cluster", {})
        assert set(cbc.keys()) == {str(c) for c in picks}, (
            f"cells_by_cluster keys mismatch: {cbc.keys()}"
        )
        union_size = len({c for v in cbc.values() for c in v})
        assert union_size == total_cells_in_picks, (
            f"union size {union_size} != expected {total_cells_in_picks}"
        )
        print(f"  -> user_prompt_required carried clusters={prompt['clusters']}, "
              f"cells_by_cluster sizes={[len(v) for v in cbc.values()]}")

        # Confirm all-200, second call should run a SINGLE sub-pipeline on
        # the union of selected cells.
        session.context["all200_ready"] = True
        session.context["refine_dataset_id"] = "builtin-all-200"

        print("[3/4] Refinement (multi) — second call after confirm should produce ONE sub-pipeline")
        history_before = len(bus.history(session.session_id))
        r2 = await RefinementAgent(session).run(cluster_ids=picks)
        assert r2.get("passed"), f"refinement failed: {r2}"
        assert r2.get("parent_clusters") == picks, (
            f"expected parent_clusters={picks}, got {r2.get('parent_clusters')}"
        )
        print(f"  -> parent_clusters={r2['parent_clusters']}, "
              f"sub k={r2['k']}, sub sizes={r2['sizes']}")

        full_history = bus.history(session.session_id)
        events2 = full_history[history_before:]
        refine_events = [e for e in events2 if e.get("agent") == "refine"]

        # Check that every sub_* chart carries the new `parent_clusters` list
        # (backwards-compat single field is kept too).
        sub_charts = [e for e in refine_events if e.get("kind") == "chart_data"]
        seen_charts = {e.get("chart") for e in sub_charts}
        required = {
            "sub_mmd_progress",
            "sub_cluster_iter",
            "sub_cluster_scatter",
            "sub_capacity_box",
            "sub_fingerprint",
            "sub_cluster_summary",
        }
        missing = required - seen_charts
        assert not missing, f"missing sub-charts: {missing}, got {seen_charts}"

        sample = next(
            e for e in sub_charts if e.get("chart") == "sub_cluster_summary"
        )
        # Per-list payload check:
        sample_data = sample.get("data") or sample
        assert sample_data.get("parent_clusters") == picks, (
            f"sub_cluster_summary missing/wrong parent_clusters: {sample_data}"
        )

        print(f"     refine events captured: {len(refine_events)} / new {len(events2)}")

        # ---- artefact filename test ----
        names = [a["name"] for a in session.artifacts]
        expected_name = "Sub_Clusters_" + "-".join(str(p) for p in picks) + "_label.xlsx"
        assert expected_name in names, (
            f"missing multi-cluster artefact '{expected_name}'; got {names}"
        )
        print(f"     artefact: {expected_name} ✓")

        print("\n[OK] Multi-cluster Refinement E2E passed.")
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
    print("== test_refinement_multi_cluster ==")
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
