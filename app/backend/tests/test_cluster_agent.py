"""End-to-end test for ClusterAgent."""
from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.agents.cluster_agent import ClusterAgent
from app.backend.orchestrator.events import bus
from app.backend.orchestrator.session_store import Session
from app.backend.services import data_resolver
from app.backend.settings import settings


async def collect_until(session_id: str, kind: str, agent: str, timeout: float = 120.0):
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


async def run_one_shot(target_k: int | None = None, session_id: str = "test_cluster_001") -> int:
    session = Session(session_id=session_id)
    ds = data_resolver.resolve("builtin-first-200s")

    # seed mmd_results.csv into the session workspace (use built-in copy)
    workspace = settings.session_workspace(session.session_id)
    csv_dst = workspace / "mmd_results.csv"
    shutil.copy2(ds.folder / "mmd_results.csv", csv_dst)
    session.context["last_mmd_csv"] = str(csv_dst)

    agent = ClusterAgent(session)
    listener = asyncio.create_task(
        collect_until(session.session_id, "agent_finished", "cluster", timeout=180.0)
    )
    await asyncio.sleep(0.1)
    # Forward target_k to ClusterAgent.run() so the test can pin a specific K.
    run_kwargs = {"target_k": target_k} if target_k is not None else {}
    result = await agent.run(**run_kwargs)
    events = await listener

    if not result.get("passed"):
        print(f"[FAIL] result not passed: {result}")
        return 1

    kinds = [e.get("kind") for e in events]
    print(f"  events: {len(events)}; counts: " + ", ".join(
        f"{k}={kinds.count(k)}" for k in sorted(set(kinds))
    ))

    expected = {"agent_started", "thinking", "tool_call", "chart_data", "artifact", "agent_finished"}
    missing = expected - set(kinds)
    if missing:
        print(f"  [FAIL] missing kinds: {missing}")
        return 1

    charts_by_name: dict[str, int] = {}
    for e in events:
        if e.get("kind") == "chart_data":
            charts_by_name[e.get("chart", "?")] = charts_by_name.get(e.get("chart", "?"), 0) + 1

    expected_charts = {"cluster_iter", "cluster_scatter", "capacity_box", "fingerprint", "cluster_summary"}
    missing_charts = expected_charts - set(charts_by_name)
    if missing_charts:
        print(f"  [FAIL] missing chart_data charts: {missing_charts}")
        return 1

    scatter = next(e for e in events if e.get("kind") == "chart_data" and e.get("chart") == "cluster_scatter")
    summary = next(e for e in events if e.get("kind") == "chart_data" and e.get("chart") == "cluster_summary")
    k = scatter["data"]["k"]
    points = scatter["data"]["points"]
    axes = scatter["data"]["axes"]
    print(f"  [OK] k={k}, {len(points)} points, axes={axes}")
    print(f"  [OK] charts: {charts_by_name}")

    # Sanity: axes should now point at IR / Capacity (paper-style scatter)
    if axes.get("x") != "IR" or axes.get("y") != "Capacity":
        print(f"  [FAIL] unexpected axes: {axes} (want IR vs Capacity)")
        return 1

    # When target_k is pinned, the GA must return *exactly* that many clusters.
    if target_k is not None and k != target_k:
        print(f"  [FAIL] target_k={target_k} but got k={k}")
        return 1

    decisions = [c["decision"]["choice"] for c in summary["data"]["per_cluster"]]
    print(f"  [OK] recycle decisions: {decisions}")

    if session.pipeline_state.get("cluster") != "done":
        print(f"  [FAIL] pipeline_state.cluster = {session.pipeline_state.get('cluster')}")
        return 1
    names = [a["name"] for a in session.artifacts]
    if "Cluster_label.xlsx" not in names:
        print(f"  [FAIL] missing Cluster_label.xlsx artifact, got {names}")
        return 1
    print(f"  [OK] artifacts: {names}")
    return 0


def main() -> int:
    print("== test_cluster_agent (adaptive) ==")
    rc = asyncio.run(run_one_shot())
    if rc != 0:
        print("FAILED")
        return rc
    print("== test_cluster_agent (target_k=6) ==")
    rc = asyncio.run(run_one_shot(target_k=6, session_id="test_cluster_k6"))
    print("PASSED" if rc == 0 else "FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
