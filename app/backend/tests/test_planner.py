"""Unit tests for services.planner — LLM-driven workflow plan generation."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.services import planner  # noqa: E402


SNAPSHOT = {
    "mode": "lightweight",
    "active_dataset_id": "builtin-first-200s",
    "selected_clusters": [],
    "pipeline_state": {"router": "idle", "designer": "idle", "compute": "idle",
                       "cluster": "idle", "refine": "idle"},
    "sticky_eoh_params": None,
    "sticky_cluster_params": None,
}


def _mock(reply: str):
    return patch("app.backend.services.planner.llm_client.chat", return_value=reply)


def test_plan_happy_path() -> int:
    reply = (
        '{"goal": "Run the standard pipeline end-to-end with k=4",'
        ' "steps": ['
        '  {"agent": "designer", "description": "Design DDIA (lightweight)", "params": {"mode": "lightweight"}},'
        '  {"agent": "compute", "description": "Compute MMD across all cells", "params": {}},'
        '  {"agent": "cluster", "description": "Cluster into 4 groups", "params": {"target_k": 4}}'
        ']}'
    )
    with _mock(reply):
        plan = planner.generate_plan(user_goal="跑完整流程 k=4", session_snapshot=SNAPSHOT)
    assert plan is not None
    assert len(plan.steps) == 3
    assert plan.steps[0].agent == "designer"
    assert plan.steps[2].params["target_k"] == 4
    assert plan.status == "awaiting_confirmation"
    return 0


def test_plan_rejects_unknown_agent() -> int:
    reply = (
        '{"goal": "x", "steps": ['
        '  {"agent": "make_coffee", "description": "bad", "params": {}},'
        '  {"agent": "compute", "description": "ok", "params": {}}'
        ']}'
    )
    with _mock(reply):
        plan = planner.generate_plan(user_goal="x", session_snapshot=SNAPSHOT)
    assert plan is not None
    assert len(plan.steps) == 1  # the bad one was dropped
    assert plan.steps[0].agent == "compute"
    return 0


def test_plan_handles_garbage_returns_none() -> int:
    with _mock("not a JSON"):
        plan = planner.generate_plan(user_goal="x", session_snapshot=SNAPSHOT)
    assert plan is None
    return 0


def test_plan_caps_at_max_steps() -> int:
    steps = ",".join(
        f'{{"agent":"compute","description":"step {i}","params":{{}}}}' for i in range(20)
    )
    reply = f'{{"goal": "x", "steps": [{steps}]}}'
    with _mock(reply):
        plan = planner.generate_plan(user_goal="x", session_snapshot=SNAPSHOT)
    assert plan is not None
    assert len(plan.steps) <= planner.MAX_PLAN_STEPS
    return 0


def test_plan_dict_round_trip() -> int:
    p = planner.WorkflowPlan(
        goal="g",
        steps=[
            planner.PlanStep(id=0, agent="designer", description="d1", params={"mode": "lightweight"}),
            planner.PlanStep(id=1, agent="cluster", description="d2", params={"target_k": 4}),
        ],
        current_idx=1,
        status="in_progress",
    )
    d = p.to_dict()
    p2 = planner.WorkflowPlan.from_dict(d)
    assert p2.goal == "g"
    assert p2.current_idx == 1
    assert p2.status == "in_progress"
    assert p2.steps[1].agent == "cluster"
    assert p2.steps[1].params["target_k"] == 4
    return 0


def test_render_plan_markdown() -> int:
    p = planner.default_plan("run pipeline")
    md = planner.render_plan_markdown(p, header="Plan:")
    assert "Plan:" in md
    assert "Designer" in md
    assert "Compute" in md
    assert "Cluster" in md
    return 0


def test_default_plan_shape() -> int:
    p = planner.default_plan("x")
    assert len(p.steps) == 3
    assert [s.agent for s in p.steps] == ["designer", "compute", "cluster"]
    assert p.current_idx == 0
    return 0


def main() -> int:
    tests = [
        test_plan_happy_path,
        test_plan_rejects_unknown_agent,
        test_plan_handles_garbage_returns_none,
        test_plan_caps_at_max_steps,
        test_plan_dict_round_trip,
        test_render_plan_markdown,
        test_default_plan_shape,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  [OK] {fn.__name__}")
        except AssertionError as exc:
            print(f"  [FAIL] {fn.__name__}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {fn.__name__}: {exc!r}")
            failed += 1
    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
