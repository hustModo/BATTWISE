"""Unit + integration tests for plan-step parameter editing.

Covers:
  * services/planner.validate_step_params — schema-driven coercion +
    clamping, unknown-key drop, enum sanitisation, int_list filtering.
  * pipeline.handle_action(plan_update_step, ...) — patches the
    correct step; rejects edits on non-pending steps; rejects when no
    plan is active; clears a key when value is empty string / None;
    emits plan_updated WS event.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.orchestrator import pipeline  # noqa: E402
from app.backend.orchestrator.session_store import Session  # noqa: E402
from app.backend.services import planner  # noqa: E402


# ---------------------------------------------------------------------------
# planner.validate_step_params
# ---------------------------------------------------------------------------

def test_validate_designer_clamps_int_and_keeps_enum() -> int:
    out = planner.validate_step_params(
        "designer",
        {"mode": "full_eoh", "max_fe": 99, "pop_size": 6, "garbage": "x"},
    )
    # legacy mode normalized, max_fe clamped to 30 (upper bound),
    # pop_size kept, garbage dropped.
    assert out == {"mode": "reflective_search", "max_fe": 30, "pop_size": 6}, out
    return 0


def test_validate_string_ints_get_coerced() -> int:
    out = planner.validate_step_params("cluster", {"target_k": "4", "pop_size": "120"})
    assert out == {"target_k": 4, "pop_size": 120}, out
    return 0


def test_validate_drops_bad_enum() -> int:
    out = planner.validate_step_params("cluster", {"feedback_metric": "wat"})
    assert out == {}, out
    return 0


def test_validate_int_list_filters_and_sorts() -> int:
    out = planner.validate_step_params(
        "refine", {"cluster_ids": [3, 0, "5", 100, "abc", 3]},
    )
    # 100 dropped (> max 50), "abc" dropped, dedup + sort.
    assert out == {"cluster_ids": [0, 3, 5]}, out
    return 0


def test_validate_empty_for_unknown_agent() -> int:
    out = planner.validate_step_params("teapot", {"x": 1})
    assert out == {}
    return 0


def test_validate_drops_none_and_empty_string() -> int:
    out = planner.validate_step_params(
        "designer", {"mode": None, "max_fe": "", "pop_size": 6},
    )
    assert out == {"pop_size": 6}, out
    return 0


# ---------------------------------------------------------------------------
# pipeline.handle_action with kind=plan_update_step
# ---------------------------------------------------------------------------

def _seed_plan(s: Session) -> None:
    plan = planner.WorkflowPlan(
        goal="g",
        steps=[
            planner.PlanStep(id=0, agent="designer", description="d",
                             params={"mode": "lightweight"}),
            planner.PlanStep(id=1, agent="cluster", description="c",
                             params={"target_k": 4}),
        ],
        current_idx=0,
        status="awaiting_confirmation",
    )
    s.context["active_plan"] = plan.to_dict()


def _patch_ws():
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch("app.backend.orchestrator.pipeline.bus.publish", new=AsyncMock()))
    stack.enter_context(patch("app.backend.orchestrator.pipeline.store.update", new=AsyncMock()))
    return stack


def test_plan_update_step_happy_path() -> int:
    s = Session(session_id="t-edit-1")
    _seed_plan(s)
    with _patch_ws():
        result = asyncio.run(pipeline.handle_action(
            s,
            {"type": "plan_update_step", "step_id": 1,
             "params": {"target_k": 6, "feedback_metric": "variance"}},
        ))
    assert result["ok"] is True, result
    plan = s.context["active_plan"]
    assert plan["steps"][1]["params"]["target_k"] == 6
    assert plan["steps"][1]["params"]["feedback_metric"] == "variance"
    # The first step is untouched.
    assert plan["steps"][0]["params"] == {"mode": "lightweight"}
    return 0


def test_plan_update_step_clamps_out_of_range() -> int:
    s = Session(session_id="t-edit-2")
    _seed_plan(s)
    with _patch_ws():
        result = asyncio.run(pipeline.handle_action(
            s,
            {"type": "plan_update_step", "step_id": 1,
             "params": {"target_k": 999}},
        ))
    assert result["ok"] is True, result
    assert s.context["active_plan"]["steps"][1]["params"]["target_k"] == 12
    return 0


def test_plan_update_step_clear_value_with_empty_string() -> int:
    s = Session(session_id="t-edit-3")
    _seed_plan(s)
    with _patch_ws():
        result = asyncio.run(pipeline.handle_action(
            s,
            {"type": "plan_update_step", "step_id": 1, "params": {"target_k": ""}},
        ))
    assert result["ok"] is True, result
    assert "target_k" not in s.context["active_plan"]["steps"][1]["params"]
    return 0


def test_plan_update_step_rejected_when_running() -> int:
    s = Session(session_id="t-edit-4")
    _seed_plan(s)
    s.context["active_plan"]["steps"][1]["status"] = "done"
    with _patch_ws():
        result = asyncio.run(pipeline.handle_action(
            s, {"type": "plan_update_step", "step_id": 1, "params": {"target_k": 5}},
        ))
    assert result["ok"] is False, result
    assert "pending" in result["error"]
    # Original target_k must not have been overwritten.
    assert s.context["active_plan"]["steps"][1]["params"]["target_k"] == 4
    return 0


def test_plan_update_step_rejected_when_no_plan() -> int:
    s = Session(session_id="t-edit-5")
    with _patch_ws():
        result = asyncio.run(pipeline.handle_action(
            s, {"type": "plan_update_step", "step_id": 0, "params": {}},
        ))
    assert result["ok"] is False
    assert "no active plan" in result["error"]
    return 0


def test_plan_update_step_unknown_step_id() -> int:
    s = Session(session_id="t-edit-6")
    _seed_plan(s)
    with _patch_ws():
        result = asyncio.run(pipeline.handle_action(
            s, {"type": "plan_update_step", "step_id": 99, "params": {}},
        ))
    assert result["ok"] is False
    assert "step 99 not found" in result["error"]
    return 0


def test_plan_update_step_emits_plan_updated_event() -> int:
    s = Session(session_id="t-edit-7")
    _seed_plan(s)
    publish_mock = AsyncMock()
    with patch("app.backend.orchestrator.pipeline.bus.publish", new=publish_mock), \
         patch("app.backend.orchestrator.pipeline.store.update", new=AsyncMock()):
        asyncio.run(pipeline.handle_action(
            s, {"type": "plan_update_step", "step_id": 1,
                "params": {"target_k": 6}},
        ))
    assert publish_mock.called, "plan_updated WS event must fire"
    # The published event's kind is plan_updated.
    ev = publish_mock.call_args.args[0]
    assert ev.kind == "plan_updated", ev.kind
    return 0


def main() -> int:
    tests = [
        test_validate_designer_clamps_int_and_keeps_enum,
        test_validate_string_ints_get_coerced,
        test_validate_drops_bad_enum,
        test_validate_int_list_filters_and_sorts,
        test_validate_empty_for_unknown_agent,
        test_validate_drops_none_and_empty_string,
        test_plan_update_step_happy_path,
        test_plan_update_step_clamps_out_of_range,
        test_plan_update_step_clear_value_with_empty_string,
        test_plan_update_step_rejected_when_running,
        test_plan_update_step_rejected_when_no_plan,
        test_plan_update_step_unknown_step_id,
        test_plan_update_step_emits_plan_updated_event,
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
