"""Integration tests for the workflow-plan flow inside pipeline.handle_message.

Covers:
  * plan_workflow tool from the LLM router → plan persisted on session +
    plan_created event emitted + plan markdown posted to chat.
  * plan-confirm fast path executes the current step (we stub the
    underlying agent so no real LLM / IO runs).
  * skip / cancel actions work.
  * After a step completes successfully the next step prompt is posted.
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
from app.backend.services import llm_router, planner  # noqa: E402


def _make_session() -> Session:
    return Session(session_id="test-pipeline-plan")


def _make_plan_workflow_decision() -> llm_router.IntentDecision:
    return llm_router.IntentDecision(
        tool="plan_workflow",
        next_agent="planner",
        intent="plan_workflow",
        params={"goal": "Run pipeline with k=4", "target_k": 4},
        confidence=0.9,
        needs_clarification=False,
        clarification=None,
        reasoning="user asked for full run",
    )


def _patch_no_ws(*more):
    """Patch the WS-touching helpers + store.update so handle_message can
    run without a live event loop subscriber. Returns a context manager."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch("app.backend.orchestrator.pipeline.bus.publish", new=AsyncMock()))
    stack.enter_context(patch("app.backend.orchestrator.pipeline.store.update", new=AsyncMock()))
    stack.enter_context(
        patch("app.backend.orchestrator.pipeline.conversation_memory.refresh_in_background")
    )
    for cm in more:
        stack.enter_context(cm)
    return stack


def test_plan_workflow_persists_plan() -> int:
    s = _make_session()
    plan_reply = (
        '{"goal": "Run pipeline with k=4", "steps": ['
        '{"agent": "designer", "description": "Design DDIA", "params": {"mode": "lightweight"}},'
        '{"agent": "compute", "description": "Compute MMD", "params": {}},'
        '{"agent": "cluster", "description": "Cluster k=4", "params": {"target_k": 4}}'
        ']}'
    )
    with _patch_no_ws(
        patch.object(pipeline.RouterAgent, "run",
                     new=AsyncMock(return_value=llm_router.decision_to_pipeline_dict(_make_plan_workflow_decision()))),
        patch("app.backend.services.planner.llm_client.chat", return_value=plan_reply),
    ):
        result = asyncio.run(pipeline.handle_message(s, "把整个流程跑一遍 k=4"))

    assert result["routed_to"] == "planner", result
    plan = s.context.get("active_plan")
    assert isinstance(plan, dict), "active_plan should be stored on session"
    assert plan["goal"] == "Run pipeline with k=4"
    assert plan["status"] == "awaiting_confirmation"
    assert len(plan["steps"]) == 3
    # Router-hint propagation: target_k must reach the cluster step even
    # if the planner LLM forgot to include it.
    assert plan["steps"][2]["params"]["target_k"] == 4
    return 0


def test_plan_workflow_falls_back_when_planner_fails() -> int:
    s = _make_session()
    with _patch_no_ws(
        patch.object(pipeline.RouterAgent, "run",
                     new=AsyncMock(return_value=llm_router.decision_to_pipeline_dict(_make_plan_workflow_decision()))),
        # planner LLM returns garbage → planner.generate_plan returns None
        # → pipeline must fall back to planner.default_plan().
        patch("app.backend.services.planner.llm_client.chat", return_value="not JSON"),
    ):
        asyncio.run(pipeline.handle_message(s, "搞起来"))

    plan = s.context.get("active_plan")
    assert isinstance(plan, dict)
    assert len(plan["steps"]) == 3
    assert [step["agent"] for step in plan["steps"]] == ["designer", "compute", "cluster"]
    return 0


def test_plan_confirm_fast_path_runs_step() -> int:
    s = _make_session()
    s.context["active_plan"] = planner.WorkflowPlan(
        goal="g",
        steps=[
            planner.PlanStep(id=0, agent="compute", description="Compute", params={}),
            planner.PlanStep(id=1, agent="cluster", description="Cluster", params={"target_k": 4}),
        ],
        current_idx=0,
        status="awaiting_confirmation",
    ).to_dict()

    fake_compute = AsyncMock(return_value={"passed": True, "computed": 71})

    class _FakeCompute:
        def __init__(self, _session):
            pass
        async def run(self, **kw):
            return await fake_compute(**kw)

    with _patch_no_ws(
        patch.dict(pipeline._AGENT_REGISTRY, {"compute": _FakeCompute, "cluster": _FakeCompute}),
        # No router should ever be called for the fast path.
        patch.object(pipeline.RouterAgent, "run",
                     new=AsyncMock(side_effect=AssertionError("router must not be called for fast-path"))),
    ):
        result = asyncio.run(pipeline.handle_message(s, "继续"))

    assert result["routed_to"] == "planner", result
    plan = s.context["active_plan"]
    assert plan["steps"][0]["status"] == "done", plan
    assert plan["current_idx"] == 1, plan
    assert plan["status"] == "in_progress", plan
    fake_compute.assert_called_once()
    return 0


def test_plan_skip_advances_without_running() -> int:
    s = _make_session()
    s.context["active_plan"] = planner.WorkflowPlan(
        goal="g",
        steps=[
            planner.PlanStep(id=0, agent="compute", description="Compute", params={}),
            planner.PlanStep(id=1, agent="cluster", description="Cluster", params={}),
        ],
        current_idx=0,
        status="awaiting_confirmation",
    ).to_dict()

    must_not_run = AsyncMock(side_effect=AssertionError("step must not execute on skip"))

    class _ShouldNotRun:
        def __init__(self, _session): pass
        async def run(self, **kw):
            return await must_not_run(**kw)

    with _patch_no_ws(
        patch.dict(pipeline._AGENT_REGISTRY, {"compute": _ShouldNotRun, "cluster": _ShouldNotRun}),
        patch.object(pipeline.RouterAgent, "run",
                     new=AsyncMock(side_effect=AssertionError("router must not be called"))),
    ):
        result = asyncio.run(pipeline.handle_message(s, "跳过"))

    plan = s.context["active_plan"]
    assert plan["steps"][0]["status"] == "skipped", plan
    assert plan["current_idx"] == 1, plan
    return 0


def test_plan_cancel_sets_status() -> int:
    s = _make_session()
    s.context["active_plan"] = planner.WorkflowPlan(
        goal="g",
        steps=[planner.PlanStep(id=0, agent="compute", description="x", params={})],
        current_idx=0,
        status="awaiting_confirmation",
    ).to_dict()

    with _patch_no_ws(
        patch.object(pipeline.RouterAgent, "run",
                     new=AsyncMock(side_effect=AssertionError("router must not be called"))),
    ):
        asyncio.run(pipeline.handle_message(s, "取消计划"))

    plan = s.context["active_plan"]
    assert plan["status"] == "cancelled", plan
    return 0


def test_plan_fast_path_not_used_when_no_plan() -> int:
    """Without an active plan, 'continue' shouldn't be misinterpreted as
    plan control — it must go through the smart router."""
    s = _make_session()
    fake_router_dec = {"intent": "chitchat", "next_agent": "chat", "source": "llm-router",
                       "tool": "answer_question"}

    chat_run = AsyncMock(return_value={"reply": "sure"})

    class _FakeChat:
        def __init__(self, _session): pass
        async def run(self, **kw):
            return await chat_run(**kw)

    with _patch_no_ws(
        patch.dict(pipeline._AGENT_REGISTRY, {"chat": _FakeChat}),
        patch.object(pipeline.RouterAgent, "run",
                     new=AsyncMock(return_value=fake_router_dec)),
    ):
        result = asyncio.run(pipeline.handle_message(s, "继续"))

    assert result["routed_to"] == "chat", result
    chat_run.assert_called_once()
    return 0


def test_direct_agent_run_does_not_post_separate_narrator_message() -> int:
    """Direct worker dispatch should rely on the agent's own summary message.

    The old behavior ran `_narrate` after every successful Designer / Compute /
    Cluster result, adding a second router bubble like "Would you like to...".
    That split the result and guidance across two assistant messages.
    """
    s = _make_session()
    fake_router_dec = {
        "intent": "compute_mmd",
        "next_agent": "compute",
        "source": "llm-router",
        "tool": "compute_mmd",
    }
    compute_run = AsyncMock(return_value={"passed": True, "computed": 71})

    class _FakeCompute:
        def __init__(self, _session): pass
        async def run(self, **kw):
            return await compute_run(**kw)

    with _patch_no_ws(
        patch.dict(pipeline._AGENT_REGISTRY, {"compute": _FakeCompute}),
        patch.object(pipeline.RouterAgent, "run",
                     new=AsyncMock(return_value=fake_router_dec)),
        patch("app.backend.orchestrator.pipeline._narrate",
              new=AsyncMock(side_effect=AssertionError("direct path must not narrate separately"))),
    ):
        result = asyncio.run(pipeline.handle_message(s, "compute now"))

    assert result["routed_to"] == "compute", result
    compute_run.assert_called_once()
    return 0


def main() -> int:
    tests = [
        test_plan_workflow_persists_plan,
        test_plan_workflow_falls_back_when_planner_fails,
        test_plan_confirm_fast_path_runs_step,
        test_plan_skip_advances_without_running,
        test_plan_cancel_sets_status,
        test_plan_fast_path_not_used_when_no_plan,
        test_direct_agent_run_does_not_post_separate_narrator_message,
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
