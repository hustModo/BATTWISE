"""Tests for the cluster parameter confirmation gate.

The pipeline now pauses on the first ClusterAgent dispatch and shows the
effective GA-KMeans knobs (target_k / pop_size / max_iter /
feedback_metric) so the user can confirm or tune them in chat — mirroring
the existing DDIA design gate.

Covered:
  * First cluster intent → pending flag set, prompt posted, ClusterAgent
    NOT dispatched.
  * Sticky cluster_user_params + new target_k from the same turn show up
    in the prompt.
  * Pending + confirm word ("默认") → ClusterAgent dispatches.
  * Pending + cluster cancel ("取消" / "不聚类") → flag cleared, cluster
    NOT dispatched.
  * Pending + further tuning ("pop=300 反馈=方差") → flag stays, new
    params persisted, prompt re-shown with updated values.
  * Plan execution path bypasses the gate entirely (plans run inline via
    _run_current_plan_step, not through handle_message).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.orchestrator import pipeline  # noqa: E402
from app.backend.orchestrator.session_store import Session  # noqa: E402


def _new_session(sid: str) -> Session:
    s = Session(session_id=sid)
    s.mode = "lightweight"
    s.active_dataset_id = "builtin-first-200"
    return s


def _mk_router_decision(
    *, target_k: int | None = None, cparams: Dict[str, Any] | None = None,
    next_agent: str = "cluster", source: str = "regex",
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "intent": "cluster",
        "next_agent": next_agent,
        "source": source,
        "confidence": 0.95,
    }
    if target_k is not None:
        d["target_k"] = target_k
    if cparams:
        d["cluster_params"] = cparams
    return d


class _RouterStub:
    """Drop-in for RouterAgent that returns a pre-baked decision."""

    def __init__(self, decision: Dict[str, Any]):
        self._decision = decision

    def __call__(self, session: Session):
        return self  # constructor returns instance

    async def run(self, user_text: str) -> Dict[str, Any]:  # noqa: D401
        return dict(self._decision)


def _patch_pipeline(router_decision: Dict[str, Any]):
    """Common patches: router stub + silenced store/bus + cluster mock.

    Returns a context manager + ``cluster_mock`` for inspection.
    """
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch("app.backend.orchestrator.pipeline.RouterAgent",
                              _RouterStub(router_decision)))
    stack.enter_context(patch("app.backend.orchestrator.pipeline.store.update",
                              new=AsyncMock()))
    stack.enter_context(patch("app.backend.orchestrator.pipeline.bus.publish",
                              new=AsyncMock()))
    stack.enter_context(patch("app.backend.orchestrator.pipeline.conversation_memory."
                              "refresh_in_background", new=lambda *a, **kw: None))

    # Mock ClusterAgent so we can assert it WAS or WASN'T dispatched.
    cluster_cls = type("_ClusterStub", (), {})
    cluster_runs: List[Dict[str, Any]] = []

    class _ClusterStub:
        def __init__(self, session: Session):
            self.session = session

        async def run(self, *, user_text: str, **kwargs):  # noqa: D401
            cluster_runs.append({"user_text": user_text, **kwargs})
            return {"ok": True}

    stack.enter_context(patch.dict(
        pipeline._AGENT_REGISTRY,
        {"cluster": _ClusterStub},
    ))

    stack.enter_context(patch("app.backend.orchestrator.pipeline._narrate",
                              new=AsyncMock()))
    return stack, cluster_runs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_first_cluster_intent_triggers_gate() -> int:
    s = _new_session("t-cluster-gate-1")
    stack, runs = _patch_pipeline(_mk_router_decision())
    with stack:
        result = asyncio.run(pipeline.handle_message(s, "聚类一下"))
    assert result["routed_to"] == "cluster", result
    assert result["result"]["awaiting"] == "cluster_parameter_confirmation"
    assert s.context.get("pending_cluster_confirmation") is True
    assert runs == [], "cluster should NOT have been dispatched yet"
    # An assistant prompt was added.
    assistant_msgs = [m for m in s.messages if m["role"] == "assistant"]
    assert assistant_msgs, "expected a cluster-confirmation prompt"
    assert "聚类" in assistant_msgs[-1]["text"]
    assert "默认" in assistant_msgs[-1]["text"]
    return 0


def test_prompt_reflects_current_turn_params() -> int:
    s = _new_session("t-cluster-gate-2")
    # Sticky from a prior turn.
    s.context["cluster_user_params"] = {"feedback_metric": "variance"}
    # Plus a fresh target_k + pop_size in this turn.
    decision = _mk_router_decision(
        target_k=4, cparams={"pop_size": 300},
    )
    stack, _ = _patch_pipeline(decision)
    with stack:
        asyncio.run(pipeline.handle_message(s, "用 k=4 pop=300 聚类"))
    prompt = s.messages[-1]["text"]
    # Effective params must merge sticky + this-turn.
    assert "target_k: **4**" in prompt, prompt
    assert "pop_size: **300**" in prompt, prompt
    assert "feedback_metric: **variance**" in prompt, prompt
    assert "OCV=1.0" in prompt and "IR=1.0" in prompt and "CDDI=1.0" in prompt, prompt
    # Both got persisted onto sticky storage.
    sticky = s.context.get("cluster_user_params") or {}
    assert sticky.get("pop_size") == 300
    assert sticky.get("feedback_metric") == "variance"
    assert s.context.get("target_k") == 4
    return 0


def test_confirm_word_dispatches_cluster() -> int:
    s = _new_session("t-cluster-gate-3")
    s.context["pending_cluster_confirmation"] = True
    s.context["target_k"] = 5
    s.context["cluster_user_params"] = {"pop_size": 250}
    # Router would classify "默认" as chitchat — but the gate's regex
    # intercepts first.
    decision = _mk_router_decision(next_agent="chat", source="llm-classify")
    decision["intent"] = "chitchat"
    stack, runs = _patch_pipeline(decision)
    with stack:
        result = asyncio.run(pipeline.handle_message(s, "默认"))
    assert result["routed_to"] == "cluster", result
    assert "pending_cluster_confirmation" not in s.context
    assert len(runs) == 1, runs
    assert runs[0]["target_k"] == 5
    assert runs[0]["cluster_params"] == {"pop_size": 250}
    return 0


def test_natural_confirm_sentence_dispatches_cluster() -> int:
    s = _new_session("t-cluster-gate-natural-confirm")
    s.context["pending_cluster_confirmation"] = True
    s.context["target_k"] = 4
    s.context["cluster_user_params"] = {"max_iter": 80}
    # This mirrors a real user reply after seeing the confirmation card:
    # not just "默认", but a natural sentence with echoed parameters.
    decision = _mk_router_decision(next_agent="chat", source="llm-classify")
    decision["intent"] = "chitchat"
    stack, runs = _patch_pipeline(decision)
    with stack:
        result = asyncio.run(
            pipeline.handle_message(s, "开始聚类 k=4，迭代=80，就按这个跑")
        )
    assert result["routed_to"] == "cluster", result
    assert "pending_cluster_confirmation" not in s.context
    assert len(runs) == 1, runs
    assert runs[0]["target_k"] == 4
    assert runs[0]["cluster_params"] == {"max_iter": 80}
    return 0


def test_cancel_word_clears_gate_no_dispatch() -> int:
    s = _new_session("t-cluster-gate-4")
    s.context["pending_cluster_confirmation"] = True
    # Router classifies "取消" as chat (regex inside gate intercepts).
    decision = _mk_router_decision(next_agent="chat", source="llm-classify")
    decision["intent"] = "chitchat"
    stack, runs = _patch_pipeline(decision)
    with stack:
        result = asyncio.run(pipeline.handle_message(s, "取消"))
    assert result["result"]["cancelled"] is True
    assert "pending_cluster_confirmation" not in s.context
    assert runs == [], "cluster must not run when user cancelled"
    return 0


def test_pending_with_more_tuning_keeps_gate_open() -> int:
    s = _new_session("t-cluster-gate-5")
    s.context["pending_cluster_confirmation"] = True
    decision = _mk_router_decision(
        target_k=6, cparams={"feedback_metric": "dbi"},
    )
    stack, runs = _patch_pipeline(decision)
    with stack:
        result = asyncio.run(pipeline.handle_message(s, "k=6 反馈=dbi"))
    assert result["result"]["awaiting"] == "cluster_parameter_confirmation"
    # Gate stays armed until the user confirms.
    assert s.context.get("pending_cluster_confirmation") is True
    assert s.context.get("target_k") == 6
    assert s.context["cluster_user_params"]["feedback_metric"] == "dbi"
    # Prompt was re-shown with new values.
    assert "target_k: **6**" in s.messages[-1]["text"]
    assert "feedback_metric: **dbi**" in s.messages[-1]["text"]
    assert runs == [], "cluster must not run while user is still tuning"
    return 0


def test_pending_with_weight_tuning_keeps_gate_open() -> int:
    s = _new_session("t-cluster-gate-weights")
    s.context["pending_cluster_confirmation"] = True
    decision = _mk_router_decision(next_agent="chat", source="llm-classify")
    decision["intent"] = "chitchat"
    stack, runs = _patch_pipeline(decision)
    with stack:
        result = asyncio.run(pipeline.handle_message(s, "OCV=1 IR=1 DDIA=2"))
    assert result["result"]["awaiting"] == "cluster_parameter_confirmation"
    assert s.context.get("pending_cluster_confirmation") is True
    weights = s.context["cluster_user_params"]["feature_weights"]
    assert weights == {"ocv": 1.0, "ir": 1.0, "ddia": 2.0}, weights
    assert "CDDI=2.0" in s.messages[-1]["text"], s.messages[-1]["text"]
    assert runs == []
    return 0


def test_confirm_sentence_with_weights_dispatches_cluster() -> int:
    s = _new_session("t-cluster-gate-confirm-weights")
    s.context["pending_cluster_confirmation"] = True
    decision = _mk_router_decision(next_agent="chat", source="llm-classify")
    decision["intent"] = "chitchat"
    stack, runs = _patch_pipeline(decision)
    with stack:
        result = asyncio.run(pipeline.handle_message(s, "开始聚类 OCV=1 IR=1 DDIA=2"))
    assert result["routed_to"] == "cluster", result
    assert len(runs) == 1, runs
    assert runs[0]["cluster_params"]["feature_weights"] == {
        "ocv": 1.0,
        "ir": 1.0,
        "ddia": 2.0,
    }
    return 0


def test_pending_but_unrelated_message_nudges_user() -> int:
    s = _new_session("t-cluster-gate-6")
    s.context["pending_cluster_confirmation"] = True
    # Router classifies as chat — user asked an unrelated question.
    decision = _mk_router_decision(next_agent="chat", source="llm-classify")
    decision["intent"] = "chitchat"
    stack, runs = _patch_pipeline(decision)
    with stack:
        result = asyncio.run(pipeline.handle_message(s, "what does DBI mean"))
    # We didn't run cluster, and we re-nudged the user about the gate.
    assert result["result"]["awaiting"] == "cluster_parameter_confirmation"
    assert s.context.get("pending_cluster_confirmation") is True
    assert "参数确认" in s.messages[-1]["text"]
    assert runs == []
    return 0


def test_plan_execution_bypasses_gate() -> int:
    """Plan steps drive ClusterAgent directly via _run_current_plan_step
    without going through handle_message → no gate fires.
    """
    s = _new_session("t-cluster-gate-7")
    # Seed a plan with one cluster step.
    from app.backend.services import planner
    plan = planner.WorkflowPlan(
        goal="cluster k=4",
        steps=[planner.PlanStep(id=0, agent="cluster", description="cluster",
                                params={"target_k": 4, "pop_size": 250})],
        status="awaiting_confirmation",
    )
    s.context["active_plan"] = plan.to_dict()

    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch("app.backend.orchestrator.pipeline.store.update",
                              new=AsyncMock()))
    stack.enter_context(patch("app.backend.orchestrator.pipeline.bus.publish",
                              new=AsyncMock()))
    stack.enter_context(patch("app.backend.orchestrator.pipeline.conversation_memory."
                              "refresh_in_background", new=lambda *a, **kw: None))
    stack.enter_context(patch("app.backend.orchestrator.pipeline._narrate",
                              new=AsyncMock()))

    cluster_runs: List[Dict[str, Any]] = []

    class _ClusterStub:
        def __init__(self, session: Session): pass

        async def run(self, *, user_text: str = "", **kw):
            cluster_runs.append({"user_text": user_text, **kw})
            return {"ok": True}

    stack.enter_context(patch.dict(pipeline._AGENT_REGISTRY, {"cluster": _ClusterStub}))

    with stack:
        # "确认" + active plan → plan fast path → _handle_plan_action(confirm)
        # → _run_current_plan_step → ClusterAgent.run, no gate.
        asyncio.run(pipeline.handle_message(s, "确认"))
    assert len(cluster_runs) == 1, cluster_runs
    assert cluster_runs[0]["target_k"] == 4
    assert cluster_runs[0]["cluster_params"] == {"pop_size": 250}
    # The gate flag must NOT have been raised by the plan path.
    assert "pending_cluster_confirmation" not in s.context
    return 0


def main() -> int:
    tests = [
        test_first_cluster_intent_triggers_gate,
        test_prompt_reflects_current_turn_params,
        test_confirm_word_dispatches_cluster,
        test_natural_confirm_sentence_dispatches_cluster,
        test_cancel_word_clears_gate_no_dispatch,
        test_pending_with_more_tuning_keeps_gate_open,
        test_pending_with_weight_tuning_keeps_gate_open,
        test_confirm_sentence_with_weights_dispatches_cluster,
        test_pending_but_unrelated_message_nudges_user,
        test_plan_execution_bypasses_gate,
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
            import traceback
            print(f"  [ERROR] {fn.__name__}: {exc!r}")
            traceback.print_exc()
            failed += 1
    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
