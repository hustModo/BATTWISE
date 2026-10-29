"""Pipeline orchestrator: turns a user message into a chain of agent runs.

For M1 the pipeline only wires Router → (Chat | placeholder downstream agents).
Later milestones (M2..M4) plug in Designer / Compute / Cluster / Refine.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, Optional

from app.backend.agents.base import AgentBase
from app.backend.agents.chat_agent import ChatAgent
from app.backend.agents.cluster_agent import ClusterAgent
from app.backend.agents.compute_agent import ComputeAgent
from app.backend.agents.designer_agent import (
    DesignerAgent,
    REFLECTIVE_SEARCH_LIGHT_DEFAULTS,
)
from app.backend.agents.refinement_agent import RefinementAgent
from app.backend.agents.router_agent import (
    RouterAgent,
    _extract_cluster_params,
    _extract_designer_params,
    _extract_eoh_params,
)
from app.backend.naming import (
    LEGACY_REFLECTIVE_SEARCH_MODE,
    LIGHTWEIGHT_MODE,
    REFLECTIVE_SEARCH_MODE,
    normalize_mode,
)
from app.backend.orchestrator.events import AgentEvent, bus, safe_json
from app.backend.orchestrator.session_store import Session, store
from app.backend.services import conversation_memory, narrator, planner
from app.backend.settings import settings

logger = logging.getLogger(__name__)


_AGENT_REGISTRY: Dict[str, type[AgentBase]] = {
    "router": RouterAgent,
    "chat": ChatAgent,
    "designer": DesignerAgent,
    "compute": ComputeAgent,
    "cluster": ClusterAgent,
    "refine": RefinementAgent,
}


def register_agent(name: str, cls: type[AgentBase]) -> None:
    """Allow later milestones to add Designer/Compute/Cluster/Refine."""
    _AGENT_REGISTRY[name] = cls


_DESIGN_CONFIRM_RE = re.compile(
    r"^\s*(?:"
    r"默认|确认|确定|可以|用默认|开始|开始吧|开始设计(?:.*)?"
    r"|按(?:这个|当前|以上|默认).*(?:开始|设计|执行|跑)"
    r"|就(?:这样|按这个).*(?:开始|设计|执行|跑)?"
    r"|直接(?:开始|设计|执行|跑)(?:.*)?"
    r"|go|start|confirm|ok(?:ay)?|run(?:\s+it)?"
    r")\s*[。.!！?？]*\s*$",
    re.I,
)
_DESIGN_CANCEL_RE = re.compile(r"^\s*(取消|算了|不设计|cancel|nevermind|never\s+mind)\s*$", re.I)

# Cluster-stage equivalents of the design gate. Same vocabulary, just with
# "开始聚类" instead of "开始设计" and "不聚类" instead of "不设计".
_CLUSTER_CONFIRM_RE = re.compile(
    r"^\s*(?:"
    r"默认|确认|确定|可以|用默认|开始|开始吧|开始聚类(?:.*)?"
    r"|按(?:这个|当前|以上|默认).*(?:开始|聚类|执行|跑)"
    r"|就(?:这样|按这个).*(?:开始|聚类|执行|跑)?"
    r"|直接(?:开始|聚类|执行|跑)(?:.*)?"
    r"|go|start|confirm|ok(?:ay)?|run(?:\s+it)?"
    r")\s*[。.!！?？]*\s*$",
    re.I,
)
_CLUSTER_CANCEL_RE = re.compile(
    r"^\s*(取消|算了|不聚类|不要|cancel|nevermind|never\s+mind)\s*$", re.I,
)


# ---------------------------------------------------------------------------
# Plan-control regexes — short, unambiguous phrases that drive an
# already-active workflow plan WITHOUT a round-trip to the LLM router.
# Anything longer / more ambiguous falls through to the smart router which
# can emit ``advance_plan`` with an explicit action.
# ---------------------------------------------------------------------------

_PLAN_CONFIRM_RE = re.compile(
    r"^\s*("
    r"(?:run|start|go|begin|confirm|ok(?:ay)?|sure|do\s+it|let'?s\s+go|proceed|continue|next)"
    r"|确认|确定|开始|开始执行|继续|下一步|下一?步|可以|好的?|没问题|按计划|跑吧|跑起来|执行"
    r")\s*[.!?。！]*\s*$",
    re.I,
)

_PLAN_SKIP_RE = re.compile(
    r"^\s*(skip|skip\s+(?:this|it)|skip\s+step|pass)\s*[.!?。！]*\s*$"
    r"|^\s*(跳过|跳过这步|跳过此步|不跑这步|不要这步)\s*[.!?。！]*\s*$",
    re.I,
)

_PLAN_CANCEL_RE = re.compile(
    r"^\s*("
    r"(?:cancel|stop|abort|kill)\s+(?:the\s+)?plan"
    r"|abandon\s+plan|forget\s+the\s+plan"
    r")\s*[.!?。！]*\s*$"
    r"|^\s*(取消计划|停止计划|放弃计划|不跑了|计划取消)\s*[.!?。！]*\s*$",
    re.I,
)


async def _router_say(session: Session, text: str) -> None:
    """Persist + push a router assistant message without dispatching an agent."""
    msg = session.add_message("assistant", text, agent="router")
    await store.update(session)
    await bus.publish(
        AgentEvent(
            kind="message",
            session_id=session.session_id,
            agent="router",
            payload={"message": safe_json(msg)},
        )
    )


def _design_parameter_prompt(session: Session) -> str:
    mode = session.mode
    dataset = session.active_dataset_id
    eoh_params = dict(session.context.get("eoh_user_params") or {})
    designer_params = dict(session.context.get("designer_user_params") or {})
    max_fe = eoh_params.get("max_fe", REFLECTIVE_SEARCH_LIGHT_DEFAULTS["max_fe"])
    pop_size = eoh_params.get("pop_size", REFLECTIVE_SEARCH_LIGHT_DEFAULTS["pop_size"])
    batch_size = eoh_params.get("batch_size", REFLECTIVE_SEARCH_LIGHT_DEFAULTS["batch_size"])
    repair_attempts = designer_params.get(
        "repair_attempts",
        settings.designer_max_repair_attempts,
    )
    sandbox_cells = designer_params.get(
        "sandbox_cells",
        settings.designer_sandbox_cells,
    )
    if normalize_mode(mode) == REFLECTIVE_SEARCH_MODE:
        param_line = (
            f"Reflective Search / CDDI 参数：max_fe={max_fe} 代，"
            f"pop_size={pop_size} 个体/代，batch_size={batch_size}。"
        )
    else:
        param_line = (
            f"Lightweight 设计参数：repair_attempts={repair_attempts}，"
            f"sandbox_cells={sandbox_cells}。"
            f"（Reflective Search 默认会用 max_fe={max_fe}, pop_size={pop_size}, batch_size={batch_size}。）"
        )
    return (
        "在开始 CDDI 设计前，请先确认参数：\n"
        f"- Dataset: `{dataset}`\n"
        f"- Mode: `{mode}`\n"
        f"- {param_line}\n\n"
        "如果使用当前参数，回复：`默认` / `确认` / `开始设计`。\n"
        "如果要改参数，可以直接说：`切换 Reflective Search，5 代，每代 6 个体，批 4`，"
        "或 `轻量化 repair_attempts=6 sandbox_cells=5`，"
        "或先在右上角切换 Mode 后再确认。"
    )


# Mirror of the ClusterAgent defaults — kept in sync with
# ``ClusterAgent.run`` (see app/backend/agents/cluster_agent.py). The
# confirmation prompt resolves the effective values from
# defaults <- sticky (cluster_user_params) <- decision (target_k /
# cluster_params just parsed from this user turn).
_CLUSTER_DEFAULTS = {
    "pop_size": 200,
    "max_iter": 80,
    "feedback_metric": "combined",
    "min_k": 2,
    "max_k": 12,
    "feature_weights": {"ocv": 1.0, "ir": 1.0, "ddia": 1.0},
}


def _effective_cluster_params(session: Session) -> Dict[str, Any]:
    """Defaults + sticky overrides — used in the confirmation prompt."""
    eff = dict(_CLUSTER_DEFAULTS)
    sticky = session.context.get("cluster_user_params") or {}
    if isinstance(sticky, dict):
        for k, v in sticky.items():
            if v is not None:
                if k == "feature_weights" and isinstance(v, dict):
                    weights = dict(eff.get("feature_weights") or {})
                    weights.update(v)
                    eff[k] = weights
                else:
                    eff[k] = v
    return eff


def _merge_cluster_params(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge cluster params while preserving per-feature weights one by one."""
    out = dict(existing or {})
    for k, v in (incoming or {}).items():
        if k == "feature_weights" and isinstance(v, dict):
            weights = dict(out.get("feature_weights") or {})
            weights.update(v)
            out["feature_weights"] = weights
        elif v is not None:
            out[k] = v
    return out


def _cluster_parameter_prompt(session: Session) -> str:
    """Markdown prompt shown before the first ClusterAgent dispatch.

    Lists the effective GA-KMeans knobs (target_k / k range, pop_size,
    max_iter, feedback_metric) so the user can either confirm or override
    with one chat line. ``session.context["target_k"]`` reflects the
    explicit fixed-k choice from the current turn; if absent we say "Auto
    (GA picks k in [min..max])".
    """
    eff = _effective_cluster_params(session)
    target_k = session.context.get("target_k")
    if target_k is not None:
        try:
            k_line = f"target_k: **{int(target_k)}** (fixed)"
        except (TypeError, ValueError):
            k_line = f"target_k: `{target_k}`"
    else:
        k_line = f"target_k: **Auto** (GA picks k ∈ [{eff['min_k']}..{eff['max_k']}])"
    weights = eff.get("feature_weights") or {}
    return (
        "在开始聚类前，请先确认参数：\n"
        f"- {k_line}\n"
        f"- GA pop_size: **{eff['pop_size']}**\n"
        f"- GA max_iter: **{eff['max_iter']}**\n"
        f"- feedback_metric: **{eff['feedback_metric']}**\n"
        f"- feature weights: **OCV={weights.get('ocv', 1.0)} / "
        f"IR={weights.get('ir', 1.0)} / CDDI={weights.get('ddia', 1.0)}** "
        "（默认 1/1/1，归一化后等权）\n\n"
        "如果使用以上参数，回复：`默认` / `确认` / `开始聚类`。\n"
        "如果要改参数，可以直接说：`k=4 种群=200 迭代=80 反馈=方差`，"
        "或 `k 3 到 8 反馈=DBI`，"
        "或 `OCV=1 IR=1 CDDI=2`。"
    )


async def handle_message(session: Session, user_text: str) -> Dict[str, Any]:
    session.add_message("user", user_text, agent=None)
    await store.update(session)

    # Schedule a running-summary refresh once handle_message returns. The
    # service throttles so a burst of short turns won't re-call the LLM
    # more than once every ``SUMMARY_MIN_INTERVAL_S`` seconds.
    def _schedule_summary_refresh() -> None:
        conversation_memory.refresh_in_background(session)

    # ---- workflow-plan fast path ---------------------------------------
    # If a plan is sitting on the session, short unambiguous control words
    # ("继续" / "go" / "skip" / "取消计划") shouldn't pay for an LLM
    # round-trip. They map straight to plan actions. Anything else falls
    # through to the smart router below.
    active_plan = session.context.get("active_plan")
    if isinstance(active_plan, dict) and active_plan.get("status") in (
        "awaiting_confirmation",
        "in_progress",
    ):
        plan_action = _classify_plan_control(user_text)
        if plan_action is not None:
            out = await _handle_plan_action(session, plan_action)
            _schedule_summary_refresh()
            return out

    # RouterAgent itself briefly marks `active_agent="router"` while it
    # classifies the message. Preserve the previously-running worker so a
    # conversational cancel can target Compute/Cluster/Designer correctly.
    active_before_router = session.active_agent
    router = RouterAgent(session)
    decision = await router.run(user_text)
    next_agent = decision.get("next_agent", "chat")

    # ---- LLM-router plan tools -----------------------------------------
    # The smart router can also drive a plan, either by drafting a fresh
    # one (``plan_workflow``) or by interpreting an ambiguous natural-
    # language plan-control reply (``advance_plan``).
    if decision.get("tool") == "plan_workflow":
        out = await _create_plan_from_decision(session, decision, user_text)
        _schedule_summary_refresh()
        return out
    if decision.get("tool") == "advance_plan" and isinstance(active_plan, dict):
        action = decision.get("plan_action") or "confirm"
        out = await _handle_plan_action(session, action)
        _schedule_summary_refresh()
        return out

    # ---- conversational cancel ------------------------------------------
    # User said "stop"/"abort"/"停" while an agent is running. Set only the
    # currently-running agent's cancel flag; setting all flags leaves stale
    # cancel requests behind and can abort the next unrelated run.
    if next_agent == "cancel":
        for flag in (
            "cancel_compute",
            "cancel_cluster",
            "cancel_reflective_search",
            "cancel_eoh",
        ):
            session.context[flag] = False
        cancel_flags = {
            "compute": "cancel_compute",
            "cluster": "cancel_cluster",
            "designer": "cancel_reflective_search",
        }
        active = active_before_router
        flag = cancel_flags.get(active or "")

        # Also clear any pending confirmation gates. Previously the
        # cancel branch returned BEFORE the gate logic, so saying
        # "取消" while the design / cluster prompt was waiting just
        # produced "No active computation is running" and the gate
        # stayed armed — the user couldn't actually get out. Now any
        # cancel phrase clears the gate too.
        cleared_gates: list[str] = []
        if session.context.pop("pending_design_confirmation", None):
            cleared_gates.append("design")
        if session.context.pop("pending_cluster_confirmation", None):
            cleared_gates.append("cluster")

        ack_bits: list[str] = []
        if flag:
            session.context[flag] = True
            if flag == "cancel_reflective_search":
                session.context["cancel_eoh"] = True
            ack_bits.append(
                "Cancellation requested. The running agent will stop on its next iteration.",
            )
        if cleared_gates:
            label = " / ".join(cleared_gates)
            ack_bits.append(f"已取消等待中的 {label} 参数确认。")
        if not ack_bits:
            ack_bits.append("No active computation is running right now.")
        ack_text = " ".join(ack_bits)
        msg = session.add_message("assistant", ack_text, agent="router")
        await store.update(session)
        # Push the ack so the chat panel shows it immediately, exactly the
        # same shape AgentBase.say() emits.
        await bus.publish(
            AgentEvent(
                kind="message",
                session_id=session.session_id,
                agent="router",
                payload={"message": safe_json(msg)},
            )
        )
        _schedule_summary_refresh()
        return {"routed_to": "cancel", "result": {"ok": True}}

    # ---- LLM-router slot filling (clarification) ------------------------
    # When the smart router judges the request ambiguous or partial, it
    # supplies a follow-up question instead of a tool decision. Show the
    # question, do not dispatch a worker, wait for the user's next reply.
    if (
        decision.get("needs_clarification")
        and decision.get("clarification")
        and next_agent != "cancel"
    ):
        await _router_say(session, str(decision["clarification"]))
        return {"routed_to": "router", "result": {"awaiting": "clarification"}}

    pending_design = bool(session.context.get("pending_design_confirmation"))
    design_confirmed = False
    if pending_design:
        pending_design_params = _extract_designer_params(user_text)
        pending_eoh_params = _extract_eoh_params(user_text)
        if _DESIGN_CANCEL_RE.search(user_text):
            session.context.pop("pending_design_confirmation", None)
            await store.update(session)
            await _router_say(session, "已取消本次 CDDI 设计。你可以继续聊天或重新点击 Design。")
            _schedule_summary_refresh()
            return {"routed_to": "designer", "result": {"passed": False, "cancelled": True}}
        if _DESIGN_CONFIRM_RE.search(user_text):
            # Natural confirmation sentences can also carry lightweight /
            # EOH knobs ("开始设计 lightweight repair_attempts=6 sandbox=5").
            # Persist them before replacing the router decision with the
            # synthetic confirmation decision.
            dparams = _extract_designer_params(user_text)
            if dparams:
                existing_d = dict(session.context.get("designer_user_params") or {})
                existing_d.update(dparams)
                session.context["designer_user_params"] = existing_d
            eparams = _extract_eoh_params(user_text)
            if eparams:
                existing_e = dict(session.context.get("eoh_user_params") or {})
                existing_e.update(eparams)
                session.context["eoh_user_params"] = existing_e
            session.context.pop("pending_design_confirmation", None)
            await store.update(session)
            next_agent = "designer"
            decision = {"intent": "design_mmd", "next_agent": "designer", "source": "design_confirmation"}
            design_confirmed = True
        elif next_agent == "designer" or pending_design_params or pending_eoh_params:
            # User is still adjusting CDDI/EOH params; persist parsed params
            # below, then show the updated prompt instead of running.
            next_agent = "designer"
            decision = {
                **decision,
                "intent": "design_mmd",
                "next_agent": "designer",
            }
            if pending_design_params:
                decision["designer_params"] = {
                    **dict(decision.get("designer_params") or {}),
                    **pending_design_params,
                }
            if pending_eoh_params:
                decision["eoh_params"] = {
                    **dict(decision.get("eoh_params") or {}),
                    **pending_eoh_params,
                }
            pass
        else:
            await _router_say(
                session,
                "我已准备好设计 CDDI，但还在等待参数确认。"
                "如果用当前参数，请回复 `默认`；如果要改参数，请直接说明。"
            )
            return {"routed_to": "designer", "result": {"awaiting": "design_parameter_confirmation"}}

    if next_agent == "refine":
        # Accept either a multi-cluster list (`selected_clusters`) or the
        # legacy single-cluster id; keep both fields in sync on the
        # session so downstream code can read whichever it prefers.
        if "selected_clusters" in decision:
            ids = [int(c) for c in decision["selected_clusters"] if c is not None]
            if ids:
                session.selected_clusters = sorted(set(ids))
                session.selected_cluster = session.selected_clusters[0]
                await store.update(session)
        elif "selected_cluster" in decision:
            cid = int(decision["selected_cluster"])
            session.selected_cluster = cid
            session.selected_clusters = [cid]
            await store.update(session)

    # ---- cluster confirmation gate -------------------------------------
    # First the "we're already waiting on you" branch, then the
    # "you just asked to cluster — let's confirm params" branch. Mirrors
    # the CDDI design gate (same vocab, same UX). Bypassed inside plan
    # execution because the user already confirmed the plan up front.
    pending_cluster = bool(session.context.get("pending_cluster_confirmation"))
    cluster_confirmed = False
    if pending_cluster:
        pending_cluster_params = _extract_cluster_params(user_text)
        if _CLUSTER_CANCEL_RE.search(user_text):
            session.context.pop("pending_cluster_confirmation", None)
            await store.update(session)
            await _router_say(session, "已取消本次聚类。你可以继续聊天或重新触发 Cluster。")
            _schedule_summary_refresh()
            return {"routed_to": "cluster", "result": {"passed": False, "cancelled": True}}
        if _CLUSTER_CONFIRM_RE.search(user_text):
            # Natural confirmation sentences can also carry knobs:
            # "开始聚类 k=4 OCV=1 IR=1 CDDI=2".
            if pending_cluster_params:
                existing = dict(session.context.get("cluster_user_params") or {})
                session.context["cluster_user_params"] = _merge_cluster_params(
                    existing,
                    pending_cluster_params,
                )
            session.context.pop("pending_cluster_confirmation", None)
            await store.update(session)
            next_agent = "cluster"
            decision = {"intent": "cluster", "next_agent": "cluster",
                        "source": "cluster_confirmation"}
            # Pre-populate target_k from prior gate turns so the
            # context-clear branch below doesn't wipe it out. cluster_params
            # are read straight from session.context (sticky), so no need
            # to copy them onto the decision.
            prior_tk = session.context.get("target_k")
            if prior_tk is not None:
                try:
                    decision["target_k"] = int(prior_tk)
                except (TypeError, ValueError):
                    pass
            cluster_confirmed = True
        elif next_agent == "cluster" or pending_cluster_params:
            # User is still tuning cluster params; persist them below,
            # then re-show the updated prompt instead of running.
            next_agent = "cluster"
            decision = {
                **decision,
                "intent": "cluster",
                "next_agent": "cluster",
            }
            if pending_cluster_params:
                decision["cluster_params"] = {
                    **dict(decision.get("cluster_params") or {}),
                    **pending_cluster_params,
                }
        else:
            await _router_say(
                session,
                "我已准备好启动聚类，但还在等待参数确认。"
                "如果用当前参数，请回复 `默认`；如果要改参数，请直接说明。",
            )
            return {"routed_to": "cluster", "result": {"awaiting": "cluster_parameter_confirmation"}}

    # Record the last explicit K choice for audit/UI only. ClusterAgent no
    # longer reads this implicitly, otherwise "Auto" would silently reuse the
    # previous fixed K.
    if next_agent == "cluster":
        if "target_k" in decision:
            session.context["target_k"] = int(decision["target_k"])
        else:
            session.context.pop("target_k", None)
        # Conversational GA / metric tuning is sticky across runs in the
        # same session — say "use variance feedback" once, then "Auto"
        # later still uses the variance criterion.
        if "cluster_params" in decision:
            existing = dict(session.context.get("cluster_user_params") or {})
            session.context["cluster_user_params"] = _merge_cluster_params(
                existing,
                decision["cluster_params"],
            )
        await store.update(session)

    # Show the cluster prompt once params + target_k from this turn have
    # been persisted, so the message reflects the user's just-spoken
    # tweaks instead of stale defaults.
    if next_agent == "cluster" and not cluster_confirmed:
        session.context["pending_cluster_confirmation"] = True
        await store.update(session)
        await _router_say(session, _cluster_parameter_prompt(session))
        return {"routed_to": "cluster", "result": {"awaiting": "cluster_parameter_confirmation"}}

    # Persist conversational lightweight Designer params (repair attempts /
    # sandbox validation cells). Sticky across runs so the user can say
    # "轻量化 sandbox=5" once, then "开始设计" later.
    if next_agent == "designer" and decision.get("proposed_mode") in {
        LIGHTWEIGHT_MODE,
        REFLECTIVE_SEARCH_MODE,
        LEGACY_REFLECTIVE_SEARCH_MODE,
    }:
        session.mode = normalize_mode(decision["proposed_mode"])
        await store.update(session)

    if next_agent == "designer" and "designer_params" in decision:
        params = {k: int(v) for k, v in decision["designer_params"].items()}
        existing = dict(session.context.get("designer_user_params") or {})
        existing.update(params)
        session.context["designer_user_params"] = existing
        await store.update(session)

    # Persist conversational EOH-param tuning. DesignerAgent picks these up
    # on its next Reflective Search run (max_fe / pop_size / batch_size).
    # Sticky on
    # purpose so the user can say "EOH 5 代" once then just "go" later.
    if next_agent == "designer" and "eoh_params" in decision:
        params = {k: int(v) for k, v in decision["eoh_params"].items()}
        existing = dict(session.context.get("eoh_user_params") or {})
        existing.update(params)
        session.context["eoh_user_params"] = existing
        await store.update(session)

    if next_agent == "designer" and not design_confirmed:
        session.context["pending_design_confirmation"] = True
        await store.update(session)
        await _router_say(session, _design_parameter_prompt(session))
        return {"routed_to": "designer", "result": {"awaiting": "design_parameter_confirmation"}}

    agent_cls = _AGENT_REGISTRY.get(next_agent)
    if agent_cls is None:
        chat_agent = ChatAgent(session)
        await chat_agent.run(
            f"The {next_agent} agent is not yet available. "
            "Please continue chatting; integration is in progress."
        )
        return {"routed_to": next_agent, "fallback": "chat"}

    agent = agent_cls(session)
    # Forward optional decision-level kwargs (e.g. target_k for ClusterAgent,
    # cluster_ids for RefinementAgent).
    extra_kwargs: Dict[str, Any] = {}
    if next_agent == "cluster":
        if "target_k" in decision:
            extra_kwargs["target_k"] = int(decision["target_k"])
        # Always forward the sticky cluster params (may be empty dict).
        cparams = session.context.get("cluster_user_params") or {}
        if cparams:
            extra_kwargs["cluster_params"] = dict(cparams)
    if next_agent == "designer":
        dparams = session.context.get("designer_user_params") or {}
        if dparams:
            if "repair_attempts" in dparams:
                extra_kwargs["max_attempts"] = int(dparams["repair_attempts"])
            if "sandbox_cells" in dparams:
                extra_kwargs["sandbox_cells"] = int(dparams["sandbox_cells"])
    if next_agent == "refine":
        if "selected_clusters" in decision:
            ids = [int(c) for c in decision["selected_clusters"] if c is not None]
            if ids:
                extra_kwargs["cluster_ids"] = sorted(set(ids))
        elif "selected_cluster" in decision:
            extra_kwargs["cluster_ids"] = [int(decision["selected_cluster"])]
    result = await agent.run(user_text=user_text, **extra_kwargs)

    # Skip the summary LLM call for chat-only turns — chat doesn't add
    # durable pipeline state worth re-summarising, and the user just
    # paid for a chat LLM call already. Refresh only when a real worker
    # advanced the state.
    if next_agent != "chat":
        _schedule_summary_refresh()
    return {"routed_to": next_agent, "result": result}


# ===========================================================================
# Workflow plan execution
#
# A plan is a list of agent dispatches stored on session.context["active_plan"]
# (see services/planner.py for the shape). The pipeline handles:
#
#   1. plan_workflow tool from the LLM router → generate + present a plan.
#   2. plan-control phrases (confirm / skip / cancel) — either via fast-path
#      regex or via the LLM router's advance_plan tool.
#   3. step-by-step execution with a chat prompt between every step so the
#      user can intervene.
#
# Plans run "inline" — the same agent registry that handle_message uses, but
# we bypass the design-confirmation gate because the user already confirmed
# the whole plan up front.
# ===========================================================================


def _plan_snapshot(session: Session) -> Dict[str, Any]:
    """Lightweight context block we give the planner LLM."""
    return {
        "mode": session.mode,
        "active_dataset_id": session.active_dataset_id,
        "selected_clusters": list(session.selected_clusters or []),
        "pipeline_state": dict(session.pipeline_state),
        "sticky_eoh_params": session.context.get("eoh_user_params"),
        "sticky_cluster_params": session.context.get("cluster_user_params"),
    }


def _step_kwargs(step: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a PlanStep's params into agent.run() kwargs."""
    agent = step.get("agent")
    params = step.get("params") or {}
    kwargs: Dict[str, Any] = {}
    if agent == "cluster":
        if "target_k" in params:
            try:
                kwargs["target_k"] = int(params["target_k"])
            except (TypeError, ValueError):
                pass
        cparams = {
            k: v for k, v in params.items()
            if k in ("min_k", "max_k", "pop_size", "max_iter", "feedback_metric")
        }
        weights = {}
        for src, dst in (
            ("ocv_weight", "ocv"),
            ("ir_weight", "ir"),
            ("ddia_weight", "ddia"),
            ("cddi_weight", "ddia"),
        ):
            if src in params:
                try:
                    weights[dst] = float(params[src])
                except (TypeError, ValueError):
                    pass
        if weights:
            cparams["feature_weights"] = weights
        if cparams:
            kwargs["cluster_params"] = cparams
    elif agent == "refine":
        ids = params.get("cluster_ids")
        if isinstance(ids, (list, tuple)):
            try:
                kwargs["cluster_ids"] = sorted({int(x) for x in ids})
            except (TypeError, ValueError):
                pass
    return kwargs


def _apply_step_side_effects(session: Session, step: Dict[str, Any]) -> None:
    """Mutate session state for params Designer / Compute read from context.

    Designer reads ``session.mode`` and ``session.context['eoh_user_params']``
    directly. We honour those overrides here so the user doesn't have to
    flip the Mode toggle for a plan that uses Reflective Search on one step and
    Lightweight on another.
    """
    if step.get("agent") != "designer":
        return
    params = step.get("params") or {}
    mode = params.get("mode")
    if mode in (
        LIGHTWEIGHT_MODE,
        REFLECTIVE_SEARCH_MODE,
        LEGACY_REFLECTIVE_SEARCH_MODE,
    ):
        session.mode = normalize_mode(mode)
    eoh_overrides = {
        k: int(params[k]) for k in ("max_fe", "pop_size", "batch_size")
        if isinstance(params.get(k), int)
    }
    if eoh_overrides:
        existing = dict(session.context.get("eoh_user_params") or {})
        existing.update(eoh_overrides)
        session.context["eoh_user_params"] = existing


async def _narrate(
    session: Session,
    *,
    agent_id: str,
    result: Any,
    prompt_hint: Optional[str] = None,
) -> Optional[str]:
    """Run the narrator service in a worker thread (the LLM client is sync).

    Returns narration text only; callers decide whether / how to merge it
    into a user-visible message. Do not post a separate router message here:
    splitting "agent result" and "next-step guidance" into two assistant
    bubbles made the conversation feel duplicated and disjointed.
    """
    try:
        text = await asyncio.to_thread(
            narrator.narrate,
            agent_id=agent_id,
            result=result,
            session=session,
            prompt_hint=prompt_hint,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("narrator dispatch failed: %s", exc)
        return None
    if not text:
        return None
    return text


async def _emit_plan(session: Session, kind: str = "plan_updated") -> None:
    plan = session.context.get("active_plan")
    if not plan:
        return
    await bus.publish(
        AgentEvent(
            kind=kind,
            session_id=session.session_id,
            agent="planner",
            payload={"plan": safe_json(plan)},
        )
    )


async def _present_plan(
    session: Session,
    plan: planner.WorkflowPlan,
    *,
    header: str,
) -> None:
    """Persist a freshly-created or freshly-updated plan and notify chat + UI."""
    session.context["active_plan"] = plan.to_dict()
    await store.update(session)
    await _emit_plan(session, kind="plan_created")
    await _router_say(
        session,
        planner.render_plan_markdown(plan, header=header)
        + "\n\n回复 `开始` / `继续` 启动第 1 步；`跳过` 跳过当前步；`取消计划` 中止。",
    )


async def _create_plan_from_decision(
    session: Session, decision: Dict[str, Any], user_text: str
) -> Dict[str, Any]:
    """Handle the LLM router's ``plan_workflow`` tool: build + present a plan."""
    goal = decision.get("plan_goal") or user_text
    snapshot = _plan_snapshot(session)

    # The planner LLM call is sync — offload so we don't stall the loop.
    plan_obj = await asyncio.to_thread(
        planner.generate_plan, user_goal=goal, session_snapshot=snapshot
    )
    if plan_obj is None:
        plan_obj = planner.default_plan(goal)
        header = (
            "我没能从大模型拿到结构化计划，先按默认 3 步走："
        )
    else:
        header = "我打算按下面这个计划执行："

    # Apply optional router-provided hints (target_k, cluster_ids, mode) to
    # the corresponding steps, so a user-stated knob isn't lost when the
    # planner LLM forgot to copy it through.
    target_k = decision.get("plan_hint_target_k")
    if isinstance(target_k, int):
        for s in plan_obj.steps:
            if s.agent == "cluster" and "target_k" not in s.params:
                s.params["target_k"] = int(target_k)
    cluster_ids = decision.get("plan_hint_cluster_ids")
    if isinstance(cluster_ids, list) and cluster_ids:
        for s in plan_obj.steps:
            if s.agent == "refine" and not s.params.get("cluster_ids"):
                s.params["cluster_ids"] = list(cluster_ids)
    mode_hint = decision.get("plan_hint_mode")
    if mode_hint in (
        LIGHTWEIGHT_MODE,
        REFLECTIVE_SEARCH_MODE,
        LEGACY_REFLECTIVE_SEARCH_MODE,
    ):
        for s in plan_obj.steps:
            if s.agent == "designer" and "mode" not in s.params:
                s.params["mode"] = normalize_mode(mode_hint)

    await _present_plan(session, plan_obj, header=header)
    return {"routed_to": "planner", "result": {"plan": plan_obj.to_dict()}}


async def _run_current_plan_step(session: Session) -> Dict[str, Any]:
    """Dispatch the plan's current step and advance the index.

    Caller has already validated the plan is active. Posts a follow-up
    chat message inviting the user to confirm the next step (or signalling
    completion) and broadcasts a ``plan_updated`` WS event.
    """
    plan_dict = session.context.get("active_plan")
    if not plan_dict:
        return {"error": "no active plan"}
    plan = planner.WorkflowPlan.from_dict(plan_dict)
    step = plan.current_step()
    if step is None:
        plan.status = "done"
        session.context["active_plan"] = plan.to_dict()
        await store.update(session)
        await _emit_plan(session, kind="plan_updated")
        await _router_say(session, "✅ 计划已完成。需要下一步操作请随时告诉我。")
        return {"plan_status": "done"}

    plan.status = "in_progress"
    step.status = "running"
    session.context["active_plan"] = plan.to_dict()
    await store.update(session)
    await _emit_plan(session, kind="plan_updated")

    agent_cls = _AGENT_REGISTRY.get(step.agent)
    if agent_cls is None:
        step.status = "error"
        plan.steps[plan.current_idx] = step
        plan.current_idx += 1
        if plan.is_complete():
            plan.status = "done"
        session.context["active_plan"] = plan.to_dict()
        await store.update(session)
        await _emit_plan(session, kind="plan_updated")
        await _router_say(session, f"❌ 步骤 {step.id + 1} 失败：未知 agent `{step.agent}`。")
        return {"error": "unknown agent", "agent": step.agent}

    _apply_step_side_effects(session, step.to_dict())
    await store.update(session)

    agent = agent_cls(session)
    try:
        result = await agent.run(user_text="", **_step_kwargs(step.to_dict()))
        step.status = "done"
    except Exception as exc:  # noqa: BLE001
        logger.exception("plan step failed")
        step.status = "error"
        result = {"error": str(exc)}

    plan.steps[plan.current_idx] = step
    plan.current_idx += 1
    if plan.is_complete():
        plan.status = "done"
    session.context["active_plan"] = plan.to_dict()
    await store.update(session)
    await _emit_plan(session, kind="plan_updated")

    # Tail message — try the contextual narrator first; if it's disabled
    # or fails, fall back to the deterministic template so the user is
    # never left hanging without a next-step prompt.
    if plan.status == "done":
        hint = (
            f"The user just finished the final step of a {len(plan.steps)}-step "
            f"workflow plan ('{plan.goal}'). Celebrate briefly and invite them "
            "to inspect artifacts, refine a cluster, or start something new."
        )
        narration = await _narrate(
            session, agent_id=step.agent, result=result, prompt_hint=hint,
        )
        await _router_say(
            session,
            ((narration + "\n\n") if narration else "✅ 计划全部完成！\n\n")
            + planner.render_plan_markdown(plan, header="最终状态："),
        )
    elif step.status == "error":
        # Errors stay templated — narrating a failure feels off, and the
        # error string itself is what the user actually needs to see.
        await _router_say(
            session,
            f"❌ 第 {step.id + 1} 步出错了：{result.get('error', 'unknown error')}\n"
            "回复 `继续` 跳过这步并尝试下一步，或 `取消计划` 中止。",
        )
    else:
        nxt = plan.current_step()
        if nxt is not None:
            nxt_label = planner._AGENT_DISPLAY.get(nxt.agent, nxt.agent)
            hint = (
                f"The user just finished step {step.id + 1} of {len(plan.steps)} "
                f"in plan '{plan.goal}'. The next step is "
                f"**{nxt.id + 1}. {nxt_label}** — {nxt.description}. "
                "Invite them to confirm with `继续` (or `跳过` / `取消计划`)."
            )
            narration = await _narrate(
                session, agent_id=step.agent, result=result, prompt_hint=hint,
            )
            # One combined follow-up bubble. If narration is available, use it
            # as the opening sentence, then still append deterministic controls
            # so the user always knows exactly what to type.
            await _router_say(
                session,
                ((narration + "\n\n") if narration else f"✅ 步骤 {step.id + 1} 完成。\n\n")
                + f"下一步：**{nxt.id + 1}. {nxt_label}** — {nxt.description}\n"
                "回复 `继续` 执行；`跳过` 跳过；`取消计划` 中止。",
            )
    return {"plan_status": plan.status, "step_result": result}


async def _handle_plan_action(
    session: Session, action: str
) -> Dict[str, Any]:
    """Apply a confirm / next / skip / cancel action to the active plan."""
    plan_dict = session.context.get("active_plan")
    if not plan_dict:
        await _router_say(session, "目前没有正在进行的计划。")
        return {"routed_to": "planner", "result": {"error": "no active plan"}}

    plan = planner.WorkflowPlan.from_dict(plan_dict)

    if action == "cancel":
        plan.status = "cancelled"
        session.context["active_plan"] = plan.to_dict()
        await store.update(session)
        await _emit_plan(session, kind="plan_updated")
        await _router_say(session, "已取消当前计划。你可以继续聊天或重新告诉我目标。")
        return {"routed_to": "planner", "result": {"plan_status": "cancelled"}}

    if action == "skip":
        step = plan.current_step()
        if step is None:
            plan.status = "done"
            session.context["active_plan"] = plan.to_dict()
            await store.update(session)
            await _emit_plan(session, kind="plan_updated")
            await _router_say(session, "计划已经走完，没有可跳过的步骤。")
            return {"routed_to": "planner", "result": {"plan_status": "done"}}
        step.status = "skipped"
        plan.steps[plan.current_idx] = step
        plan.current_idx += 1
        if plan.is_complete():
            plan.status = "done"
            session.context["active_plan"] = plan.to_dict()
            await store.update(session)
            await _emit_plan(session, kind="plan_updated")
            await _router_say(
                session,
                f"↷ 已跳过步骤 {step.id + 1} ({step.description})。计划已结束。",
            )
            return {"routed_to": "planner", "result": {"plan_status": "done"}}
        session.context["active_plan"] = plan.to_dict()
        await store.update(session)
        await _emit_plan(session, kind="plan_updated")
        nxt = plan.current_step()
        nxt_label = planner._AGENT_DISPLAY.get(nxt.agent, nxt.agent) if nxt else "?"
        nxt_desc = nxt.description if nxt else ""
        await _router_say(
            session,
            f"↷ 已跳过步骤 {step.id + 1}。下一步：**{(nxt.id + 1) if nxt else '?'}. "
            f"{nxt_label}** — {nxt_desc}\n"
            "回复 `继续` 执行；`跳过` 再跳过一步；`取消计划` 中止。",
        )
        return {"routed_to": "planner", "result": {"plan_status": plan.status}}

    # confirm or next ⇒ run the current step.
    return {"routed_to": "planner", **await _run_current_plan_step(session)}


def _classify_plan_control(user_text: str) -> Optional[str]:
    """Map a short fast-path message to a plan action (or None)."""
    if _PLAN_CANCEL_RE.search(user_text):
        return "cancel"
    if _PLAN_SKIP_RE.search(user_text):
        return "skip"
    if _PLAN_CONFIRM_RE.search(user_text):
        return "confirm"
    return None


def _coerce_cluster_list(action: Dict[str, Any]) -> list[int]:
    """Pull a normalised List[int] of cluster ids out of an action payload.

    The frontend may send either ``clusters: [0,3,5]`` (new multi-select
    UI) or ``cluster: 3`` (legacy single-cluster shortcut button on each
    decision card). We accept both and de-dupe.
    """
    raw = action.get("clusters")
    if raw is None:
        single = action.get("cluster")
        raw = [single] if single is not None else []
    cleaned: list[int] = []
    for v in raw if isinstance(raw, (list, tuple)) else [raw]:
        try:
            cleaned.append(int(v))
        except (TypeError, ValueError):
            continue
    return sorted(set(cleaned))


async def handle_action(session: Session, action: Dict[str, Any]) -> Dict[str, Any]:
    kind = action.get("type")
    if kind == "select_cluster":
        # Legacy single-cluster path (also accepts the new `clusters` list
        # for forward-compat in case the frontend stops sending the old
        # action name).
        clusters = _coerce_cluster_list(action)
        if clusters:
            session.selected_clusters = clusters
            session.selected_cluster = clusters[0]
        else:
            session.selected_cluster = int(action.get("cluster", 0))
            session.selected_clusters = [session.selected_cluster]
        await store.update(session)
        return {
            "ok": True,
            "selected_cluster": session.selected_cluster,
            "selected_clusters": session.selected_clusters,
        }
    if kind == "select_clusters":
        # New multi-select action: persists `selected_clusters` without
        # immediately running a refinement (the UI uses this to keep the
        # tickbox state synced to the backend).
        clusters = _coerce_cluster_list(action)
        session.selected_clusters = clusters
        session.selected_cluster = clusters[0] if clusters else None
        await store.update(session)
        return {"ok": True, "selected_clusters": clusters}
    if kind == "toggle_mode":
        mode = action.get("mode", LIGHTWEIGHT_MODE)
        if mode in {
            LIGHTWEIGHT_MODE,
            REFLECTIVE_SEARCH_MODE,
            LEGACY_REFLECTIVE_SEARCH_MODE,
        }:
            session.mode = normalize_mode(mode)
            await store.update(session)
            return {"ok": True, "mode": session.mode}
    if kind == "set_dataset":
        ds = action.get("dataset_id")
        if ds:
            session.active_dataset_id = ds
            await store.update(session)
            return {"ok": True, "active_dataset_id": ds}
    if kind == "confirm_all200":
        session.context["all200_ready"] = True
        if action.get("dataset_id"):
            session.context["refine_dataset_id"] = action["dataset_id"]
        await store.update(session)
        # If there is a pending refinement waiting on this confirmation,
        # resume it — preferring the multi-cluster pending payload if
        # present, otherwise the legacy single one.
        pending_multi = session.context.get("pending_refine_clusters")
        pending_single = session.context.get("pending_refine_cluster")
        if pending_multi:
            agent = RefinementAgent(session)
            session.context.pop("pending_refine_clusters", None)
            session.context.pop("pending_refine_cluster", None)
            session.context.pop("pending_refine_cells", None)
            await store.update(session)
            result = await agent.run(cluster_ids=list(pending_multi))
            return {"ok": True, "refinement": result}
        if pending_single is not None:
            agent = RefinementAgent(session)
            session.context.pop("pending_refine_cluster", None)
            session.context.pop("pending_refine_cells", None)
            await store.update(session)
            result = await agent.run(user_text=f"refine cluster {pending_single}")
            return {"ok": True, "refinement": result}
        return {"ok": True}
    if kind == "refine_cluster":
        # Legacy single-cluster Refine →  button on a card.
        cluster = int(action.get("cluster", 0))
        session.selected_cluster = cluster
        session.selected_clusters = [cluster]
        await store.update(session)
        agent = RefinementAgent(session)
        result = await agent.run(user_text=f"refine cluster {cluster}")
        return {"ok": True, "refinement": result}
    if kind == "refine_clusters":
        # New multi-cluster path. Same internal RefinementAgent — passes
        # the user's tickbox selection as cluster_ids and lets the agent
        # union their cells before running the All-200 sub-pipeline.
        clusters = _coerce_cluster_list(action)
        if not clusters:
            return {"ok": False, "error": "refine_clusters requires non-empty cluster list"}
        session.selected_clusters = clusters
        session.selected_cluster = clusters[0]
        await store.update(session)
        agent = RefinementAgent(session)
        result = await agent.run(cluster_ids=clusters)
        return {"ok": True, "refinement": result}
    if kind in {"cancel_reflective_search", "cancel_eoh"}:
        session.context["cancel_reflective_search"] = True
        # Older workers and persisted sessions use this key.
        session.context["cancel_eoh"] = True
        await store.update(session)
        return {"ok": True, "cancel": "requested"}
    if kind == "plan_action":
        # UI button — Run Next Step / Skip / Cancel Plan in PlanPanel.tsx.
        # The 4 actions match _handle_plan_action exactly.
        plan_action = action.get("action")
        if plan_action not in ("confirm", "next", "skip", "cancel"):
            return {"ok": False, "error": f"invalid plan_action: {plan_action!r}"}
        result = await _handle_plan_action(session, plan_action)
        return {"ok": True, **result}
    if kind == "plan_update_step":
        # PlanPanel form save → patch params on a single pending step.
        # Only pending steps can be edited (running/done/skipped/cancelled
        # are immutable). Params are clamped to the schema in
        # services/planner.validate_step_params so the user can't smuggle
        # a max_fe=999 through the UI.
        plan = session.context.get("active_plan")
        if not isinstance(plan, dict):
            return {"ok": False, "error": "no active plan"}
        try:
            step_id = int(action.get("step_id"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "missing or invalid step_id"}
        new_params = action.get("params")
        if not isinstance(new_params, dict):
            return {"ok": False, "error": "params must be an object"}
        steps = plan.get("steps") or []
        target_idx = next(
            (i for i, s in enumerate(steps) if int(s.get("id", -1)) == step_id),
            None,
        )
        if target_idx is None:
            return {"ok": False, "error": f"step {step_id} not found"}
        step = steps[target_idx]
        if step.get("status") != "pending":
            return {"ok": False,
                    "error": f"only pending steps can be edited (got status={step.get('status')!r})"}
        # Merge: keep existing keys that the user didn't touch, override
        # with validated new values. Pass an explicit ``None`` in the
        # payload to clear a single key (validator drops Nones).
        merged: Dict[str, Any] = dict(step.get("params") or {})
        for k, v in new_params.items():
            if v is None or v == "":
                merged.pop(k, None)
            else:
                merged[k] = v
        cleaned = planner.validate_step_params(step.get("agent", ""), merged)
        step["params"] = cleaned
        steps[target_idx] = step
        plan["steps"] = steps
        session.context["active_plan"] = plan
        await store.update(session)
        await _emit_plan(session, kind="plan_updated")
        return {"ok": True, "step_id": step_id, "params": cleaned}
    return {"ok": False, "error": f"Unknown action: {kind}"}
