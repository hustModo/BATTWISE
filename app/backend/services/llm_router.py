"""LLM-based intent router with structured tool-call output.

Replaces the regex-only heuristic that lived in router_agent.py. The user's
complaint was that the old approach felt stiff — paraphrases that didn't
hit a pattern fell through to a wrong intent or a generic ChatAgent reply.

Design
------
* Each "thing the user might ask for" is defined as a Tool with a JSON
  parameter schema (same shape as OpenAI / DashScope function calling).
* Routing uses a small/fast model (qwen3.5-flash) so the call stays
  sub-2s on the critical path of every user message.
* The router LLM sees:
    - the user's latest message
    - a compact session snapshot (mode / dataset / pipeline state /
      selected_clusters / sticky params / pending flags)
    - a running conversation summary (built by conversation_memory)
    - the available tools
* It emits JSON with shape::

    {"tool": "...", "params": {...}, "confidence": 0-1,
     "needs_clarification": bool, "clarification": "...",
     "reasoning": "..."}

* If the LLM is unreachable or confidence is low → caller falls back to
  the legacy regex heuristic so the system never hangs on a network
  hiccup.

Slot filling
------------
When the user says something ambiguous ("再来一次", "redesign with what we
had"), the model is asked to set ``needs_clarification=true`` and supply a
short follow-up question in the user's language. The pipeline shows that
question and waits for the user's reply instead of dispatching a worker.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from app.backend.services import llm_client
from app.backend.naming import (
    LEGACY_REFLECTIVE_SEARCH_MODE,
    LIGHTWEIGHT_MODE,
    REFLECTIVE_SEARCH_MODE,
    normalize_mode,
)
from app.backend.settings import settings

logger = logging.getLogger(__name__)


# Model used for routing decisions. qwen3.5-flash trades a touch of
# accuracy for ~3-5× speed vs qwen3.6-plus (the main / pinned model).
# Routing must stay snappy because it sits on the critical path of every
# user message — even sub-second matters here.
ROUTER_MODEL = "qwen3.5-flash"

# Decision confidence below this threshold ⇒ fall back to legacy regex.
# 0.55 picked empirically: above this the model is choosing a tool, below
# it is hedging between two.
MIN_CONFIDENCE = 0.55


# Tool name → downstream agent name expected by orchestrator/pipeline.py.
# Keep this map in sync with _AGENT_REGISTRY there. "cancel" is a virtual
# agent handled inline by the pipeline.
TOOL_TO_AGENT: Dict[str, str] = {
    "design_cddi": "designer",
    "design_ddia": "designer",
    "compute_cddi": "compute",
    "compute_mmd": "compute",
    "run_clustering": "cluster",
    "select_clusters": "refine",
    "refine_clusters": "refine",
    "cancel_running": "cancel",
    "answer_question": "chat",
    "start_pipeline": "designer",
    # Multi-step planner — pipeline handles this inline, not via a worker
    # agent. ``next_agent`` is purely informational; the pipeline branches
    # on the tool name first.
    "plan_workflow": "planner",
    "advance_plan": "planner",
}

# Tool → legacy `intent` string, preserved so existing pipeline branches
# (target_k handling, selected_clusters sync, design-confirmation gate)
# keep working unchanged. Frontend also keys on intent for the timeline.
TOOL_TO_INTENT: Dict[str, str] = {
    "design_cddi": "design_mmd",
    "design_ddia": "design_mmd",
    "compute_cddi": "compute_mmd",
    "compute_mmd": "compute_mmd",
    "run_clustering": "cluster",
    "select_clusters": "select_cluster",
    "refine_clusters": "refine_all200",
    "cancel_running": "cancel",
    "answer_question": "chitchat",
    "start_pipeline": "start",
    "plan_workflow": "plan_workflow",
    "advance_plan": "advance_plan",
}


# Tool schema. Parameters are kept minimal — only the knobs the user
# actually controls in conversation. Anything else stays on agent-side
# defaults so the LLM has fewer chances to hallucinate values.
AGENT_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "design_cddi",
        "description": (
            "Design a new CDDI function "
            "to characterise battery cell curves. Use when the user wants to "
            "regenerate the CDDI function, run Reflective Search, or kick "
            "off algorithm design."
        ),
        "parameters": {
            "mode": {
                "enum": ["lightweight", "reflective_search"],
                "optional": True,
                "description": "Lightweight = single LLM design pass. Reflective Search = performance-guided program search.",
            },
            "max_fe": {"type": "int", "min": 1, "max": 30, "optional": True,
                       "description": "Number of generations for Reflective Search."},
            "pop_size": {"type": "int", "min": 2, "max": 32, "optional": True,
                         "description": "Individuals per generation for Reflective Search."},
            "batch_size": {"type": "int", "min": 1, "max": 32, "optional": True,
                           "description": "Batch size for Reflective Search evaluation."},
            "repair_attempts": {"type": "int", "min": 1, "max": 12, "optional": True,
                                "description": "Lightweight CDDI repair attempts."},
            "sandbox_cells": {"type": "int", "min": 1, "max": 20, "optional": True,
                              "description": "Lightweight CDDI sandbox validation cells."},
        },
    },
    {
        "name": "compute_cddi",
        "description": "Run the latest CDDI function across all cell files. No parameters needed.",
        "parameters": {},
    },
    {
        "name": "run_clustering",
        "description": (
            "Run adaptive GA-KMeans clustering on the OCV + IR + CDDI. "
            "Optionally pin the cluster count k or change GA / feedback knobs."
        ),
        "parameters": {
            "target_k": {"type": "int", "min": 2, "max": 12, "optional": True,
                         "description": "If user specifies a fixed cluster count, pin it here."},
            "min_k": {"type": "int", "min": 2, "max": 12, "optional": True},
            "max_k": {"type": "int", "min": 2, "max": 12, "optional": True},
            "pop_size": {"type": "int", "min": 20, "max": 500, "optional": True,
                         "description": "GA population size for the clustering search."},
            "max_iter": {"type": "int", "min": 10, "max": 500, "optional": True,
                         "description": "GA max iterations."},
            "feedback_metric": {"enum": ["combined", "dbi", "variance"], "optional": True,
                                "description": "GA fitness criterion."},
            "ocv_weight": {"type": "number", "min": 0.05, "max": 20, "optional": True,
                           "description": "Relative weight for OCV after normalization; default 1."},
            "ir_weight": {"type": "number", "min": 0.05, "max": 20, "optional": True,
                          "description": "Relative weight for internal resistance after normalization; default 1."},
            "cddi_weight": {"type": "number", "min": 0.05, "max": 20, "optional": True,
                            "description": "Relative weight for CDDI loss after normalization; default 1."},
        },
    },
    {
        "name": "select_clusters",
        "description": (
            "User indicates which cluster(s) they're interested in — no refinement yet, "
            "just records the selection."
        ),
        "parameters": {
            "cluster_ids": {"type": "int_list",
                            "description": "Cluster ids the user named, e.g. [0, 3, 5]."},
        },
    },
    {
        "name": "refine_clusters",
        "description": (
            "Re-cluster a previously chosen subset of cells using the All-200 dataset. "
            "Use when the user mentions 'refine', 'All-200', 'second stage', or names "
            "clusters to drill into."
        ),
        "parameters": {
            "cluster_ids": {"type": "int_list", "optional": True,
                            "description": "Clusters to refine. If omitted, use session.selected_clusters."},
        },
    },
    {
        "name": "cancel_running",
        "description": (
            "Stop the agent currently running (Designer / Compute / Cluster). "
            "Use for any 'stop' / 'cancel' / 'enough' / '停' / '算了' style message."
        ),
        "parameters": {},
    },
    {
        "name": "answer_question",
        "description": (
            "User is chatting or asking about the system / results / what to do next. "
            "No pipeline action — ChatAgent composes a reply."
        ),
        "parameters": {
            "topic": {"type": "string", "optional": True,
                      "description": "Short tag for what they're asking about."},
        },
    },
    {
        "name": "start_pipeline",
        "description": "Generic 'let's go' / 'start' — wake up the Designer at the head of the pipeline.",
        "parameters": {},
    },
    {
        "name": "plan_workflow",
        "description": (
            "User wants a multi-step / end-to-end workflow ('跑完整流程', 'do "
            "everything', 'design then compute then cluster'). Use this when "
            "ONE message implies ≥ 2 sequential agent calls. The pipeline will "
            "ask the planner to draft an ordered plan, present it to the user "
            "for confirmation, and execute step-by-step with confirmations "
            "between steps."
        ),
        "parameters": {
            "goal": {"type": "string",
                     "description": "One-line restatement of what the user wants."},
            "target_k": {"type": "int", "min": 2, "max": 12, "optional": True,
                         "description": "If the user pinned a cluster count for the plan."},
            "mode": {"enum": ["lightweight", "reflective_search"], "optional": True,
                     "description": "Designer mode to use, if specified."},
            "cluster_ids": {"type": "int_list", "optional": True,
                            "description": "If the plan ends in a refine step on specific clusters."},
        },
    },
    {
        "name": "advance_plan",
        "description": (
            "An active plan is waiting on the user. Interpret natural-language "
            "replies like 'go', '继续', '下一步', 'skip', '跳过', 'stop the plan', "
            "'取消计划' as plan control. Use the ``action`` parameter to say which."
        ),
        "parameters": {
            "action": {"enum": ["confirm", "next", "skip", "cancel"],
                       "description": "What the user wants to do with the active plan."},
        },
    },
]


_TOOL_NAMES = {t["name"] for t in AGENT_TOOLS} | {"design_ddia", "compute_mmd"}


@dataclass
class IntentDecision:
    """Structured result returned by ``classify``."""

    tool: str
    next_agent: str
    intent: str
    params: Dict[str, Any]
    confidence: float
    needs_clarification: bool
    clarification: Optional[str]
    reasoning: Optional[str]
    source: str = "llm"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build_system_prompt() -> str:
    tool_lines: List[str] = []
    for t in AGENT_TOOLS:
        params_str = json.dumps(t["parameters"], ensure_ascii=False)
        tool_lines.append(f"  - {t['name']}: {t['description']}\n     params: {params_str}")
    tools_blob = "\n".join(tool_lines)
    return (
        "You are the routing brain of BATTWISE, a multi-agent battery "
        "sorting platform. Given the user's latest message, the current "
        "pipeline state, and a running conversation summary, decide which "
        "TOOL to call next. Output exactly one tool call as strict JSON.\n\n"
        f"AVAILABLE TOOLS:\n{tools_blob}\n\n"
        "DECISION RULES:\n"
        "1. STOP / cancel intent → cancel_running (highest priority).\n"
        "2. Greeting, question, chit-chat, 'how does X work' → answer_question.\n"
        "3. User names specific cluster ids AND mentions refining / All-200 / "
        "sub-cluster → refine_clusters with those ids.\n"
        "4. User names cluster ids of interest WITHOUT a refinement verb "
        "→ select_clusters.\n"
        "5. Otherwise pick the most specific actionable tool.\n"
        "6. Extract parameters explicitly stated. NEVER invent values.\n"
        "7. If the request is ambiguous, contradicts state, or you need one "
        "more fact (e.g. which dataset?), set needs_clarification=true and "
        "supply ``clarification`` as a short follow-up question in the "
        "user's language (Chinese if the user wrote Chinese).\n"
        "8. confidence ∈ [0,1]. Use < 0.6 if you're guessing — caller will "
        "fall back to regex heuristics in that case.\n\n"
        "RESPOND WITH STRICT JSON ONLY (no markdown fences, no prose):\n"
        '{"tool": "<tool_name>", "params": {...}, "confidence": 0.0-1.0,\n'
        ' "needs_clarification": true|false,\n'
        ' "clarification": "<question or null>",\n'
        ' "reasoning": "<one short sentence>"}'
    )


def _build_context_block(
    *,
    session_snapshot: Dict[str, Any],
    summary: Optional[Dict[str, Any]] = None,
    recent_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Render the session/conversation context the LLM sees beside the user msg."""
    parts: List[str] = [
        "CURRENT PIPELINE STATE:",
        json.dumps(session_snapshot, ensure_ascii=False, indent=2),
    ]
    if summary:
        parts.append("\nCONVERSATION SUMMARY (running):")
        parts.append(json.dumps(summary, ensure_ascii=False, indent=2))
    if recent_history:
        parts.append("\nRECENT TURNS:")
        for h in recent_history[-6:]:
            who = h.get("role", "?")
            txt = (h.get("content") or "")[:240]
            parts.append(f"  [{who}] {txt}")
    return "\n".join(parts)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.M)
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = _FENCE_RE.sub("", text.strip())
    match = _JSON_BLOCK_RE.search(cleaned)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _coerce_decision(raw: Dict[str, Any]) -> IntentDecision:
    tool = raw.get("tool")
    if not isinstance(tool, str) or tool not in _TOOL_NAMES:
        tool = "answer_question"
    params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    needs = bool(raw.get("needs_clarification", False))
    clar = raw.get("clarification")
    clar_str = clar if isinstance(clar, str) and clar.strip() else None
    reasoning = raw.get("reasoning")
    reasoning_str = reasoning if isinstance(reasoning, str) else None
    return IntentDecision(
        tool=tool,
        next_agent=TOOL_TO_AGENT.get(tool, "chat"),
        intent=TOOL_TO_INTENT.get(tool, "chitchat"),
        params=params,
        confidence=confidence,
        needs_clarification=needs,
        clarification=clar_str,
        reasoning=reasoning_str,
        source="llm",
    )


def classify(
    *,
    user_text: str,
    session_snapshot: Dict[str, Any],
    summary: Optional[Dict[str, Any]] = None,
    recent_history: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
) -> Optional[IntentDecision]:
    """Run one LLM routing pass. Returns None on hard failure.

    Total tokens kept small (~1.5k system + ~0.6k context + ~0.1k reply)
    so qwen3.5-flash returns well under 2s on a healthy network.
    """
    try:
        system = _build_system_prompt()
        ctx = _build_context_block(
            session_snapshot=session_snapshot,
            summary=summary,
            recent_history=recent_history,
        )
        raw_response = llm_client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{ctx}\n\nUSER MESSAGE:\n{user_text}"},
            ],
            temperature=0.0,
            model=model or ROUTER_MODEL,
            max_retries=1,
            timeout=settings.llm_router_timeout,
            max_tokens=320,
            purpose="router",
        )
    except Exception as exc:
        logger.warning("llm_router.classify LLM call failed: %s", exc)
        return None

    payload = _extract_json(raw_response)
    if not payload:
        logger.warning(
            "llm_router got unparseable response (first 240 chars): %r",
            (raw_response or "")[:240],
        )
        return None
    return _coerce_decision(payload)


async def classify_async(
    *,
    user_text: str,
    session_snapshot: Dict[str, Any],
    summary: Optional[Dict[str, Any]] = None,
    recent_history: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
) -> Optional[IntentDecision]:
    """Async variant used by the live RouterAgent.

    Keeps ``classify`` for script-style tests and legacy callers, but the live
    request path must use this async version so the 8s router timeout is a real
    wall-clock budget rather than an httpx read-timeout hint.
    """
    try:
        system = _build_system_prompt()
        ctx = _build_context_block(
            session_snapshot=session_snapshot,
            summary=summary,
            recent_history=recent_history,
        )
        raw_response = await llm_client.chat_async(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{ctx}\n\nUSER MESSAGE:\n{user_text}"},
            ],
            temperature=0.0,
            model=model or ROUTER_MODEL,
            max_retries=1,
            timeout=settings.llm_router_timeout,
            max_tokens=320,
            purpose="router",
        )
    except Exception as exc:
        logger.warning("llm_router.classify_async LLM call failed: %s", exc)
        return None

    payload = _extract_json(raw_response)
    if not payload:
        logger.warning(
            "llm_router async got unparseable response (first 240 chars): %r",
            (raw_response or "")[:240],
        )
        return None
    return _coerce_decision(payload)


# ---------------------------------------------------------------------------
# Adapter: turn an IntentDecision into the legacy "decision dict" shape that
# pipeline.handle_message already understands. Keeps the blast radius of this
# refactor small — only RouterAgent learns the new contract; the pipeline
# stays on the existing keys (intent / next_agent / target_k / cluster_params
# / eoh_params / selected_clusters / ...).
# ---------------------------------------------------------------------------

def decision_to_pipeline_dict(d: IntentDecision) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "intent": d.intent,
        "next_agent": d.next_agent,
        "source": "llm-router",
        "tool": d.tool,
        "confidence": d.confidence,
        "needs_clarification": d.needs_clarification,
        "clarification": d.clarification,
        "reasoning": d.reasoning,
    }
    p = d.params or {}

    # Designer / EOH parameters.
    if d.tool in {"design_cddi", "design_ddia"}:
        if isinstance(p.get("mode"), str) and p["mode"] in (
            LIGHTWEIGHT_MODE,
            REFLECTIVE_SEARCH_MODE,
            LEGACY_REFLECTIVE_SEARCH_MODE,
        ):
            out["proposed_mode"] = normalize_mode(p["mode"])
        eoh = {}
        for key in ("max_fe", "pop_size", "batch_size"):
            v = p.get(key)
            if isinstance(v, int):
                eoh[key] = v
            elif isinstance(v, str) and v.isdigit():
                eoh[key] = int(v)
        if eoh:
            out["eoh_params"] = eoh
        designer_params = {}
        for key in ("repair_attempts", "sandbox_cells"):
            v = p.get(key)
            if isinstance(v, int):
                designer_params[key] = v
            elif isinstance(v, str) and v.isdigit():
                designer_params[key] = int(v)
        if designer_params:
            out["designer_params"] = designer_params

    # Clustering parameters.
    if d.tool == "run_clustering":
        cluster_params: Dict[str, Any] = {}
        if "target_k" in p:
            try:
                out["target_k"] = int(p["target_k"])
            except (TypeError, ValueError):
                pass
        for key in ("min_k", "max_k", "pop_size", "max_iter"):
            v = p.get(key)
            try:
                if v is not None:
                    cluster_params[key] = int(v)
            except (TypeError, ValueError):
                continue
        fb = p.get("feedback_metric")
        if isinstance(fb, str) and fb in ("combined", "dbi", "variance"):
            cluster_params["feedback_metric"] = fb
        weights: Dict[str, float] = {}
        for src_key, dst_key in (
            ("ocv_weight", "ocv"),
            ("ir_weight", "ir"),
            ("ddia_weight", "ddia"),
            ("cddi_weight", "ddia"),
        ):
            v = p.get(src_key)
            try:
                if v is not None:
                    fv = float(v)
                    if fv > 0:
                        weights[dst_key] = max(0.05, min(20.0, fv))
            except (TypeError, ValueError):
                continue
        if weights:
            cluster_params["feature_weights"] = weights
        if cluster_params:
            out["cluster_params"] = cluster_params

    # Cluster selection / refinement.
    if d.tool in ("select_clusters", "refine_clusters"):
        ids_raw = p.get("cluster_ids") or []
        ids: List[int] = []
        if isinstance(ids_raw, (list, tuple)):
            for v in ids_raw:
                try:
                    ids.append(int(v))
                except (TypeError, ValueError):
                    continue
        if ids:
            ids_sorted = sorted(set(ids))
            out["selected_clusters"] = ids_sorted
            out["selected_cluster"] = ids_sorted[0]

    # Multi-step planner.
    if d.tool == "plan_workflow":
        goal = p.get("goal")
        if isinstance(goal, str) and goal.strip():
            out["plan_goal"] = goal.strip()
        for key in ("target_k", "mode"):
            if key in p and p[key] is not None:
                out[f"plan_hint_{key}"] = p[key]
        if isinstance(p.get("cluster_ids"), list):
            try:
                out["plan_hint_cluster_ids"] = sorted({int(x) for x in p["cluster_ids"]})
            except (TypeError, ValueError):
                pass
    if d.tool == "advance_plan":
        action = p.get("action")
        if isinstance(action, str) and action in ("confirm", "next", "skip", "cancel"):
            out["plan_action"] = action
    return out
