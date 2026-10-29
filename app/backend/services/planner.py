"""LLM-driven workflow planner.

When the user asks for a multi-step thing ("跑完整流程 k=4", "do everything
end-to-end then refine cluster 3"), the smart router hands the request to
this service. We then:

1. Ask an LLM to draft an ordered list of agent dispatches that would
   satisfy the goal. Each step is one of the existing agents (designer /
   compute / cluster / refine).
2. Validate / coerce the LLM's output into a typed structure.
3. The caller (orchestrator/pipeline.py) presents the plan to the user
   for confirmation, then executes the steps one by one — pausing between
   steps so the user can confirm, skip, modify, or stop.

We deliberately keep plans short and "single-shot" — no branching, no
loops. The user can always cancel and start over.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app.backend.services import llm_client
from app.backend.naming import (
    LEGACY_REFLECTIVE_SEARCH_MODE,
    LIGHTWEIGHT_MODE,
    REFLECTIVE_SEARCH_MODE,
    normalize_mode,
)

logger = logging.getLogger(__name__)


PLANNER_MODEL = "qwen3.5-flash"

# Hard upper bound: plans longer than this are silently truncated. Demos
# rarely need more than ~6 sequential steps; bigger plans are usually a
# sign the LLM hallucinated.
MAX_PLAN_STEPS = 8

# Agents the planner is allowed to call.
ALLOWED_AGENTS = {"designer", "compute", "cluster", "refine"}


# ---------------------------------------------------------------------------
# Per-agent parameter schema. The single source of truth for what knobs are
# editable on a plan step (PlanPanel form on the frontend) and how to clamp
# values when the user posts a ``plan_update_step`` action. Frontend mirrors
# this shape — keep them in sync.
#
# Shape:
#   { "<agent>": { "<param>": {"type": "int"|"float"|"enum"|"int_list",
#                              "min": int, "max": int,         # int only
#                              "options": [...],                # enum only
#                              "label": "Display label",
#                              "description": "tooltip text"} } }
# ---------------------------------------------------------------------------

PARAM_SCHEMA: Dict[str, Dict[str, Dict[str, Any]]] = {
    "designer": {
        "mode": {
            "type": "enum",
            "options": ["lightweight", "reflective_search"],
            "label": "Mode",
            "description": "Lightweight = one-shot LLM design; Reflective Search = performance-guided program search.",
        },
        "max_fe": {"type": "int", "min": 1, "max": 30, "label": "Max generations",
                   "description": "Search generations (only used in Reflective Search)."},
        "pop_size": {"type": "int", "min": 2, "max": 32, "label": "Population",
                     "description": "Individuals per generation."},
        "batch_size": {"type": "int", "min": 1, "max": 32, "label": "Batch size",
                       "description": "Per-generation evaluation batch."},
    },
    # Compute has no user-tunable knobs at the moment.
    "compute": {},
    "cluster": {
        "target_k": {"type": "int", "min": 2, "max": 12, "label": "Target k",
                     "description": "Fixed cluster count; leave blank to let GA pick."},
        "min_k": {"type": "int", "min": 2, "max": 12, "label": "Min k",
                  "description": "Lower bound for the GA k search."},
        "max_k": {"type": "int", "min": 2, "max": 12, "label": "Max k",
                  "description": "Upper bound for the GA k search."},
        "pop_size": {"type": "int", "min": 20, "max": 500, "label": "GA population",
                     "description": "Cluster-GA population size."},
        "max_iter": {"type": "int", "min": 10, "max": 500, "label": "GA iterations",
                     "description": "Cluster-GA max generations."},
        "feedback_metric": {
            "type": "enum",
            "options": ["combined", "dbi", "variance"],
            "label": "Feedback metric",
            "description": "GA fitness criterion.",
        },
        "ocv_weight": {"type": "float", "min": 0.05, "max": 20.0, "label": "OCV weight",
                       "description": "Relative OCV weight after normalization; default 1."},
        "ir_weight": {"type": "float", "min": 0.05, "max": 20.0, "label": "IR weight",
                      "description": "Relative internal-resistance weight after normalization; default 1."},
        "cddi_weight": {"type": "float", "min": 0.05, "max": 20.0, "label": "CDDI weight",
                        "description": "Relative CDDI weight after normalization; default 1."},
    },
    "refine": {
        "cluster_ids": {
            "type": "int_list",
            "min": 0, "max": 50,
            "label": "Cluster IDs",
            "description": "Which clusters to drill into for refinement.",
        },
    },
}


def validate_step_params(agent: str, raw_params: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce + clamp user-supplied params against the schema.

    * Unknown keys are dropped.
    * ``int`` keys are clamped to [min, max]; non-int values that
      can be parsed (``"4"``) are coerced, others dropped.
    * ``enum`` keys are kept only when the value is in ``options``.
    * ``int_list`` keys are filtered to int values inside [min, max]
      and de-duped + sorted.

    Returns a fresh dict — never mutates the caller's input.
    """
    schema = PARAM_SCHEMA.get(agent, {})
    out: Dict[str, Any] = {}
    if not isinstance(raw_params, dict):
        return out
    raw_params = dict(raw_params)
    if agent == "cluster" and "cddi_weight" not in raw_params and "ddia_weight" in raw_params:
        raw_params["cddi_weight"] = raw_params["ddia_weight"]
    for key, spec in schema.items():
        if key not in raw_params:
            continue
        v = raw_params[key]
        if v is None or v == "":
            continue
        kind = spec.get("type")
        if kind == "int":
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            lo = int(spec.get("min", -10**9))
            hi = int(spec.get("max", 10**9))
            out[key] = max(lo, min(hi, iv))
        elif kind == "float":
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            lo = float(spec.get("min", -10**9))
            hi = float(spec.get("max", 10**9))
            out[key] = max(lo, min(hi, fv))
        elif kind == "enum":
            if key == "mode" and v in {
                LIGHTWEIGHT_MODE,
                REFLECTIVE_SEARCH_MODE,
                LEGACY_REFLECTIVE_SEARCH_MODE,
            }:
                out[key] = normalize_mode(v)
            elif v in (spec.get("options") or ()):
                out[key] = v
        elif kind == "int_list":
            if not isinstance(v, (list, tuple)):
                continue
            lo = int(spec.get("min", -10**9))
            hi = int(spec.get("max", 10**9))
            cleaned: List[int] = []
            for x in v:
                try:
                    ix = int(x)
                except (TypeError, ValueError):
                    continue
                if lo <= ix <= hi:
                    cleaned.append(ix)
            if cleaned:
                out[key] = sorted(set(cleaned))
    return out


# ---------------------------------------------------------------------------
# Typed plan representation
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    id: int
    agent: str           # designer | compute | cluster | refine
    description: str     # short user-facing label
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending | running | done | skipped | cancelled | error
    result_summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowPlan:
    goal: str
    steps: List[PlanStep]
    current_idx: int = 0
    status: str = "awaiting_confirmation"  # awaiting_confirmation | in_progress | done | cancelled

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "current_idx": self.current_idx,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowPlan":
        steps_raw = data.get("steps") or []
        steps: List[PlanStep] = []
        for s in steps_raw:
            if not isinstance(s, dict):
                continue
            steps.append(
                PlanStep(
                    id=int(s.get("id", len(steps))),
                    agent=str(s.get("agent", "")),
                    description=str(s.get("description", "")),
                    params=dict(s.get("params") or {}),
                    status=str(s.get("status", "pending")),
                    result_summary=s.get("result_summary"),
                )
            )
        return cls(
            goal=str(data.get("goal", "")),
            steps=steps,
            current_idx=int(data.get("current_idx", 0)),
            status=str(data.get("status", "awaiting_confirmation")),
        )

    def current_step(self) -> Optional[PlanStep]:
        if 0 <= self.current_idx < len(self.steps):
            return self.steps[self.current_idx]
        return None

    def is_complete(self) -> bool:
        return self.current_idx >= len(self.steps)


# ---------------------------------------------------------------------------
# LLM-driven plan generation
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """You are the workflow planner for BATTWISE, a
multi-agent battery sorting platform. Given a user's high-level goal and
the current pipeline state, draft an ordered list of agent dispatches that
will satisfy that goal.

Available agents (and what they do):
- designer: design / regenerate the CDDI (MMD-like) function. Params:
    {"mode": "lightweight" | "reflective_search",
     "max_fe"?: int, "pop_size"?: int, "batch_size"?: int}
- compute: run the latest CDDI function across all cell files. No params.
- cluster: run adaptive GA-KMeans clustering. Params:
    {"target_k"?: int, "min_k"?: int, "max_k"?: int,
     "pop_size"?: int, "max_iter"?: int,
     "feedback_metric"?: "combined" | "dbi" | "variance"}
- refine: re-cluster a chosen subset of cells using the All-200 dataset.
    Params: {"cluster_ids"?: [int]}.

RULES:
- Produce 1 to 6 sequential steps. No branches, no parallelism.
- Each step uses exactly one allowed agent. Never invent agent names.
- Use the user's language for the `description` field.
- Pull parameters EXPLICITLY from the user's goal. Do NOT invent values
  for k, pop_size, max_fe, etc.
- The classic full-pipeline order is: designer → compute → cluster
  → (optional) refine.
- If the pipeline state shows a step is already 'done', you MAY omit it.
- If the goal is vague ("do everything"), default to:
    1) designer (lightweight) 2) compute 3) cluster (auto-k).

OUTPUT STRICT JSON (no markdown fences, no prose):
{
  "goal": "<one-line restatement of what the user asked>",
  "steps": [
    {"agent": "<name>", "description": "<short label>", "params": {...}},
    ...
  ]
}
"""


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


def _coerce_plan(raw: Dict[str, Any], *, fallback_goal: str = "") -> Optional[WorkflowPlan]:
    """Validate and normalise an LLM plan dict into a WorkflowPlan.

    Returns None if there are no valid steps left after filtering.
    """
    goal = str(raw.get("goal", fallback_goal) or fallback_goal).strip()
    raw_steps = raw.get("steps") or []
    if not isinstance(raw_steps, list):
        return None

    steps: List[PlanStep] = []
    for s in raw_steps[:MAX_PLAN_STEPS]:
        if not isinstance(s, dict):
            continue
        agent = str(s.get("agent", "")).strip().lower()
        if agent not in ALLOWED_AGENTS:
            continue
        description = str(s.get("description") or f"Run {agent}").strip()[:120]
        # Route LLM-emitted params through the same validator the UI uses
        # — clamps numeric ranges, drops unknown keys, sanitises enums.
        params = validate_step_params(agent, s.get("params") or {})
        steps.append(PlanStep(
            id=len(steps),
            agent=agent,
            description=description,
            params=params,
            status="pending",
        ))

    if not steps:
        return None
    return WorkflowPlan(goal=goal, steps=steps, current_idx=0, status="awaiting_confirmation")


def generate_plan(
    *,
    user_goal: str,
    session_snapshot: Dict[str, Any],
    model: Optional[str] = None,
) -> Optional[WorkflowPlan]:
    """Ask the LLM to draft a plan for ``user_goal``.

    Returns None on hard failure — the caller should fall back to a safe
    default (e.g. design → compute → cluster) or just route the message
    normally.
    """
    user_block = (
        f"USER GOAL:\n{user_goal}\n\n"
        f"CURRENT PIPELINE STATE:\n{json.dumps(session_snapshot, ensure_ascii=False, indent=2)}"
    )
    try:
        raw = llm_client.chat(
            messages=[
                {"role": "system", "content": _PLANNER_SYSTEM},
                {"role": "user", "content": user_block},
            ],
            temperature=0.0,
            model=model or PLANNER_MODEL,
            max_retries=1,
            timeout=20.0,
            max_tokens=600,
            purpose="planner",
        )
    except Exception as exc:
        logger.warning("planner.generate_plan LLM call failed: %s", exc)
        return None

    payload = _extract_json(raw)
    if not payload:
        logger.warning("planner got unparseable LLM reply: %r", (raw or "")[:240])
        return None
    return _coerce_plan(payload, fallback_goal=user_goal)


def default_plan(user_goal: str) -> WorkflowPlan:
    """Hard-coded fallback used when the LLM can't draft a plan.

    Matches the canonical demo flow: design → compute → cluster.
    """
    return WorkflowPlan(
        goal=user_goal or "Run the standard pipeline (design → compute → cluster).",
        steps=[
            PlanStep(id=0, agent="designer",
                     description="设计 CDDI 函数 (lightweight)",
                     params={"mode": "lightweight"}),
            PlanStep(id=1, agent="compute",
                     description="在全部 cell 上计算 CDDI",
                     params={}),
            PlanStep(id=2, agent="cluster",
                     description="自适应 GA-KMeans 聚类 (auto k)",
                     params={}),
        ],
        current_idx=0,
        status="awaiting_confirmation",
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_AGENT_DISPLAY = {
    "designer": "Designer",
    "compute":  "Compute",
    "cluster":  "Cluster",
    "refine":   "Refine",
}

_STATUS_GLYPH = {
    "pending":   "○",
    "running":   "▶",
    "done":      "✓",
    "skipped":   "↷",
    "cancelled": "✗",
    "error":     "!",
}


def render_plan_markdown(plan: WorkflowPlan, *, header: Optional[str] = None) -> str:
    """Format a plan as a chat-message-friendly markdown block."""
    lines: List[str] = []
    if header:
        lines.append(header)
    lines.append(f"**目标**: {plan.goal}")
    lines.append("")
    for s in plan.steps:
        cursor = "→ " if s.id == plan.current_idx and plan.status == "in_progress" else "   "
        glyph = _STATUS_GLYPH.get(s.status, "?")
        agent_label = _AGENT_DISPLAY.get(s.agent, s.agent)
        params_str = ""
        if s.params:
            kv = ", ".join(f"{k}={v}" for k, v in s.params.items())
            params_str = f"  _({kv})_"
        lines.append(f"{cursor}{glyph} **{s.id + 1}. {agent_label}** — {s.description}{params_str}")
    return "\n".join(lines)
