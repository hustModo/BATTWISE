"""LLM-powered "narrator" that wraps agent finishes in natural language.

Each worker agent (Designer / Compute / Cluster / Refine) already posts a
precise technical summary when it finishes — file paths, stats, attempt
counts. That message remains. The narrator's job is to add ONE extra
short message right after, in the user's language, informed by the
running conversation summary, that:

1. acknowledges what happened in human terms ("the design looks solid",
   "clustering converged to 5 groups"),
2. suggests the most natural next step.

The point is to make the system feel like a thoughtful assistant rather
than a status pipeline. Per user feedback: "让人用的智能一点".

Behaviour
---------
* Narration runs async after a successful agent run (or after a plan
  step). Failure (LLM error, garbage reply) is non-fatal — we return
  None and the caller skips posting anything extra.
* Disabled entirely when ``BATRECYCLER_DISABLE_NARRATOR=1`` is set in
  the env. Useful for unit tests that want deterministic output.
* Skipped for "boring" agents (router / chat) whose own messages are
  already conversational.
* Skipped on error / cancellation — narrating a failure feels off; the
  agent's own error message stays untouched.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from app.backend.orchestrator.session_store import Session
from app.backend.services import conversation_memory, llm_client

logger = logging.getLogger(__name__)


NARRATOR_MODEL = "qwen3.5-flash"
NARRATOR_DISABLED_ENV = "BATRECYCLER_DISABLE_NARRATOR"

# Agents the narrator cares about. Router / chat already speak naturally.
_NARRATED_AGENTS = {"designer", "compute", "cluster", "refine"}


_SYSTEM_PROMPT = """You are the closing voice of BATTWISE, a
multi-agent battery sorting platform. A worker agent just finished a
task. Compose ONE short follow-up message that:

1. Acknowledges what was accomplished in HUMAN terms — not jargon, not
   exact numbers (the user just saw those in the agent's own summary).
   Phrasings like "the design looks solid", "clustering converged
   cleanly", "the refinement is in" are good.
2. Suggests ONE concrete next step the user can take, phrased as an
   invitation rather than an order.

CRITICAL RULES:
- Respond in the user's dominant language. If the recent dialogue is
  Chinese (Simplified), reply in Chinese; if English, reply in English.
- Output 1–2 sentences total, ≤ 220 characters. NO bullet points, NO
  markdown headers, NO JSON. Just plain prose.
- Do NOT repeat exact statistics already in the agent's summary.
- If a workflow plan is currently in progress, reference it (e.g.
  "step 2 is next — ready when you are") instead of suggesting a
  totally different direction.
- If `prompt_hint` is provided, weave it into your closing.
"""


def is_disabled() -> bool:
    """True iff narration is turned off via env. Useful for tests."""
    return os.environ.get(NARRATOR_DISABLED_ENV) == "1"


# Result-summarisation helpers — agent results can carry deeply nested
# objects (cluster_iter histories, per-cell stats) that we don't want
# to dump into the narrator prompt. We extract just the salient fields.

_RESULT_FIELDS_BY_AGENT: Dict[str, tuple[str, ...]] = {
    "designer": ("passed", "function_name", "attempts", "mean_r", "mode",
                 "best_objective", "generations"),
    "compute":  ("passed", "computed", "skipped", "mean_sig", "elapsed_s"),
    "cluster":  ("passed", "best_k", "n_cells", "cancelled",
                 "dbi", "variance", "capacity_f_ratio"),
    "refine":   ("passed", "subclustered", "merged_k_before", "merged_k_after"),
}


def _result_brief(agent_id: str, result: Any) -> Dict[str, Any]:
    """Trim a noisy result dict to the small set of headline fields we want
    the LLM to see. Unknown agents → return as-is, capped to 8 keys."""
    if not isinstance(result, dict):
        return {"summary": str(result)[:200]}
    keys = _RESULT_FIELDS_BY_AGENT.get(agent_id)
    if keys:
        return {k: result[k] for k in keys if k in result}
    return {k: result[k] for k in list(result)[:8]}


def _build_context(
    *,
    agent_id: str,
    result: Any,
    session: Session,
    prompt_hint: Optional[str],
) -> str:
    summary = conversation_memory.get_summary(session)
    plan = session.context.get("active_plan")
    plan_brief: Optional[Dict[str, Any]] = None
    if isinstance(plan, dict):
        steps = plan.get("steps") or []
        plan_brief = {
            "goal": plan.get("goal"),
            "status": plan.get("status"),
            "current_idx": plan.get("current_idx"),
            "step_count": len(steps),
            "next_step": (
                steps[plan["current_idx"]]
                if 0 <= int(plan.get("current_idx", -1)) < len(steps)
                else None
            ),
        }
    ctx = {
        "agent": agent_id,
        "result_brief": _result_brief(agent_id, result),
        "session": {
            "mode": session.mode,
            "active_dataset_id": session.active_dataset_id,
            "selected_clusters": list(session.selected_clusters or []),
            "pipeline_state": dict(session.pipeline_state),
        },
        "active_plan": plan_brief,
        "conversation_summary": summary,
        "prompt_hint": prompt_hint,
    }
    return json.dumps(ctx, ensure_ascii=False, indent=2)


def _is_failure(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("error"):
        return True
    if result.get("passed") is False:
        return True
    if result.get("cancelled"):
        return True
    return False


def narrate(
    *,
    agent_id: str,
    result: Any,
    session: Session,
    prompt_hint: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """Produce a contextual closing message for an agent run.

    Returns ``None`` (caller posts nothing extra) when:
      * the narrator is disabled by env,
      * the agent isn't in the narrated set,
      * the agent run failed / was cancelled,
      * the LLM call fails or returns empty text.

    ``prompt_hint`` is a short hint from the caller — e.g.
    "Plan step 2/4 just finished; step 3 is Cluster" — that the LLM
    should weave in.
    """
    if is_disabled():
        return None
    if agent_id not in _NARRATED_AGENTS:
        return None
    if _is_failure(result):
        return None

    user_block = _build_context(
        agent_id=agent_id, result=result, session=session, prompt_hint=prompt_hint,
    )
    try:
        raw = llm_client.chat(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_block},
            ],
            temperature=0.4,
            model=model or NARRATOR_MODEL,
            max_retries=1,
            timeout=10.0,
            max_tokens=240,
            purpose="narrator",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("narrator.narrate LLM call failed: %s", exc)
        return None

    text = (raw or "").strip()
    if not text:
        return None
    # Strip accidental markdown fences / JSON wrappers — the LLM should
    # have followed instructions, but defensive trimming is cheap.
    if text.startswith("```"):
        text = text.strip("`").strip()
    # Single-paragraph cap so the message panel doesn't blow up.
    return text[:600]
