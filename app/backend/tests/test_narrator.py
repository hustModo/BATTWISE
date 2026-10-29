"""Unit tests for services.narrator — contextual agent-finish wrapper."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.orchestrator.session_store import Session  # noqa: E402
from app.backend.services import narrator  # noqa: E402


def _make_session() -> Session:
    s = Session(session_id="test-narrator")
    s.messages = [
        {"id": "1", "role": "user", "agent": None, "text": "design a new DDIA", "t": 0},
    ]
    return s


def _mock_llm(reply: str):
    return patch("app.backend.services.narrator.llm_client.chat", return_value=reply)


def test_narrate_happy_path_designer() -> int:
    s = _make_session()
    with _mock_llm("DDIA 的设计看起来不错，你可以让我在所有 cell 上跑一次 Compute。"):
        text = narrator.narrate(
            agent_id="designer",
            result={"passed": True, "function_name": "calculate_mmd_v3",
                    "attempts": 1, "mean_r": 0.83, "mode": "lightweight"},
            session=s,
        )
    assert text and "Compute" in text, text
    return 0


def test_narrate_skips_router_and_chat() -> int:
    s = _make_session()
    with _mock_llm("should not be called"):
        for agent_id in ("router", "chat"):
            text = narrator.narrate(agent_id=agent_id, result={"ok": True}, session=s)
            assert text is None, f"narrator should skip agent={agent_id}, got {text!r}"
    return 0


def test_narrate_skips_on_agent_error() -> int:
    s = _make_session()
    failure_results = [
        {"error": "compute timeout"},
        {"passed": False, "attempts": 3},
        {"cancelled": True, "best_k": 4},
    ]
    with _mock_llm("should not be called"):
        for r in failure_results:
            text = narrator.narrate(agent_id="cluster", result=r, session=s)
            assert text is None, f"expected None for failure result {r}, got {text!r}"
    return 0


def test_narrate_disabled_by_env() -> int:
    s = _make_session()
    old = os.environ.get(narrator.NARRATOR_DISABLED_ENV)
    os.environ[narrator.NARRATOR_DISABLED_ENV] = "1"
    try:
        with _mock_llm("should not be reached"):
            text = narrator.narrate(
                agent_id="designer",
                result={"passed": True},
                session=s,
            )
        assert text is None, "env-disabled narrator must return None"
        assert narrator.is_disabled() is True
    finally:
        if old is None:
            os.environ.pop(narrator.NARRATOR_DISABLED_ENV, None)
        else:
            os.environ[narrator.NARRATOR_DISABLED_ENV] = old
    return 0


def test_narrate_empty_llm_reply_returns_none() -> int:
    s = _make_session()
    for reply in ("", "   \n  ", "\n"):
        with _mock_llm(reply):
            text = narrator.narrate(
                agent_id="compute", result={"passed": True}, session=s,
            )
        assert text is None, f"empty reply {reply!r} should yield None"
    return 0


def test_narrate_llm_exception_returns_none() -> int:
    s = _make_session()
    with patch(
        "app.backend.services.narrator.llm_client.chat",
        side_effect=RuntimeError("network down"),
    ):
        text = narrator.narrate(agent_id="compute", result={"passed": True}, session=s)
    assert text is None
    return 0


def test_narrate_strips_markdown_fences() -> int:
    s = _make_session()
    with _mock_llm("```\nDone. Try Compute next.\n```"):
        text = narrator.narrate(
            agent_id="designer", result={"passed": True}, session=s,
        )
    assert text is not None
    assert "```" not in text, f"fence should be stripped: {text!r}"
    return 0


def test_result_brief_designer_keeps_only_headlines() -> int:
    raw = {
        "passed": True,
        "function_name": "calculate_mmd_v3",
        "attempts": 1,
        "mean_r": 0.83,
        "mode": "lightweight",
        # Noise that should NOT make it into the brief.
        "raw_corr_matrix": [[0.1] * 50 for _ in range(50)],
        "per_cell_results": [{"id": i, "mmd": 0.0} for i in range(71)],
    }
    brief = narrator._result_brief("designer", raw)
    assert "raw_corr_matrix" not in brief, brief.keys()
    assert "per_cell_results" not in brief, brief.keys()
    assert brief["mean_r"] == 0.83
    return 0


def test_narrate_context_includes_plan_and_summary() -> int:
    s = _make_session()
    s.context["conversation_summary"] = {
        "what_user_wants": "End-to-end pipeline on First-200s",
        "what_was_done": ["DDIA designed"],
        "current_state": "Just finished Compute",
        "user_preferences": {"feedback_metric": "variance"},
        "open_questions": [],
    }
    s.context["active_plan"] = {
        "goal": "Run pipeline",
        "steps": [
            {"id": 0, "agent": "designer", "description": "Design",
             "params": {}, "status": "done"},
            {"id": 1, "agent": "compute", "description": "Compute",
             "params": {}, "status": "done"},
            {"id": 2, "agent": "cluster", "description": "Cluster",
             "params": {}, "status": "pending"},
        ],
        "current_idx": 2,
        "status": "in_progress",
    }

    captured: dict = {}

    def _spy(*, messages, **kw):
        captured["messages"] = messages
        return "Cluster is queued up next — say 继续 when ready."

    with patch("app.backend.services.narrator.llm_client.chat", side_effect=_spy):
        text = narrator.narrate(
            agent_id="compute",
            result={"passed": True, "computed": 71, "mean_sig": 0.1, "elapsed_s": 0.4},
            session=s,
            prompt_hint="Plan step 2/3 just finished; step 3 is Cluster.",
        )
    assert text and "继续" in text, text
    user_msg = captured["messages"][-1]["content"]
    assert "active_plan" in user_msg, "plan should be passed in context"
    assert "conversation_summary" in user_msg, "summary should be passed in context"
    assert "prompt_hint" in user_msg
    return 0


def main() -> int:
    tests = [
        test_narrate_happy_path_designer,
        test_narrate_skips_router_and_chat,
        test_narrate_skips_on_agent_error,
        test_narrate_disabled_by_env,
        test_narrate_empty_llm_reply_returns_none,
        test_narrate_llm_exception_returns_none,
        test_narrate_strips_markdown_fences,
        test_result_brief_designer_keeps_only_headlines,
        test_narrate_context_includes_plan_and_summary,
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
