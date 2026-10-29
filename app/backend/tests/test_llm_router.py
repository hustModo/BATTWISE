"""Unit tests for the LLM-based intent router (``services.llm_router``).

Mocks the LLM call so the tests run offline. Covers:
  * happy-path tool dispatch with parameter extraction
  * malformed JSON → returns None (caller falls back to regex)
  * unknown tool → coerced to ``answer_question``
  * low / high confidence respected
  * ``decision_to_pipeline_dict`` shape matches the keys the pipeline
    branches read (target_k / cluster_params / eoh_params /
    selected_clusters / selected_cluster / proposed_mode / needs_clarification)
  * clarifications round-trip
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.services import llm_router  # noqa: E402


SNAPSHOT = {
    "active_agent": None,
    "pipeline_state": {"router": "idle", "designer": "idle", "compute": "idle",
                       "cluster": "idle", "refine": "idle"},
    "mode": "lightweight",
    "active_dataset_id": "builtin-first-200s",
    "selected_clusters": [],
    "selected_cluster": None,
    "sticky_eoh_params": None,
    "sticky_cluster_params": None,
    "pending_design_confirmation": False,
    "recent_artifacts": [],
    "message_count": 1,
}


def _mock_llm(reply: str):
    """Return a context manager that patches llm_client.chat to a fixed reply."""
    return patch("app.backend.services.llm_router.llm_client.chat", return_value=reply)


def test_design_ddia_happy_path() -> int:
    reply = (
        '{"tool": "design_ddia", "params": {"mode": "full_eoh", "max_fe": 5, "pop_size": 6,'
        ' "repair_attempts": 7, "sandbox_cells": 4},'
        ' "confidence": 0.9, "needs_clarification": false, "clarification": null,'
        ' "reasoning": "User explicitly asked for Full EOH with 5 generations."}'
    )
    with _mock_llm(reply):
        d = llm_router.classify(user_text="run full EOH 5 generations pop 6",
                                session_snapshot=SNAPSHOT)
    assert d is not None, "expected a decision"
    assert d.tool == "design_ddia", d
    assert d.next_agent == "designer", d
    assert d.intent == "design_mmd", d
    assert d.confidence == 0.9, d
    pl = llm_router.decision_to_pipeline_dict(d)
    assert pl["intent"] == "design_mmd"
    assert pl["next_agent"] == "designer"
    assert pl["proposed_mode"] == "reflective_search"
    assert pl["eoh_params"] == {"max_fe": 5, "pop_size": 6}, pl
    assert pl["designer_params"] == {"repair_attempts": 7, "sandbox_cells": 4}, pl
    return 0


def test_cluster_with_params() -> int:
    reply = (
        '{"tool": "run_clustering", "params": {"target_k": 6, "feedback_metric": "variance"},'
        ' "confidence": 0.85, "needs_clarification": false}'
    )
    with _mock_llm(reply):
        d = llm_router.classify(user_text="cluster k=6 use variance feedback",
                                session_snapshot=SNAPSHOT)
    pl = llm_router.decision_to_pipeline_dict(d)
    assert pl["intent"] == "cluster", pl
    assert pl["target_k"] == 6, pl
    assert pl["cluster_params"]["feedback_metric"] == "variance", pl
    return 0


def test_refine_clusters_with_ids() -> int:
    reply = (
        '{"tool": "refine_clusters", "params": {"cluster_ids": [0, 3, 5]},'
        ' "confidence": 0.88, "needs_clarification": false}'
    )
    with _mock_llm(reply):
        d = llm_router.classify(user_text="对类别 0、3、5 做 All-200 精细聚类",
                                session_snapshot=SNAPSHOT)
    pl = llm_router.decision_to_pipeline_dict(d)
    assert pl["intent"] == "refine_all200", pl
    assert pl["selected_clusters"] == [0, 3, 5], pl
    assert pl["selected_cluster"] == 0, pl
    return 0


def test_select_clusters_dedup_and_sort() -> int:
    reply = (
        '{"tool": "select_clusters", "params": {"cluster_ids": [3, 0, 3, 5]},'
        ' "confidence": 0.7, "needs_clarification": false}'
    )
    with _mock_llm(reply):
        d = llm_router.classify(user_text="interested in clusters 3, 0, 3, 5",
                                session_snapshot=SNAPSHOT)
    pl = llm_router.decision_to_pipeline_dict(d)
    assert pl["selected_clusters"] == [0, 3, 5], pl
    return 0


def test_clarification_round_trip() -> int:
    reply = (
        '{"tool": "design_ddia", "params": {}, "confidence": 0.65,'
        ' "needs_clarification": true,'
        ' "clarification": "你想用 Lightweight 还是 Full EOH 模式？"}'
    )
    with _mock_llm(reply):
        d = llm_router.classify(user_text="再来一次", session_snapshot=SNAPSHOT)
    assert d.needs_clarification is True, d
    assert d.clarification and "Lightweight" in d.clarification, d
    pl = llm_router.decision_to_pipeline_dict(d)
    assert pl["needs_clarification"] is True
    assert pl["clarification"] == d.clarification
    return 0


def test_cancel_routes_through_llm_too() -> int:
    reply = '{"tool": "cancel_running", "params": {}, "confidence": 0.95, "needs_clarification": false}'
    with _mock_llm(reply):
        d = llm_router.classify(user_text="enough already", session_snapshot=SNAPSHOT)
    pl = llm_router.decision_to_pipeline_dict(d)
    assert pl["intent"] == "cancel", pl
    assert pl["next_agent"] == "cancel", pl
    return 0


def test_unknown_tool_coerced_to_chat() -> int:
    reply = '{"tool": "make_coffee", "params": {}, "confidence": 0.9, "needs_clarification": false}'
    with _mock_llm(reply):
        d = llm_router.classify(user_text="make me a flat white", session_snapshot=SNAPSHOT)
    assert d.tool == "answer_question", d
    assert d.next_agent == "chat", d
    return 0


def test_unparseable_reply_returns_none() -> int:
    with _mock_llm("Hi! I think you want to design something maybe."):
        d = llm_router.classify(user_text="hi", session_snapshot=SNAPSHOT)
    assert d is None, "expected None on unparseable reply"
    return 0


def test_fenced_json_is_unwrapped() -> int:
    reply = """```json
{"tool": "compute_mmd", "params": {}, "confidence": 0.92, "needs_clarification": false}
```"""
    with _mock_llm(reply):
        d = llm_router.classify(user_text="run mmd", session_snapshot=SNAPSHOT)
    assert d is not None
    assert d.tool == "compute_mmd", d
    return 0


def test_llm_exception_returns_none() -> int:
    with patch(
        "app.backend.services.llm_router.llm_client.chat",
        side_effect=RuntimeError("network down"),
    ):
        d = llm_router.classify(user_text="design", session_snapshot=SNAPSHOT)
    assert d is None, "expected None when LLM raises"
    return 0


def main() -> int:
    tests = [
        test_design_ddia_happy_path,
        test_cluster_with_params,
        test_refine_clusters_with_ids,
        test_select_clusters_dedup_and_sort,
        test_clarification_round_trip,
        test_cancel_routes_through_llm_too,
        test_unknown_tool_coerced_to_chat,
        test_unparseable_reply_returns_none,
        test_fenced_json_is_unwrapped,
        test_llm_exception_returns_none,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  [OK] {fn.__name__}")
        except AssertionError as exc:
            print(f"  [FAIL] {fn.__name__}: {exc}")
            failed += 1
    print("PASSED" if failed == 0 else f"FAILED ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
