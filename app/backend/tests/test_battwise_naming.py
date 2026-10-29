"""Naming and legacy-mode compatibility contract for BATTWISE."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backend.naming import (  # noqa: E402
    LEGACY_REFLECTIVE_SEARCH_MODE,
    LIGHTWEIGHT_MODE,
    REFLECTIVE_SEARCH_MODE,
    SYSTEM_LONG_NAME,
    SYSTEM_NAME,
    is_reflective_search_mode,
    normalize_mode,
)
from app.backend.orchestrator.session_store import Session  # noqa: E402


def main() -> None:
    assert SYSTEM_NAME == "BATTWISE"
    assert SYSTEM_LONG_NAME == (
        "Battery Agentic Triage and Testing Workflow with Internal-State Evidence"
    )
    assert LIGHTWEIGHT_MODE == "lightweight"
    assert REFLECTIVE_SEARCH_MODE == "reflective_search"
    assert LEGACY_REFLECTIVE_SEARCH_MODE == "full_eoh"

    assert normalize_mode("lightweight") == LIGHTWEIGHT_MODE
    assert normalize_mode("reflective_search") == REFLECTIVE_SEARCH_MODE
    assert normalize_mode("full_eoh") == REFLECTIVE_SEARCH_MODE
    assert normalize_mode("  FULL_EOH  ") == REFLECTIVE_SEARCH_MODE
    assert is_reflective_search_mode("reflective_search")
    assert is_reflective_search_mode("full_eoh")
    assert not is_reflective_search_mode("lightweight")

    legacy = Session(session_id="legacy-mode", mode="full_eoh")
    assert legacy.mode == REFLECTIVE_SEARCH_MODE
    assert legacy.to_dict()["mode"] == REFLECTIVE_SEARCH_MODE
    print("test_battwise_naming: PASS")


if __name__ == "__main__":
    main()
