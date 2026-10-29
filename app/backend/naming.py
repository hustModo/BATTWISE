"""Canonical names and compatibility aliases for the BATTWISE system.

This module is intentionally dependency-free so persisted sessions, REST
handlers, agents, and the frontend contract can share one stable vocabulary.
Legacy identifiers are accepted at system boundaries only.
"""
from __future__ import annotations

SYSTEM_NAME = "BATTWISE"
METRIC_NAME = "CDDI"
METHOD_NAME = "BATTWISE-CDDI"
SYSTEM_LONG_NAME = (
    "Battery Agentic Triage and Testing Workflow with Internal-State Evidence"
)

LIGHTWEIGHT_MODE = "lightweight"
REFLECTIVE_SEARCH_MODE = "reflective_search"
LEGACY_REFLECTIVE_SEARCH_MODE = "full_eoh"

SUPPORTED_MODES = (LIGHTWEIGHT_MODE, REFLECTIVE_SEARCH_MODE)


def normalize_mode(mode: object) -> str:
    """Return the canonical runtime mode for new and persisted values."""
    value = str(mode or "").strip().lower().replace("-", "_")
    if value in {
        REFLECTIVE_SEARCH_MODE,
        LEGACY_REFLECTIVE_SEARCH_MODE,
        "reflective",
        "reflection",
        "reflective_metric_search",
    }:
        return REFLECTIVE_SEARCH_MODE
    return LIGHTWEIGHT_MODE


def is_reflective_search_mode(mode: object) -> bool:
    """Return whether a canonical or legacy value selects Reflective Search."""
    return normalize_mode(mode) == REFLECTIVE_SEARCH_MODE
