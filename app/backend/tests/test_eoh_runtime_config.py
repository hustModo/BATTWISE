"""Regression tests for passing resolved LLM settings to Full EOH."""
from __future__ import annotations

from app.backend.services.eoh_runner import prepare_eoh_runtime
from app.backend.services.llm_client import ResolvedConfig


def _config() -> ResolvedConfig:
    return ResolvedConfig(
        model="configured-model",
        base_url="https://llm.example/v1",
        api_key="test-key-not-a-real-secret",
        temperature=0.5,
        source="test",
    )


def test_resolved_config_is_added_to_child_only() -> None:
    overrides, env = prepare_eoh_runtime(
        ["max_fe=4"], {"UNCHANGED": "yes"}, _config()
    )
    assert overrides == ["max_fe=4", "model=configured-model"]
    assert env["BATRECYCLER_EOH_API_KEY"] == "test-key-not-a-real-secret"
    assert env["BATRECYCLER_EOH_BASE_URL"] == "https://llm.example/v1"
    assert env["UNCHANGED"] == "yes"


def test_explicit_eoh_settings_take_priority() -> None:
    overrides, env = prepare_eoh_runtime(
        ["model=explicit-model"],
        {
            "BATRECYCLER_EOH_API_KEYS": "explicit-one,explicit-two",
            "BATRECYCLER_EOH_BASE_URL": "https://explicit.example/v1",
        },
        _config(),
    )
    assert overrides == ["model=explicit-model"]
    assert "BATRECYCLER_EOH_API_KEY" not in env
    assert env["BATRECYCLER_EOH_API_KEYS"] == "explicit-one,explicit-two"
    assert env["BATRECYCLER_EOH_BASE_URL"] == "https://explicit.example/v1"


def main() -> None:
    test_resolved_config_is_added_to_child_only()
    test_explicit_eoh_settings_take_priority()
    print("test_eoh_runtime_config: PASS")


if __name__ == "__main__":
    main()

