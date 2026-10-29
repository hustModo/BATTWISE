"""Regression tests for fields sent by the plan UI."""
from __future__ import annotations

from app.backend.api.rest import ActionBody


def test_plan_action_preserves_action() -> None:
    body = ActionBody.model_validate(
        {"type": "plan_action", "action": "approve"}
    )
    assert body.model_dump(exclude_none=True) == {
        "type": "plan_action",
        "action": "approve",
    }


def test_plan_update_preserves_step_and_params() -> None:
    body = ActionBody.model_validate(
        {
            "type": "plan_update_step",
            "step_id": 2,
            "params": {"max_fe": 12, "clusters": [0, 1]},
        }
    )
    assert body.model_dump(exclude_none=True) == {
        "type": "plan_update_step",
        "step_id": 2,
        "params": {"max_fe": 12, "clusters": [0, 1]},
    }


def main() -> None:
    test_plan_action_preserves_action()
    test_plan_update_preserves_step_and_params()
    print("test_action_body_contract: PASS")


if __name__ == "__main__":
    main()

