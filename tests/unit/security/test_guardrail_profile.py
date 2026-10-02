"""AU-SEC requirement 007: a declared guardrail ladder -- bounded, ordered, and exact about what moves."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agent_utilities.security.guardrail_profile import (
    ErrorBudgetDeclaration,
    GuardrailBounds,
    ProfileMove,
    ThrottlePolicy,
    level_of,
    plan_move,
)

LOOSEST = {
    "error_budget_ppm": 50_000,
    "recovery_ppm": 10_000,
    "decrease_per_mille": 700,
    "increase_step": 4,
}
TIGHTEST = {
    "error_budget_ppm": 10_000,
    "recovery_ppm": 2_000,
    "decrease_per_mille": 300,
    "increase_step": 1,
}
FIXED = {"min_samples": 20, "floor": 1, "cooldown_ms": 30_000}


def bounds(levels: int = 5) -> GuardrailBounds:
    return GuardrailBounds.model_validate(
        {"loosest": LOOSEST, "tightest": TIGHTEST, "levels": levels}
    )


def policy_at(level: int, levels: int = 5) -> dict[str, Any]:
    return {**bounds(levels).knobs_at(level).model_dump(), **FIXED}


def cell(level: int, *, epoch: int = 7, levels: int = 5) -> dict[str, Any]:
    return {
        "cell_id": "fleet/child/github",
        "capacity": 8,
        "epoch": epoch,
        "throttle": {"policy": policy_at(level, levels), "ceiling": 8, "history": []},
    }


def test_the_ladder_runs_from_loosest_to_tightest() -> None:
    ladder = bounds()
    assert ladder.knobs_at(0).model_dump() == LOOSEST
    assert ladder.knobs_at(4).model_dump() == TIGHTEST
    budgets = [ladder.knobs_at(level).error_budget_ppm for level in range(5)]
    assert budgets == sorted(budgets, reverse=True), "each level is tighter"


def test_a_ladder_whose_tight_end_is_looser_is_refused() -> None:
    with pytest.raises(ValidationError):
        GuardrailBounds.model_validate(
            {"loosest": TIGHTEST, "tightest": LOOSEST, "levels": 3}
        )


def test_a_level_is_found_only_for_a_policy_on_the_ladder() -> None:
    ladder = bounds()
    assert level_of(ladder, ThrottlePolicy.model_validate(policy_at(2))) == 2
    hand_set = {**policy_at(2), "error_budget_ppm": 33_333}
    assert level_of(ladder, ThrottlePolicy.model_validate(hand_set)) is None


def test_a_tightening_plan_moves_only_the_knobs_and_binds_the_epoch() -> None:
    plan = plan_move(bounds(), cell(1), ProfileMove.TIGHTEN)
    assert plan is not None
    assert (plan.from_level, plan.to_level, plan.epoch) == (1, 2, 7)
    assert plan.policy.model_dump() == policy_at(2)
    assert plan.approval_params()["epoch"] == 7


@pytest.mark.parametrize(
    ("level", "move"),
    [(4, ProfileMove.TIGHTEN), (0, ProfileMove.LOOSEN), (2, ProfileMove.HOLD)],
)
def test_no_plan_leaves_the_declared_bounds(level: int, move: ProfileMove) -> None:
    assert plan_move(bounds(), cell(level), move) is None


def test_an_unthrottled_or_hand_set_cell_is_not_planned() -> None:
    assert (
        plan_move(bounds(), {"cell_id": "c", "epoch": 1}, ProfileMove.TIGHTEN) is None
    )
    hand = cell(1)
    hand["throttle"]["policy"]["recovery_ppm"] = 1
    assert plan_move(bounds(), hand, ProfileMove.TIGHTEN) is None


def test_a_declaration_must_start_on_its_ladder_and_under_capacity() -> None:
    declared = {"capacity": 8, "policy": policy_at(0), "bounds": bounds().model_dump()}
    assert ErrorBudgetDeclaration.model_validate(declared).bounds is not None
    with pytest.raises(ValidationError):
        ErrorBudgetDeclaration.model_validate(
            {**declared, "policy": {**policy_at(0), "error_budget_ppm": 49_999}}
        )
    with pytest.raises(ValidationError):
        ErrorBudgetDeclaration.model_validate(
            {**declared, "policy": {**policy_at(0), "floor": 9}}
        )
