"""Declared guardrail profiles for error-budget throttling (AU-SEC-R007).

An operator declares, per protected resource (a GraphOS fleet child today), the
AIMD policy its EG ``CapacityCell`` throttle runs under and -- optionally -- the
bounds a profile may evolve within. The bounds are a ladder of ``levels``
profiles from ``loosest`` (level 0) to ``tightest`` (level ``levels - 1``),
interpolated per knob, so "tighter" and "looser" are total orders a reviewer
can read, not judgements a model makes.

Only the four AIMD knobs move; ``min_samples``, ``floor`` and ``cooldown_ms``
stay exactly as declared. A cell whose live policy is not on its ladder (an
operator set it by hand) is *unmanaged*: evolution leaves it alone.

This module is pure: it declares, validates and plans. Applying a plan is
:mod:`agent_utilities.security.guardrail_evolution`'s job.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "KNOBS",
    "ErrorBudgetDeclaration",
    "GuardrailBounds",
    "ProfileMove",
    "ProfilePlan",
    "ThrottleKnobs",
    "ThrottlePolicy",
    "level_of",
    "plan_move",
]

PPM = 1_000_000
#: The knobs a profile moves; every one is "lower is tighter".
KNOBS = ("error_budget_ppm", "recovery_ppm", "decrease_per_mille", "increase_step")


class _Declared(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ThrottleKnobs(_Declared):
    """The four AIMD knobs one profile level sets."""

    error_budget_ppm: int = Field(ge=0, lt=PPM)
    recovery_ppm: int = Field(ge=0, lt=PPM)
    decrease_per_mille: int = Field(ge=1, le=999)
    increase_step: int = Field(ge=1, le=1_000_000)

    @model_validator(mode="after")
    def _ordered(self) -> ThrottleKnobs:
        if self.recovery_ppm > self.error_budget_ppm:
            raise ValueError("recovery_ppm must not exceed error_budget_ppm")
        return self


class ThrottlePolicy(ThrottleKnobs):
    """EG's ``CapacityThrottlePolicy``, validated the way the engine does."""

    min_samples: int = Field(ge=1, le=1_000_000_000)
    floor: int = Field(ge=1)
    cooldown_ms: int = Field(ge=0, le=24 * 60 * 60 * 1000)

    def with_knobs(self, knobs: ThrottleKnobs) -> ThrottlePolicy:
        return self.model_copy(update=knobs.model_dump())


def _interpolate(loosest: int, tightest: int, *, level: int, top: int) -> int:
    return loosest - (loosest - tightest) * level // top


class GuardrailBounds(_Declared):
    """The ladder a profile may move along; level 0 is the loosest."""

    loosest: ThrottleKnobs
    tightest: ThrottleKnobs
    levels: int = Field(ge=2, le=16)

    @model_validator(mode="after")
    def _a_ladder(self) -> GuardrailBounds:
        loose, tight = self.loosest.model_dump(), self.tightest.model_dump()
        if any(tight[knob] > loose[knob] for knob in KNOBS):
            raise ValueError("every tightest knob must be at or below its loosest")
        for level in range(self.levels):
            self.knobs_at(level)
        return self

    def knobs_at(self, level: int) -> ThrottleKnobs:
        """The knobs of ``level`` (raises when a level is not a valid policy)."""
        loose, tight = self.loosest.model_dump(), self.tightest.model_dump()
        top = self.levels - 1
        return ThrottleKnobs.model_validate(
            {
                knob: _interpolate(loose[knob], tight[knob], level=level, top=top)
                for knob in KNOBS
            }
        )


class ErrorBudgetDeclaration(_Declared):
    """One resource's declared throttle: capacity, policy and optional bounds."""

    capacity: int = Field(ge=1, le=1_000_000)
    policy: ThrottlePolicy
    bounds: GuardrailBounds | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ErrorBudgetDeclaration:
        if self.policy.floor > self.capacity:
            raise ValueError("the throttle floor must not exceed the capacity")
        if self.bounds is not None and level_of(self.bounds, self.policy) is None:
            raise ValueError("the declared policy must sit on its declared ladder")
        return self


class ProfileMove(StrEnum):
    """What a profile evolution step proposes."""

    HOLD = "hold"
    TIGHTEN = "tighten"
    LOOSEN = "loosen"


_STEP = {ProfileMove.HOLD: 0, ProfileMove.TIGHTEN: 1, ProfileMove.LOOSEN: -1}


@dataclass(frozen=True, slots=True)
class ProfilePlan:
    """One concrete, bounded profile change for one cell."""

    cell_id: str
    move: ProfileMove
    from_level: int
    to_level: int
    epoch: int
    policy: ThrottlePolicy

    def approval_params(self) -> dict[str, Any]:
        """The exact change an approval binds to, including the cell epoch.

        The epoch advances on every applied change, so an approval granted for
        this plan can never be replayed against a later, identical-looking move.
        """
        return {
            "cell_id": self.cell_id,
            "epoch": self.epoch,
            "from_level": self.from_level,
            "to_level": self.to_level,
            "policy": self.policy.model_dump(),
        }


def level_of(bounds: GuardrailBounds, policy: ThrottleKnobs) -> int | None:
    """The ladder level ``policy``'s knobs sit on, or ``None`` (unmanaged)."""
    knobs = ThrottleKnobs.model_validate(policy.model_dump(include=set(KNOBS)))
    for level in range(bounds.levels):
        if bounds.knobs_at(level) == knobs:
            return level
    return None


def plan_move(
    bounds: GuardrailBounds, cell: Mapping[str, Any], move: ProfileMove
) -> ProfilePlan | None:
    """The bounded plan for ``move`` on ``cell``, or ``None`` when it cannot move.

    ``None`` means: a hold, an unthrottled or unmanaged cell, or a step past
    either end of the ladder -- never a clamp that silently does less.
    """
    throttle = cell.get("throttle")
    if move is ProfileMove.HOLD or not isinstance(throttle, Mapping):
        return None
    current = ThrottlePolicy.model_validate(throttle["policy"])
    level = level_of(bounds, current)
    if level is None:
        return None
    target = level + _STEP[move]
    if not 0 <= target < bounds.levels:
        return None
    return ProfilePlan(
        cell_id=str(cell["cell_id"]),
        move=move,
        from_level=level,
        to_level=target,
        epoch=int(cell["epoch"]),
        policy=current.with_knobs(bounds.knobs_at(target)),
    )
