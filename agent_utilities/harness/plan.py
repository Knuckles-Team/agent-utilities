"""The model-edited scaffold of one run, separate from execution-plan prompts.

This is transient harness state, not a planner, durable record or public reader.
The run loop owns each instance and shares it between updates and injection.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PlanItem(BaseModel):
    """One immutable scaffold item in model-supplied order."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )

    text: str = Field(min_length=1)
    status: Literal["pending", "in_progress", "done"]


class PlanState:
    """One run's ordered scaffold; validate a complete replacement before applying it."""

    def __init__(self) -> None:
        self._items: tuple[PlanItem, ...] = ()

    @property
    def items(self) -> tuple[PlanItem, ...]:
        """An immutable view that remains stable after later replacements."""
        return self._items

    def replace(self, items: Sequence[PlanItem]) -> tuple[PlanItem, ...]:
        """Replace all items atomically; invalid input leaves the old list intact."""
        replacement = tuple(PlanItem.model_validate(item) for item in items)
        self._items = replacement
        return replacement
