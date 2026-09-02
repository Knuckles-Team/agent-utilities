"""The ``ModelPricing`` value type — the leaf of the pricing package.

CONCEPT:AU-ECO.toolkit.model-pricing-catalog — Unified model pricing catalog.

Holding the shared *type* in a dependency-free leaf keeps catalog composition
free of import cycles. Keep this module free of intra-package imports.

``catalog`` re-exports ``ModelPricing`` so every existing
``from agent_utilities.pricing.catalog import ModelPricing`` keeps working.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict


class ModelPricing(BaseModel):
    """Per-model token pricing in USD per million tokens."""

    model_pattern: str
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    cache_creation_per_mtok: float = 0.0
    cache_read_per_mtok: float = 0.0

    model_config = ConfigDict(extra="forbid")

    def cost_usd(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """Compute cost from token counts using this model's per-Mtok rates."""
        per = 1_000_000
        return (
            input_tokens / per * self.input_per_mtok
            + output_tokens / per * self.output_per_mtok
            + cache_creation_tokens / per * self.cache_creation_per_mtok
            + cache_read_tokens / per * self.cache_read_per_mtok
        )


def total_known_cost(costs: Iterable[float | None]) -> float | None:
    """Sum costs only when every monetary value is known."""
    values = list(costs)
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)
