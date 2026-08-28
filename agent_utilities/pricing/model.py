"""The ``ModelPricing`` value type — the leaf of the pricing package.

CONCEPT:AU-ECO.toolkit.model-pricing-catalog — Unified model pricing catalog.

★ WHY THIS MODULE EXISTS (BUG-CX-004 / WD10-B-004)
``ModelPricing`` used to live in :mod:`agent_utilities.pricing.catalog`, which
imports :mod:`agent_utilities.pricing.fallback` at module scope to seed itself —
while ``fallback`` needs ``ModelPricing`` to build its rows. That is a genuine
circular dependency, and it was previously papered over twice in one 62-line
file: an ``if TYPE_CHECKING:`` import for the annotation plus a function-local
import for the constructor.

Holding the shared *type* in a dependency-free leaf that both sides import is
the honest fix: ``catalog -> model`` and ``fallback -> model`` are both forward
edges and the cycle is gone. Keep this module free of intra-package imports —
its whole job is to have no outgoing edges.

``catalog`` re-exports ``ModelPricing`` so every existing
``from agent_utilities.pricing.catalog import ModelPricing`` keeps working.
"""

from __future__ import annotations

from pydantic import BaseModel


class ModelPricing(BaseModel):
    """Per-model token pricing in USD per million tokens."""

    model_pattern: str
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    cache_creation_per_mtok: float = 0.0
    cache_read_per_mtok: float = 0.0

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
