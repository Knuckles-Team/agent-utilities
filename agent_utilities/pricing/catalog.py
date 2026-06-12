"""Model pricing catalog — the single source of token cost.

CONCEPT:ECO-4.40 — Unified model pricing catalog.

Replaces the scattered hard-coded dicts (``models/usage.py`` defaults, the
agent-terminal-ui ``DEFAULT_PRICING`` table). Prices are stored per-million
tokens. The catalog seeds from the embedded offline ``fallback`` table (works
with no network) and is refreshed from LiteLLM by the daemon (see ``litellm``
and ``store``). Resolution uses the load-bearing ``normalize.resolve``.
"""

from __future__ import annotations

from pydantic import BaseModel

from .fallback import FALLBACK_VERSION, fallback_pricing
from .normalize import resolve


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


class PricingCatalog:
    """In-memory catalog of ``ModelPricing`` keyed by model pattern.

    Process-wide singleton via :func:`get_pricing_catalog`. Seeds from the
    offline fallback so it is always usable with zero configuration; the daemon
    overlays LiteLLM rates on top (exact patterns win on later merge).
    """

    def __init__(self, entries: list[ModelPricing] | None = None) -> None:
        self._by_pattern: dict[str, ModelPricing] = {}
        self.version: str = FALLBACK_VERSION
        self.seed_fallback()
        if entries:
            self.merge(entries)

    def seed_fallback(self) -> None:
        for entry in fallback_pricing():
            self._by_pattern[entry.model_pattern] = entry

    def merge(self, entries: list[ModelPricing]) -> None:
        """Overlay ``entries`` (e.g. from LiteLLM) onto the catalog."""
        for entry in entries:
            self._by_pattern[entry.model_pattern] = entry

    def resolve(self, model: str) -> ModelPricing | None:
        """Resolve a model id to its pricing, or ``None`` when unpriced."""
        if not model:
            return None
        value, found = resolve(self._by_pattern, model)
        return value if found else None

    def cost_for(
        self,
        model: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> tuple[float | None, bool]:
        """Return ``(cost_usd, priced)``. ``priced`` is False for unknown models."""
        pricing = self.resolve(model)
        if pricing is None:
            return None, False
        return (
            pricing.cost_usd(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
            ),
            True,
        )

    def __len__(self) -> int:
        return len(self._by_pattern)


_CATALOG: PricingCatalog | None = None


def get_pricing_catalog() -> PricingCatalog:
    """Process-wide pricing catalog (seeded from the offline fallback)."""
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = PricingCatalog()
    return _CATALOG
