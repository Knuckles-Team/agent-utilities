"""Model pricing catalog — the single source of token cost.

CONCEPT:AU-ECO.toolkit.model-pricing-catalog — Unified model pricing catalog.

Replaces scattered embedded price dicts. Prices are stored per-million
tokens. The process-wide catalog loads an operator-owned, versioned local
document when configured and may be refreshed from LiteLLM by the daemon (see
``litellm`` and ``store``). Resolution uses ``normalize.resolve``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

# ``ModelPricing`` lives in the leaf module ``.model`` to keep the schema free
# of singleton-composition imports. It is re-exported here for callers.
from .model import ModelPricing
from .normalize import resolve

__all__ = [
    "ModelPricing",
    "PricingCatalog",
    "get_pricing_catalog",
    "reset_pricing_catalog",
]


class PricingCatalog:
    """In-memory catalog of ``ModelPricing`` keyed by model pattern.

    Process-wide singleton via :func:`get_pricing_catalog`. An absent local
    catalog is an explicitly empty authority; the daemon can overlay discovered
    rates later (exact patterns win on later merge).
    """

    def __init__(
        self,
        entries: Iterable[ModelPricing] = (),
    ) -> None:
        self._by_pattern: dict[str, ModelPricing] = {}
        self.version: str = "unconfigured"
        self.merge(entries)

    @classmethod
    def load_from_file(cls, path: str | Path) -> PricingCatalog:
        """Load a versioned operator catalog for offline or historical pricing."""
        catalog_path = Path(path)
        with catalog_path.open("rb") as stream:
            payload = stream.read(1_000_001)
        if len(payload) > 1_000_000:
            raise ValueError("pricing catalog exceeds the size limit")
        raw = json.loads(payload)
        if not isinstance(raw, dict) or set(raw) != {"version", "models"}:
            raise ValueError("pricing catalog must contain version and models")
        version = raw["version"]
        models = raw["models"]
        if not isinstance(version, str) or not version.strip():
            raise ValueError("pricing catalog version must be non-empty")
        if not isinstance(models, list):
            raise ValueError("pricing catalog models must be a list")
        entries = [ModelPricing.model_validate(item) for item in models]
        catalog = cls(entries)
        catalog.version = version
        return catalog

    def merge(self, entries: Iterable[ModelPricing]) -> None:
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
    """Process-wide pricing catalog composed from operator configuration."""
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _configured_pricing_catalog()
    return _CATALOG


def reset_pricing_catalog() -> None:
    """Invalidate the singleton after an operator configuration transition."""
    global _CATALOG
    _CATALOG = None


def _configured_pricing_catalog() -> PricingCatalog:
    from agent_utilities.core.config import config

    if config.pricing_catalog_path:
        return PricingCatalog.load_from_file(config.pricing_catalog_path)
    return PricingCatalog()
