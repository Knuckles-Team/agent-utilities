"""Catalog-backed benchmark cost (CONCEPT:AU-AHE.rlm.long-context-benchmark)."""

from __future__ import annotations


def estimate_cost_usd(tokens: int, model_id: str) -> float | None:
    """Price total tokens as input tokens, or return ``None`` when unpriced."""
    from agent_utilities.pricing import get_pricing_catalog

    cost, _priced = get_pricing_catalog().cost_for(
        model_id, input_tokens=max(0, tokens)
    )
    return cost
