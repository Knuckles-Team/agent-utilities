"""Token→USD cost estimation for the RLM benchmark (CONCEPT:AHE-3.32).

A small, explicit price table (USD per 1K tokens, blended input+output) keyed by a normalized
model id, so the scoreboard can report a cost column comparable to the paper's. Unknown models
fall back to a conservative default rather than reporting zero (which would read as "free").
"""

from __future__ import annotations

# USD per 1K tokens (blended). Coarse by design — the goal is order-of-magnitude cost parity
# against the paper ($0.99 vs $1.50-2.75), not billing accuracy.
_PRICE_PER_1K: dict[str, float] = {
    "gpt-5": 0.010,
    "gpt-5-mini": 0.002,
    "gpt-4o": 0.005,
    "gpt-4o-mini": 0.0004,
    "gemini-1.5-flash": 0.0005,
    "gemini-1.5-pro": 0.005,
    "qwen3-8b": 0.0002,
    "qwen3-coder-480b": 0.002,
}
_DEFAULT_PRICE_PER_1K = 0.003


def normalize_model(model_id: str) -> str:
    """Strip a ``provider:`` prefix and lowercase, so ``openai:gpt-4o-mini`` → ``gpt-4o-mini``."""
    mid = (model_id or "").split(":", 1)[-1].strip().lower()
    return mid


def price_per_1k(model_id: str) -> float:
    mid = normalize_model(model_id)
    if mid in _PRICE_PER_1K:
        return _PRICE_PER_1K[mid]
    # prefix match (e.g. "gpt-5-2025-..." → "gpt-5")
    for known, price in _PRICE_PER_1K.items():
        if mid.startswith(known):
            return price
    return _DEFAULT_PRICE_PER_1K


def estimate_cost_usd(tokens: int, model_id: str) -> float:
    """Estimate USD cost for ``tokens`` processed by ``model_id``."""
    return (max(0, tokens) / 1000.0) * price_per_1k(model_id)
