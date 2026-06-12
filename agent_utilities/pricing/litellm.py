"""LiteLLM pricing fetch + parse.

CONCEPT:ECO-4.40 — Unified model pricing catalog.

Port of agentsview ``internal/pricing/litellm.go``. Downloads the BerriAI
LiteLLM pricing JSON and converts per-token costs to per-million-token
``ModelPricing`` entries. Entries missing both input and output cost are
skipped. Network is optional: callers fall back to the embedded table.
"""

from __future__ import annotations

import json

import httpx

from .catalog import ModelPricing

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

_PER_MTOK = 1_000_000


def parse_litellm_pricing(data: bytes | str) -> list[ModelPricing]:
    """Parse the LiteLLM JSON map into ``ModelPricing`` entries."""
    raw = json.loads(data)
    prices: list[ModelPricing] = []
    for model, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        input_cost = entry.get("input_cost_per_token")
        output_cost = entry.get("output_cost_per_token")
        if input_cost is None and output_cost is None:
            continue
        cache_creation = entry.get("cache_creation_input_token_cost")
        cache_read = entry.get("cache_read_input_token_cost")
        prices.append(
            ModelPricing(
                model_pattern=model,
                input_per_mtok=(input_cost or 0.0) * _PER_MTOK,
                output_per_mtok=(output_cost or 0.0) * _PER_MTOK,
                cache_creation_per_mtok=(cache_creation or 0.0) * _PER_MTOK,
                cache_read_per_mtok=(cache_read or 0.0) * _PER_MTOK,
            )
        )
    return prices


def fetch_litellm_pricing(
    url: str = LITELLM_URL, *, timeout: float = 30.0
) -> list[ModelPricing]:
    """Download and parse the LiteLLM pricing JSON.

    Raises on network/HTTP/parse errors — callers should catch and keep the
    offline fallback so a refresh failure is never fatal.
    """
    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return parse_litellm_pricing(resp.content)
