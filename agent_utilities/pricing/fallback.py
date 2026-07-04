"""Embedded offline pricing fallback.

CONCEPT:AU-ECO.toolkit.model-pricing-catalog — Unified model pricing catalog.

Port of agentsview ``internal/pricing/fallback.go``. Used so cost computation
works with zero network access. ``FALLBACK_VERSION`` is bumped whenever rates
change so the startup seeder re-upserts. Prices in USD per million tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .catalog import ModelPricing

# Bump whenever the rates below change.
FALLBACK_VERSION = "2026-06-10"

# (pattern, input, output, cache_creation, cache_read) per million tokens.
_FALLBACK: list[tuple[str, float, float, float, float]] = [
    # Current model names (Claude Code / Codex)
    ("claude-sonnet-4-6", 3.0, 15.0, 3.75, 0.30),
    ("claude-opus-4-6", 5.0, 25.0, 6.25, 0.50),
    ("claude-opus-4-7", 5.0, 25.0, 6.25, 0.50),
    # Opus 4.8 launched at the 4.6/4.7 rates; not yet in LiteLLM.
    ("claude-opus-4-8", 5.0, 25.0, 6.25, 0.50),
    # Fable 5 launched at double the Opus 4.8 rates; not yet in LiteLLM.
    ("claude-fable-5", 10.0, 50.0, 12.50, 1.0),
    ("claude-haiku-4-5-20251001", 1.0, 5.0, 1.25, 0.10),
    # Codex / OpenAI models
    ("gpt-5.5", 5.0, 30.0, 0.0, 0.50),
    ("gpt-5.4", 2.50, 15.0, 0.0, 0.0),
    ("gpt-5.2-codex", 1.75, 14.0, 0.0, 0.0),
    ("gpt-5.3-codex", 1.75, 14.0, 0.0, 0.0),
    ("gpt-5.4-mini", 0.75, 4.50, 0.0, 0.0),
    ("gpt-5.4-nano", 0.20, 1.25, 0.0, 0.0),
    ("gpt-5.1-codex-max", 1.25, 10.0, 0.0, 0.0),
    # Older model names (still in some session logs)
    ("claude-sonnet-4-20250514", 3.0, 15.0, 3.75, 0.30),
    ("claude-sonnet-4-5-20250514", 3.0, 15.0, 3.75, 0.30),
    ("claude-opus-4-20250514", 15.0, 75.0, 18.75, 1.50),
    ("claude-haiku-3-5-20241022", 0.80, 4.0, 1.0, 0.08),
    # Free OpenRouter model
    ("openrouter/owl-alpha", 0.0, 0.0, 0.0, 0.0),
]


def fallback_pricing() -> list[ModelPricing]:
    """Return the hardcoded offline pricing table."""
    from .catalog import ModelPricing

    return [
        ModelPricing(
            model_pattern=pattern,
            input_per_mtok=inp,
            output_per_mtok=out,
            cache_creation_per_mtok=cc,
            cache_read_per_mtok=cr,
        )
        for (pattern, inp, out, cc, cr) in _FALLBACK
    ]
