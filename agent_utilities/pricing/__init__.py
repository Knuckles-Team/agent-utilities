"""Unified model pricing catalog.

CONCEPT:ECO-4.40 — single source of token cost for the whole stack, replacing
the scattered hard-coded pricing dicts. Seeds from an embedded offline table
(zero-config, no network) and refreshes from LiteLLM via the daemon.
"""

from .catalog import ModelPricing, PricingCatalog, get_pricing_catalog
from .fallback import FALLBACK_VERSION, fallback_pricing
from .litellm import LITELLM_URL, fetch_litellm_pricing, parse_litellm_pricing
from .normalize import normalize_model_name, resolve
from .store import refresh_catalog

__all__ = [
    "FALLBACK_VERSION",
    "LITELLM_URL",
    "ModelPricing",
    "PricingCatalog",
    "fallback_pricing",
    "fetch_litellm_pricing",
    "get_pricing_catalog",
    "normalize_model_name",
    "parse_litellm_pricing",
    "refresh_catalog",
    "resolve",
]
