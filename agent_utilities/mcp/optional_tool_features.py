#!/usr/bin/python
"""Dependency-free declarations for optional Graph-OS tool families.

Optional registrars can import heavy feature dependencies and therefore cannot
be the source used by lean catalog generators.  These frozen declarations keep
only the feature name, REST route, and public action vocabulary needed to
generate and validate the canonical manifest.  Execution schemas still come
from the real registrar when that feature is installed.
"""

from __future__ import annotations

from types import MappingProxyType

OPTIONAL_TOOL_FEATURES = MappingProxyType({"quant": "finance"})
OPTIONAL_TOOL_ROUTES = MappingProxyType({"quant": "/quant"})
OPTIONAL_TOOL_ACTIONS = MappingProxyType(
    {
        "quant": (
            "analyze",
            "balances",
            "cancel_order",
            "debate",
            "ensemble_predict",
            "fundamentals",
            "historical",
            "optimize",
            "order_book",
            "positions",
            "regime",
            "risk_metrics",
            "status",
            "submit_order",
        )
    }
)
SUPPORTED_FEATURES: frozenset[str] = frozenset(OPTIONAL_TOOL_FEATURES.values())

__all__ = [
    "OPTIONAL_TOOL_ACTIONS",
    "OPTIONAL_TOOL_FEATURES",
    "OPTIONAL_TOOL_ROUTES",
    "SUPPORTED_FEATURES",
]
