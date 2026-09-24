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

# The finance ``quant`` family left with agent-utilities' finance math
# (EH-423 / AUD-30): trend signals, flip alerts and live-order proposals are the
# graph-os ``graph_finance`` tool. No optional family is declared today.
OPTIONAL_TOOL_FEATURES: MappingProxyType[str, str] = MappingProxyType({})
OPTIONAL_TOOL_ROUTES: MappingProxyType[str, str] = MappingProxyType({})
OPTIONAL_TOOL_ACTIONS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType({})
SUPPORTED_FEATURES: frozenset[str] = frozenset(OPTIONAL_TOOL_FEATURES.values())

__all__ = [
    "OPTIONAL_TOOL_ACTIONS",
    "OPTIONAL_TOOL_FEATURES",
    "OPTIONAL_TOOL_ROUTES",
    "SUPPORTED_FEATURES",
]
