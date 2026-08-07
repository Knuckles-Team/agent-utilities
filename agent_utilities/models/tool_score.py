#!/usr/bin/python
from __future__ import annotations

"""The ONE canonical Tool relevance-score domain and its legacy boundary.

D-CDX-53/54 (see ``reports/deferred/``): a Tool's relevance/quality score is
a deterministic integer point in ``0..100``. Early writers instead persisted
a normalized float in ``[0, 1]``. Two independent models —
:class:`agent_utilities.models.mcp.MCPToolInfo` (the in-process registry
view) and :class:`agent_utilities.models.knowledge_graph.ToolNode` (the
graph-persisted node) — must apply the EXACT same boundary or a value can
round-trip differently depending which model last touched it. This module is
that single source of truth so the two validators (and the
``resync_tool_relevance_scores`` data migration, CONCEPT:AU-KG) can never
drift apart again.

Only a float in the legacy ``[0, 1]`` range is treated as a legacy score and
rescaled. Every other out-of-range/ambiguous value (negative, >100, bool,
numeric string, non-legacy fractional float such as ``1.9``) is left
untouched so the caller's own strict validation (``ge=0, le=100,
strict=True``) rejects it outright — corrupt rows are quarantined at the
boundary, never silently coerced into a plausible-looking canonical value.
"""

from typing import Any

#: Inclusive bounds of the canonical integer point domain.
CANONICAL_MIN = 0
CANONICAL_MAX = 100


def normalize_legacy_relevance_score(value: Any) -> Any:
    """Rescale a legacy ``[0, 1]`` float score to canonical ``0..100`` points.

    Returns ``value`` unchanged for every other input — including ints,
    bools, strings, and floats outside ``[0, 1]`` — so a strict downstream
    validator can reject what this boundary does not recognize as the one
    known legacy convention.
    """
    if isinstance(value, float) and 0.0 <= value <= 1.0:
        return round(value * 100)
    return value


def is_canonical_relevance_score(value: Any) -> bool:
    """True when ``value`` is already a valid canonical score: a plain
    ``int`` (not ``bool`` — ``bool`` is an ``int`` subtype in Python but is
    never a valid score) in ``[CANONICAL_MIN, CANONICAL_MAX]``."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and CANONICAL_MIN <= value <= CANONICAL_MAX
    )
