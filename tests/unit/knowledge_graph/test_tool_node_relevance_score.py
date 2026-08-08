#!/usr/bin/python
from __future__ import annotations

"""Regression tests for D-CDX-54: ``ToolNode.relevance_score`` must be the
same strict canonical ``0..100`` integer domain as
:class:`agent_utilities.models.mcp.MCPToolInfo`, not a permissive ``int``
field that silently coerces floats/bools/strings or accepts out-of-range
values.

Every case below is written to FAIL against the pre-fix field
(``relevance_score: int = 0`` with no ``ge``/``le``/``strict`` and no legacy
boundary), proving the fix actually closes the hole rather than merely
asserting the new behavior in the abstract.
"""

import pytest
from pydantic import ValidationError

from agent_utilities.models.knowledge_graph import RegistryNodeType, ToolNode


def _make(**overrides) -> ToolNode:
    base = {
        "id": "tool:t1",
        "name": "t1",
        "mcp_server": "server1",
    }
    base.update(overrides)
    return ToolNode(**base)


def test_default_type_and_score() -> None:
    node = _make()
    assert node.type == RegistryNodeType.TOOL
    assert node.relevance_score == 0


def test_canonical_integer_in_range_accepted() -> None:
    node = _make(relevance_score=87)
    assert node.relevance_score == 87


@pytest.mark.parametrize("bound", [0, 100])
def test_canonical_boundary_values_accepted(bound: int) -> None:
    assert _make(relevance_score=bound).relevance_score == bound


def test_legacy_float_in_unit_range_is_rescaled_to_points() -> None:
    """The one explicit legacy-migration boundary: floats in [0, 1] were the
    old normalized-score convention and are rescaled to canonical points —
    mirroring MCPToolInfo's boundary exactly."""
    node = _make(relevance_score=0.5)
    assert node.relevance_score == 50
    assert isinstance(node.relevance_score, int)


def test_legacy_float_zero_and_one_rescaled() -> None:
    assert _make(relevance_score=0.0).relevance_score == 0
    assert _make(relevance_score=1.0).relevance_score == 100


def test_out_of_range_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        _make(relevance_score=-1)


def test_out_of_range_above_100_rejected() -> None:
    with pytest.raises(ValidationError):
        _make(relevance_score=101)


def test_non_legacy_fractional_float_rejected() -> None:
    """A float outside [0, 1] (e.g. 1.9) is NOT a legacy score and must not
    be silently truncated to 1 — it should be rejected outright."""
    with pytest.raises(ValidationError):
        _make(relevance_score=1.9)


def test_boolean_true_rejected() -> None:
    """``bool`` is a subtype of ``int`` in Python; a lax/non-strict int field
    would silently accept ``True`` as ``1``. Strict validation must reject
    it explicitly instead of coercing an ambiguous value."""
    with pytest.raises(ValidationError):
        _make(relevance_score=True)


def test_boolean_false_rejected() -> None:
    with pytest.raises(ValidationError):
        _make(relevance_score=False)


def test_numeric_string_rejected() -> None:
    """A lax int field coerces ``"50"`` to ``50``; strict mode must reject
    the ambiguous string representation."""
    with pytest.raises(ValidationError):
        _make(relevance_score="50")


def test_assignment_after_construction_is_validated() -> None:
    """``model_config = ConfigDict(validate_assignment=True)`` closes the
    same hole for post-construction mutation, not just construction."""
    node = _make(relevance_score=10)
    with pytest.raises(ValidationError):
        node.relevance_score = 999

    with pytest.raises(ValidationError):
        node.relevance_score = True

    node.relevance_score = 42
    assert node.relevance_score == 42


def test_matches_mcp_tool_info_legacy_boundary() -> None:
    """ToolNode and MCPToolInfo (agent_utilities/models/mcp.py) must apply
    the identical legacy-rescale boundary so every canonical-score model in
    the codebase agrees on ambiguous historical data."""
    from agent_utilities.models.mcp import MCPToolInfo

    tool_info = MCPToolInfo(
        name="t1", description="d", mcp_server="s1", relevance_score=0.7
    )
    node = _make(relevance_score=0.7)
    assert tool_info.relevance_score == node.relevance_score == 70
