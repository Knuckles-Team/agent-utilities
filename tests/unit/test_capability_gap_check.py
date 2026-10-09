"""AU-SEMANTIC-R027.1: capability-gap determination and its refusal behavior."""

from __future__ import annotations

import pytest

from agent_utilities.api.capability_gap_check import (
    CapabilityGapResult,
    CapabilityGapUnverified,
    EGSurface,
    EGSurfaceSearch,
)


def test_missing_verdict_refused_without_any_search() -> None:
    with pytest.raises(CapabilityGapUnverified):
        CapabilityGapResult.determine("foo_tool", searches=(), is_missing=True)


def test_search_record_refuses_empty_query() -> None:
    with pytest.raises(CapabilityGapUnverified):
        EGSurfaceSearch(surface=EGSurface.METHOD_CATALOG, query="   ")


def test_missing_verdict_accepted_with_recorded_eg_side_search() -> None:
    search = EGSurfaceSearch(
        surface=EGSurface.METHOD_CATALOG, query="list_capabilities"
    )
    result = CapabilityGapResult.determine(
        "foo_tool", searches=(search,), is_missing=True
    )
    assert result.is_missing is True
    assert result.searches == (search,)


def test_present_verdict_requires_no_search() -> None:
    result = CapabilityGapResult.determine("bar_tool", searches=(), is_missing=False)
    assert result.is_missing is False
