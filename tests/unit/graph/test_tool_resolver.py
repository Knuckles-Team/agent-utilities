#!/usr/bin/python
"""Tool gap-fill resolver for workflow materialization (a-keystone).

CONCEPT:ECO-4.0
"""

import pytest

from agent_utilities.graph.tool_resolver import resolve_agent_tools, resolve_tools

pytestmark = pytest.mark.concept("ECO-4.0")


def test_all_available_pass_through():
    r = resolve_tools(["git", "web"], available={"git", "web", "slack"})
    assert r.resolved == ["git", "web"]
    assert not r.filled and not r.missing and not r.has_gaps


def test_unknown_availability_passes_through_deduped():
    r = resolve_tools(["git", "git", "web"], available=None)
    assert r.resolved == ["git", "web"]  # deduped, nothing marked missing
    assert not r.missing


def test_gap_filled_via_capability_substitute():
    # workflow wants 'gitlab_pr' (not bound); a capability substitute is available
    r = resolve_tools(
        ["gitlab_pr"],
        available={"github_pr", "git"},
        designate_fn=lambda cap: ["github_pr"],  # capability index suggests a sub
    )
    assert r.filled == {"gitlab_pr": "github_pr"}
    assert r.resolved == ["github_pr"]
    assert not r.missing


def test_missing_when_no_substitute():
    r = resolve_tools(
        ["gitlab_pr"],
        available={"git"},
        designate_fn=lambda cap: ["nonexistent"],  # suggestion not available
    )
    assert r.missing == ["gitlab_pr"] and r.has_gaps
    assert r.resolved == []


def test_substitute_filtered_to_available():
    r = resolve_tools(
        ["x"],
        available={"a", "b"},
        designate_fn=lambda cap: ["z", "b"],  # z unavailable, b available → pick b
    )
    assert r.filled == {"x": "b"}


def test_resolve_agent_tools_defensive_on_bare_engine():
    # An engine without capability_index/available_tool_tags → passthrough, never raises.
    r = resolve_agent_tools(object(), ["git", "web"])
    assert r.resolved == ["git", "web"]


def test_empty_and_none_inputs():
    assert resolve_tools(None, available={"a"}).resolved == []
    assert resolve_tools([], available=None).resolved == []
