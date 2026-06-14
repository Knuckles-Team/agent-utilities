"""CONCEPT:ECO-4.45 — capability binding (Layer 3) + mcp prefix/alias boundary (Layer 2)."""

from __future__ import annotations

from agent_utilities.agent.capability_resolver import (
    build_alias_map,
    known_capabilities,
    register_capability_tools,
    resolve_capabilities,
    resolve_mcp_name,
    strip_mcp_prefix,
)


def test_capability_resolves_to_tool_functions():
    fns = resolve_capabilities(["code-intelligence"])
    names = [f.__name__ for f in fns]
    assert "who_calls" in names and "impacted_tests" in names
    assert "run_command" not in names  # not a code-intelligence tool


def test_software_engineering_is_the_full_swe_surface():
    fns = resolve_capabilities(["software-engineering", "web-browsing"])
    names = {f.__name__ for f in fns}
    assert {
        "find_definition",
        "edit_file",
        "run_tests",
        "run_command",
        "browse",
    } <= names


def test_resolution_dedups_across_overlapping_tags():
    fns = resolve_capabilities(["file-editing", "software-engineering"])
    names = [f.__name__ for f in fns]
    assert len(names) == len(set(names))  # edit_file/read_file not duplicated


def test_unknown_capability_resolves_to_nothing_without_kg():
    assert resolve_capabilities(["totally-unknown-intent"]) == []


def test_known_capabilities_surface():
    caps = known_capabilities()
    assert "software-engineering" in caps and "test-execution" in caps


def test_register_capability_tools_attaches_to_agent():
    registered = []

    class _Agent:
        def tool(self, fn):
            registered.append(fn.__name__)

    n = register_capability_tools(_Agent(), ["test-execution", "shell-execution"])
    assert n == 2
    assert set(registered) == {"run_tests", "run_command"}


# ── Layer 2: prefix/alias ─────────────────────────────────────────────────────


def test_alias_map_resolves_canonical_to_prefixed():
    amap = build_alias_map({"go": ["graph_query"], "gitlab": ["create_issue"]})
    assert resolve_mcp_name("graph_query", amap) == "go__graph_query"
    assert resolve_mcp_name("create_issue", amap) == "gitlab__create_issue"


def test_alias_resolution_is_idempotent_on_prefixed_input():
    amap = build_alias_map({"go": ["graph_query"]})
    assert resolve_mcp_name("go__graph_query", amap) == "go__graph_query"


def test_strip_prefix():
    assert strip_mcp_prefix("go__graph_query") == "graph_query"
    assert strip_mcp_prefix("graph_query") == "graph_query"
