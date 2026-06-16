"""Tests for the graph_code_nav Cypher builder (CONCEPT:KG-2.9g).

Pure-function tests: the templates are validated without a live engine.
"""

from __future__ import annotations

import pytest

from agent_utilities.mcp.tools.query_tools import build_code_nav_query


def test_find_definition_matches_by_name():
    cypher, params = build_code_nav_query(action="find_definition", symbol="run")
    assert "MATCH (c:Code)" in cypher
    assert "c.name = $symbol" in cypher
    assert params == {"symbol": "run"}


def test_node_id_overrides_symbol():
    cypher, params = build_code_nav_query(
        action="find_definition", symbol="run", node_id="gitlab:gl:1:symbol:abc"
    )
    assert "c.id = $node_id" in cypher
    assert "name = $symbol" not in cypher
    assert params["node_id"] == "gitlab:gl:1:symbol:abc"


def test_source_system_scope_applied():
    cypher, params = build_code_nav_query(
        action="find_definition", symbol="run", source_system="gitlab:gitlab.com"
    )
    assert "c.source_system = $src" in cypher
    assert params["src"] == "gitlab:gitlab.com"


def test_find_references_traverses_incoming_calls():
    cypher, _ = build_code_nav_query(action="find_references", symbol="shared")
    assert "(caller:Code)-[:calls]->(def:Code)" in cypher
    assert "def.name = $symbol" in cypher
    assert "RETURN DISTINCT caller.id" in cypher


def test_trace_call_graph_is_downstream_and_depth_bounded():
    cypher, _ = build_code_nav_query(action="trace_call_graph", symbol="run", depth=4)
    assert "(s:Code)-[:calls*1..4]->(callee:Code)" in cypher
    assert "s.name = $symbol" in cypher


def test_impact_of_change_is_upstream():
    cypher, _ = build_code_nav_query(action="impact_of_change", symbol="shared", depth=2)
    assert "(caller:Code)-[:calls*1..2]->(t:Code)" in cypher
    assert "t.name = $symbol" in cypher


def test_depth_clamped_and_limit_inlined():
    cypher, _ = build_code_nav_query(
        action="trace_call_graph", symbol="x", depth=999, limit=10
    )
    assert "*1..10]" in cypher  # clamped to 10
    assert "LIMIT 10" in cypher


def test_unknown_action_rejected():
    with pytest.raises(ValueError, match="unknown action"):
        build_code_nav_query(action="delete_everything", symbol="x")


def test_requires_symbol_or_node_id():
    with pytest.raises(ValueError, match="symbol' or 'node_id"):
        build_code_nav_query(action="find_definition")
