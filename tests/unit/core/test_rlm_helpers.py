"""Tests for RLM REPL helpers including OWL/KG integration.

CONCEPT:AU-007 — RLM Helpers (OWL × KG Integration)
"""

from unittest.mock import MagicMock

import pytest

from agent_utilities.rlm.repl import RLMEnvironment


@pytest.mark.asyncio
async def test_rlm_helpers():
    # Mock graph_deps and knowledge_engine
    mock_engine = MagicMock()
    mock_engine.retrieve_orthogonal_context = MagicMock(return_value={"semantic": ["test"]})
    mock_engine.query_cypher = MagicMock(return_value=[{"id": 1}])

    mock_deps = MagicMock()
    mock_deps.knowledge_engine = mock_engine

    env = RLMEnvironment(context="test data", graph_deps=mock_deps)

    # Test magma_view
    view_res = await env.magma_view("test query")
    assert "semantic" in view_res
    mock_engine.retrieve_orthogonal_context.assert_called_once()

    # Test graph_query
    query_res = await env.graph_query("MATCH (n) RETURN n")
    assert query_res == [{"id": 1}]
    mock_engine.query_cypher.assert_called_once()

    # Test sub_agent_call_helper (will trigger a mock run)
    with MagicMock():
        # We'll just check if it doesn't crash for now as it has internal imports
        pass


@pytest.mark.asyncio
async def test_owl_query_helper_delegates():
    """Verify owl_query delegates to the OWL bridge's query_sparql."""

    mock_bridge = MagicMock()
    mock_bridge.query_sparql.return_value = [
        {"manifest": "m1", "edit": "e1"},
        {"manifest": "m2", "edit": "e2"},
    ]

    mock_engine = MagicMock()
    mock_engine.owl_bridge = mock_bridge

    mock_deps = MagicMock()
    mock_deps.knowledge_engine = mock_engine

    env = RLMEnvironment(context="test", graph_deps=mock_deps)
    result = await env.owl_query(
        "PREFIX au: <http://agent-utilities.dev/ontology#> "
        "SELECT ?manifest ?edit WHERE { ?manifest au:hasEditFor ?edit }"
    )

    assert len(result) == 2
    assert result[0] == {"manifest": "m1", "edit": "e1"}
    mock_bridge.query_sparql.assert_called_once()


@pytest.mark.asyncio
async def test_owl_query_error_handling():
    """Verify owl_query handles backend exceptions gracefully."""
    mock_bridge = MagicMock()
    mock_bridge.query_sparql.side_effect = RuntimeError("OWL parse error")

    mock_engine = MagicMock()
    mock_engine.owl_bridge = mock_bridge

    mock_deps = MagicMock()
    mock_deps.knowledge_engine = mock_engine

    env = RLMEnvironment(context="test", graph_deps=mock_deps)
    result = await env.owl_query("INVALID SPARQL")

    assert len(result) == 1
    assert "error" in result[0]
    assert "SPARQL query failed" in result[0]["error"]


@pytest.mark.asyncio
async def test_kg_bulk_export_filters_by_type():
    """Verify kg_bulk_export returns only matching node types."""
    import networkx as nx

    graph = nx.MultiDiGraph()
    graph.add_node("mem_1", type="memory", name="Note 1", importance=0.8)
    graph.add_node("mem_2", type="memory", name="Note 2", importance=0.5)
    graph.add_node("tool_1", type="tool", name="search", importance=0.9)
    graph.add_node("skill_1", type="skill", name="python", importance=0.7)

    mock_engine = MagicMock()
    mock_engine.graph = graph

    mock_deps = MagicMock()
    mock_deps.knowledge_engine = mock_engine

    env = RLMEnvironment(context="test", graph_deps=mock_deps)

    # Export only memory nodes
    result = await env.kg_bulk_export("memory")
    assert len(result) == 2
    assert all(r["type"] == "memory" for r in result)

    # Export only tool nodes
    result = await env.kg_bulk_export("tool")
    assert len(result) == 1
    assert result[0]["name"] == "search"


@pytest.mark.asyncio
async def test_kg_bulk_export_respects_limit():
    """Verify kg_bulk_export respects the limit parameter."""
    import networkx as nx

    graph = nx.MultiDiGraph()
    for i in range(50):
        graph.add_node(f"mem_{i}", type="memory", name=f"Memory {i}")

    mock_engine = MagicMock()
    mock_engine.graph = graph

    mock_deps = MagicMock()
    mock_deps.knowledge_engine = mock_engine

    env = RLMEnvironment(context="test", graph_deps=mock_deps)
    result = await env.kg_bulk_export("memory", limit=5)

    assert len(result) == 5


@pytest.mark.asyncio
async def test_repl_globals_include_new_helpers():
    """Verify owl_query and kg_bulk_export are in the REPL globals."""
    env = RLMEnvironment(context="test")

    assert "owl_query" in env.globals_dict
    assert "kg_bulk_export" in env.globals_dict
    assert callable(env.globals_dict["owl_query"])
    assert callable(env.globals_dict["kg_bulk_export"])
