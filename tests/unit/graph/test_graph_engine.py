"""CONCEPT:AU-ORCH.execution.inject-signal-board-observations"""

import pytest

from agent_utilities.knowledge_graph.core.engine import (
    IntelligenceGraphEngine,
    cosine_similarity,
)
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.models.knowledge_graph import RegistryEdgeType, RegistryNodeType
from agent_utilities.observability.trace_ontology import trace_id


def test_cosine_similarity():
    import math

    assert math.isclose(cosine_similarity([1, 0], [1, 0]), 1.0)
    assert math.isclose(cosine_similarity([1, 0], [0, 1]), 0.0)
    assert math.isclose(cosine_similarity([1, 1], [1, 1]), 1.0)
    assert cosine_similarity([1, 0], [1, 1]) > 0.7


@pytest.fixture
def engine(monkeypatch, request):
    # Isolate from any active backend singleton set by earlier tests
    # so IntelligenceGraphEngine.__init__ does not pick up a polluted backend.
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.engine.get_active_backend",
        lambda: None,
    )
    g = GraphComputeEngine(backend_type="rust")
    for node in g.node_ids():
        g.remove_node(node)
    isolated = IntelligenceGraphEngine(db_path=":memory:")
    isolated.backend = None
    return isolated


def test_add_memory(engine):
    mem_id = engine.add_memory("test content", name="test memory", tags=["tag1"])
    assert mem_id.startswith("mem:")
    assert mem_id in engine.graph
    assert engine.graph.nodes[mem_id]["description"] == "test content"


def test_search_hybrid(engine):
    engine.graph.add_node(
        "node1", name="Python Expert", description="Helps with python"
    )
    engine.graph.add_node("node2", name="Rust Expert", description="Helps with rust")

    results = engine.search_hybrid("python")
    assert len(results) == 1
    assert results[0]["id"] == "node1"


def test_query_impact(engine):
    # A depends on B, B depends on C
    engine.graph.add_edge("A", "B", relationship="DEPENDS_ON")
    engine.graph.add_edge("B", "C", relationship="DEPENDS_ON")
    engine.graph.nodes["A"]["name"] = "A"
    engine.graph.nodes["B"]["name"] = "B"
    engine.graph.nodes["C"]["name"] = "C"

    impact = engine.query_impact("C")
    # A and B are ancestors of C if edges go A->B->C
    # engine.get_predecessors(G, "C") returns {"A", "B"}
    assert len(impact) == 2
    ids = [n["id"] for n in impact]
    assert "A" in ids
    assert "B" in ids


def test_find_path(engine):
    engine.graph.add_edge("A", "B", relationship="RELATED_TO")
    engine.graph.add_edge("B", "C", relationship="RELATED_TO")
    path = engine.find_path("A", "C")
    assert path == ["A", "B", "C"]


def test_get_agent_tools(engine):
    agent_id = "agent:test"
    engine.graph.add_node(agent_id, node_type="agent")
    engine.graph.add_node("tool:t1", node_type="tool")
    engine.graph.add_edge(agent_id, "tool:t1", relationship=RegistryEdgeType.PROVIDES)

    tools = engine.get_agent_tools(agent_id)
    assert tools == ["t1"]


def test_ingest_episode(engine):
    ep_id = engine.ingest_episode("did something")
    assert ep_id.startswith("ep:")
    assert ep_id in engine.graph
    assert engine.graph.nodes[ep_id]["description"] == "did something"


def test_record_outcome(engine):
    ep_id = engine.ingest_episode("task")
    eval_id = engine.record_outcome(ep_id, reward=0.9, feedback="good")
    assert eval_id.startswith("outcome:")
    assert eval_id in engine.graph
    assert engine.graph.nodes[eval_id]["reward"] == 0.9

    # Check edge
    canonical_trace_id = trace_id(ep_id)
    assert engine.graph.has_edge(canonical_trace_id, eval_id)
    edge_data = engine.graph.get_edge_data(canonical_trace_id, eval_id, 0)
    assert edge_data["relationship"] == "PRODUCED_OUTCOME"


def test_query_fails_closed_without_authoritative_backend(engine):
    # Graph reads must not silently fall back to a process-local projection.
    engine.backend = None

    # Setup a successful canonical trace in the in-memory graph
    trace_id = "trace:1"
    eval_id = "eval:1"
    tool_id = "tool:t1"

    engine.graph.add_node(trace_id, node_type="RunTrace", task="success task")
    engine.graph.add_node(
        eval_id, node_type=RegistryNodeType.OUTCOME_EVALUATION, reward=0.9
    )
    engine.graph.add_node(tool_id, node_type="tool_call", tool_name="my_tool")

    engine.graph.add_edge(trace_id, eval_id, relationship="PRODUCED_OUTCOME")
    engine.graph.add_edge(trace_id, tool_id, relationship="USED_TOOL")

    query = "MATCH (r:RunTrace)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation), (r)-[:USED_TOOL]->(t:tool_call) WHERE o.reward >= 0.8 RETURN t.tool_name as tool"
    with pytest.raises(
        RuntimeError, match="authoritative graph read service is unavailable"
    ):
        engine.query_cypher(query)
