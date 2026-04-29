import pytest
import networkx as nx
from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine, cosine_similarity
from agent_utilities.models.knowledge_graph import RegistryNodeType, RegistryEdgeType

def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 1], [1, 1]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [1, 1]) > 0.7

@pytest.fixture
def engine(monkeypatch):
    # Isolate from any active backend singleton set by earlier tests
    # so IntelligenceGraphEngine.__init__ does not pick up a polluted backend.
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.engine.get_active_backend",
        lambda: None,
    )
    g = nx.MultiDiGraph()
    return IntelligenceGraphEngine(graph=g)

def test_add_memory(engine):
    mem_id = engine.add_memory("test content", name="test memory", tags=["tag1"])
    assert mem_id.startswith("mem:")
    assert mem_id in engine.graph
    assert engine.graph.nodes[mem_id]["description"] == "test content"

def test_search_hybrid(engine):
    engine.graph.add_node("node1", name="Python Expert", description="Helps with python")
    engine.graph.add_node("node2", name="Rust Expert", description="Helps with rust")

    results = engine.search_hybrid("python")
    assert len(results) == 1
    assert results[0]["id"] == "node1"

def test_query_impact(engine):
    # A depends on B, B depends on C
    engine.graph.add_edge("A", "B", type="DEPENDS_ON")
    engine.graph.add_edge("B", "C", type="DEPENDS_ON")
    engine.graph.nodes["A"]["name"] = "A"
    engine.graph.nodes["B"]["name"] = "B"
    engine.graph.nodes["C"]["name"] = "C"

    impact = engine.query_impact("C")
    # A and B are ancestors of C if edges go A->B->C
    # nx.ancestors(G, "C") returns {"A", "B"}
    assert len(impact) == 2
    ids = [n["id"] for n in impact]
    assert "A" in ids
    assert "B" in ids

def test_find_path(engine):
    engine.graph.add_edge("A", "B")
    engine.graph.add_edge("B", "C")
    path = engine.find_path("A", "C")
    assert path == ["A", "B", "C"]

def test_get_agent_tools(engine):
    agent_id = "agent:test"
    engine.graph.add_node(agent_id, type="agent")
    engine.graph.add_node("tool:t1", type="tool")
    engine.graph.add_edge(agent_id, "tool:t1", type=RegistryEdgeType.PROVIDES)

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
    assert eval_id.startswith("eval:")
    assert eval_id in engine.graph
    assert engine.graph.nodes[eval_id]["reward"] == 0.9

    # Check edge
    assert engine.graph.has_edge(ep_id, eval_id)
    edge_data = engine.graph.get_edge_data(ep_id, eval_id, 0)
    assert edge_data["type"] == "PRODUCED_OUTCOME"

def test_nx_fallback_successful_episodes(engine):
    # Setup successful episode
    ep_id = "ep:1"
    eval_id = "eval:1"
    tool_id = "tool:t1"

    engine.graph.add_node(ep_id, type="episode", description="success task")
    engine.graph.add_node(eval_id, type=RegistryNodeType.OUTCOME_EVALUATION, reward=0.9)
    engine.graph.add_node(tool_id, type="tool_call", tool_name="my_tool")

    engine.graph.add_edge(ep_id, eval_id, type="PRODUCED_OUTCOME")
    engine.graph.add_edge(ep_id, tool_id, type="USED_TOOL")

    query = "MATCH (e:Episode)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation), (e)-[:USED_TOOL]->(t:tool_call) WHERE o.reward >= 0.8 RETURN t.tool_name as tool"
    results = engine.query_cypher(query)

    assert len(results) == 1
    assert results[0]["tool"] == "my_tool"
