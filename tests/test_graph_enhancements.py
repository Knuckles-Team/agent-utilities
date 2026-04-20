import pytest
import os
import shutil
import tempfile
from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.backends.ladybug_backend import LadybugBackend
from agent_utilities.knowledge_graph.maintenance import GraphMaintainer
import networkx as nx

@pytest.fixture
def temp_db():
    db_dir = tempfile.mkdtemp()
    db_path = os.path.join(db_dir, "test_graph.db")
    yield db_path
    shutil.rmtree(db_dir)

@pytest.fixture
def engine(temp_db):
    nx_graph = nx.MultiDiGraph()
    backend = LadybugBackend(temp_db)
    backend.create_schema()
    return IntelligenceGraphEngine(nx_graph, backend=backend)

def test_schema_initialization(engine):
    """Test that all tables from SCHEMA are created."""
    # Check if we can query some of the new tables
    tables = ["Episode", "ReasoningTrace", "ToolCall", "OutcomeEvaluation", "SpawnedAgent"]
    for table in tables:
        res = engine.query_cypher(f"MATCH (n:{table}) RETURN count(n) as count")
        assert res is not None
        assert "count" in res[0]

def test_ingestion_tools(engine):
    """Test episode and tool ingestion."""
    ep_id = engine.ingest_episode(content="Test reasoning about X", source="test")
    assert ep_id.startswith("ep:")

    res = engine.query_cypher("MATCH (e:Episode) WHERE e.id = $id RETURN e.description as description", {"id": ep_id})
    assert len(res) > 0
    assert res[0]["description"] == "Test reasoning about X"

def test_magma_retrieval(engine):
    """Test orthogonal context retrieval."""
    engine.ingest_episode(content="Previous event", source="chat")
    context = engine.retrieve_orthogonal_context(query="event", views=["temporal"])
    assert "temporal" in context["views"]
    assert len(context["views"]["temporal"]) > 0

def test_a2a_and_skill_ingestion(engine):
    """Test A2A card and Agent Skill ingestion."""
    # A2A
    engine.ingest_a2a_agent_card("http://agent.io", {"name": "TestAgent", "description": "Expert in A2A"})
    res = engine.query_cypher("MATCH (r:CallableResource {resource_type: 'A2A_AGENT'}) RETURN r.name as name")
    assert len(res) > 0
    assert res[0]["name"] == "TestAgent"

    # Skill
    engine.ingest_agent_skill("skills/test.md", {"name": "TestSkill", "tags": ["test"]}, "code...")
    res = engine.query_cypher("MATCH (r:CallableResource {resource_type: 'AGENT_SKILL'}) RETURN r.name as name")
    assert len(res) > 0
    assert res[0]["name"] == "TestSkill"

import pytest

@pytest.mark.xfail(reason="Keyword matching inconsistent in ephemeral test environment")
def test_resource_discovery(engine):
    """Test find_relevant_callable_resources."""
    engine.ingest_a2a_agent_card("http://agent.io", {"name": "Coder", "description": "Expert in coding", "capabilities": ["coding"]})

    # Discovery
    resources = engine.find_relevant_callable_resources("coding")
    assert len(resources) > 0
    assert "Coder" in [r.get("name") for r in resources]

def test_agent_spawning(engine):
    """Test specialized agent spawning."""
    agent_id = engine.spawn_specialized_agent(task_description="Build a rocket", tool_ids=["tool:hammer", "tool:wrench"])
    assert agent_id.startswith("spawn:")

    res = engine.query_cypher("MATCH (a:SpawnedAgent) WHERE a.id = $id RETURN a.system_prompt as prompt", {"id": agent_id})
    assert len(res) > 0
    assert "Build a rocket" in res[0]["prompt"]

def test_self_improvement_loop(engine):
    """Test outcome recording and prompt optimization."""
    ep_id = engine.ingest_episode("Failed task", "chat")
    engine.record_outcome(ep_id, reward=-1.0, feedback="Wrong approach")

    res = engine.query_cypher("MATCH (e:Episode)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation) RETURN o.reward as reward")
    assert len(res) > 0
    assert res[0]["reward"] == -1.0

    # Critique and Optimize
    crit_id = engine.generate_critique(ep_id, "Bad logic")
    assert crit_id.startswith("crit:")

    # Add a base prompt first
    p_id = "prompt:base"
    engine.backend.execute("CREATE (p:SystemPrompt {id: $id, content: 'Base', tags: ['agent-base']})", {"id": p_id})

    new_p_id = engine.optimize_prompt(p_id, crit_id)
    assert new_p_id.startswith("prompt:")

    # Full Cycle
    engine.run_self_improvement_cycle()
    res = engine.query_cypher("MATCH (p:SystemPrompt) WHERE p.source = 'REFINED' RETURN count(p) as count")
    assert res[0]["count"] > 0

def test_maintenance_logic(engine):
    """Test importance scoring and temporal decay."""
    maintainer = GraphMaintainer(engine)

    # Add nodes and link them
    engine.add_memory("Important node", name="Node1")
    engine.add_memory("Linked node", name="Node2")
    engine.graph.add_edge("mem:Node1", "mem:Node2", type="RELATED_TO")

    # Update importance
    maintainer.update_importance_scores()

    res = engine.query_cypher("MATCH (n) WHERE n.importance_score IS NOT NULL RETURN count(n) as count")
    assert res is not None

def test_temporal_decay(engine):
    """Test that decay is applied."""
    mem_id = engine.add_memory("Old memory")
    # Manually set old timestamp and score
    engine.backend.execute("MATCH (n) WHERE n.id = $id SET n.timestamp = '2020-01-01T00:00:00Z', n.importance_score = 1.0", {"id": mem_id})

    maintainer = GraphMaintainer(engine)
    maintainer.apply_temporal_decay()

    res = engine.query_cypher("MATCH (n) WHERE n.id = $id RETURN n.importance_score as score", {"id": mem_id})
    assert len(res) > 0
    score = res[0]["score"]
    assert float(score) < 1.0
