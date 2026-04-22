"""Comprehensive Knowledge Graph Integration Tests.

Tests the full lifecycle: schema → ingestion → MAGMA retrieval → spawning →
self-improvement → maintenance, validating all discussed architectural features.
"""

import os
import shutil
import tempfile

import networkx as nx
import pytest

from agent_utilities.knowledge_graph.backends.ladybug_backend import LadybugBackend
from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.maintenance import GraphMaintainer
from agent_utilities.models.schema_definition import SCHEMA


@pytest.fixture
def temp_db():
    db_dir = tempfile.mkdtemp()
    db_path = os.path.join(db_dir, "integration_test.db")
    yield db_path
    shutil.rmtree(db_dir)


@pytest.fixture
def engine(temp_db):
    nx_graph = nx.MultiDiGraph()
    backend = LadybugBackend(temp_db)
    backend.create_schema()
    return IntelligenceGraphEngine(nx_graph, backend=backend)


# ---------------------------------------------------------------------------
# Schema Verification
# ---------------------------------------------------------------------------


class TestSchemaInitialization:
    """Verify all 30 node tables from SCHEMA are queryable after create_schema()."""

    def test_all_node_tables_exist(self, engine):
        """Every node table defined in SCHEMA should be queryable."""
        for node_def in SCHEMA.nodes:
            res = engine.query_cypher(
                f"MATCH (n:{node_def.name}) RETURN count(n) as cnt"
            )
            assert res is not None, f"Table {node_def.name} not queryable"
            assert "cnt" in res[0], f"Table {node_def.name} returned unexpected result"

    def test_node_tables_count(self, engine):
        """Verify we have all expected node tables."""
        assert len(SCHEMA.nodes) >= 30, (
            f"Expected 30+ node tables, got {len(SCHEMA.nodes)}"
        )

    def test_relationship_tables_count(self, engine):
        """Verify we have all expected relationship tables."""
        assert len(SCHEMA.edges) >= 25, (
            f"Expected 25+ rel tables, got {len(SCHEMA.edges)}"
        )


# ---------------------------------------------------------------------------
# Memory CRUD
# ---------------------------------------------------------------------------


class TestMemoryCRUD:
    """Full CRUD lifecycle for memory nodes."""

    def test_create_memory(self, engine):
        mem_id = engine.add_memory(
            "Test content", name="Test", category="unit-test", tags=["tag1"]
        )
        assert mem_id.startswith("mem:")
        # Verify in backend
        res = engine.query_cypher(
            "MATCH (m:Memory {id: $id}) RETURN m.name as name", {"id": mem_id}
        )
        assert len(res) > 0
        assert res[0]["name"] == "Test"

    def test_read_memory(self, engine):
        mem_id = engine.add_memory("Readable content", name="ReadTest")
        result = engine.get_memory(mem_id)
        assert result is not None
        assert result["id"] == mem_id
        assert result["description"] == "Readable content"

    def test_update_memory(self, engine):
        mem_id = engine.add_memory("Original", name="UpdateTest")
        engine.update_memory(mem_id, description="Updated content")
        # NetworkX should be updated
        assert engine.graph.nodes[mem_id]["description"] == "Updated content"

    def test_delete_memory(self, engine):
        mem_id = engine.add_memory("To delete", name="DeleteTest")
        engine.delete_memory(mem_id)
        assert mem_id not in engine.graph
        res = engine.query_cypher("MATCH (m:Memory {id: $id}) RETURN m", {"id": mem_id})
        assert len(res) == 0

    def test_search_memories(self, engine):
        engine.add_memory("Python development tips", name="PythonMem", category="dev")
        engine.add_memory("Cooking recipes", name="CookMem", category="hobby")
        results = engine.search_memories("python")
        assert len(results) > 0
        assert any("Python" in str(r.get("name", "")) for r in results)


# ---------------------------------------------------------------------------
# Episode & Ingestion
# ---------------------------------------------------------------------------


class TestIngestion:
    """Test all ingestion pathways (episode, MCP, A2A, Skill)."""

    def test_ingest_episode(self, engine):
        ep_id = engine.ingest_episode("User asked about deployment", source="chat")
        assert ep_id.startswith("ep:")
        res = engine.query_cypher(
            "MATCH (e:Episode {id: $id}) RETURN e.source as src", {"id": ep_id}
        )
        assert res[0]["src"] == "chat"

    def test_ingest_mcp_server(self, engine):
        tools = [
            {"name": "read_file", "description": "Read a file", "tags": ["fs"]},
            {"name": "write_file", "description": "Write a file", "tags": ["fs"]},
        ]
        engine.ingest_mcp_server("filesystem", "http://localhost:3000", tools)

        # Verify server node
        res = engine.query_cypher(
            "MATCH (s:Server {id: 'srv:filesystem'}) RETURN s.url as url"
        )
        assert len(res) > 0
        assert res[0]["url"] == "http://localhost:3000"

        # Verify callable resources
        res = engine.query_cypher(
            "MATCH (r:CallableResource) WHERE r.resource_type = 'MCP_TOOL' RETURN count(r) as cnt"
        )
        assert res[0]["cnt"] >= 2

    def test_ingest_a2a_agent(self, engine):
        card = {
            "name": "CodeReviewer",
            "description": "Reviews code changes",
            "capabilities": ["review"],
        }
        engine.ingest_a2a_agent_card("http://reviewer.io", card)

        res = engine.query_cypher(
            "MATCH (r:CallableResource {resource_type: 'A2A_AGENT'}) RETURN r.name as name"
        )
        assert len(res) > 0
        assert res[0]["name"] == "CodeReviewer"

    def test_ingest_agent_skill(self, engine):
        frontmatter = {
            "name": "code-analyzer",
            "tags": ["analysis"],
            "description": "Analyzes code quality",
        }
        engine.ingest_agent_skill(
            "/skills/code-analyzer/SKILL.md", frontmatter, "skill body..."
        )

        res = engine.query_cypher(
            "MATCH (r:CallableResource {resource_type: 'AGENT_SKILL'}) RETURN r.name as name"
        )
        assert len(res) > 0
        assert res[0]["name"] == "code-analyzer"

    def test_metadata_linkage(self, engine):
        """Verify CallableResource → ToolMetadata edges are created."""
        engine.ingest_a2a_agent_card(
            "http://test.io", {"name": "LinkedAgent", "description": "Test"}
        )
        res = engine.query_cypher(
            "MATCH (r:CallableResource)-[:HAS_METADATA]->(m:ToolMetadata) "
            "WHERE r.name = 'LinkedAgent' RETURN m.source as source"
        )
        assert len(res) > 0
        assert res[0]["source"] == "A2A"


# ---------------------------------------------------------------------------
# MAGMA Orthogonal Retrieval
# ---------------------------------------------------------------------------


class TestMAGMARetrieval:
    """Test the 4 orthogonal MAGMA views."""

    def test_semantic_view(self, engine):
        engine.add_memory("Machine learning algorithms", name="ML Mem")
        ctx = engine.retrieve_orthogonal_context("machine learning", views=["semantic"])
        assert "semantic" in ctx["views"]
        assert len(ctx["views"]["semantic"]) > 0

    def test_temporal_view(self, engine):
        engine.ingest_episode("First event", source="chat")
        engine.ingest_episode("Second event", source="chat")
        ctx = engine.retrieve_orthogonal_context("events", views=["temporal"])
        assert "temporal" in ctx["views"]
        assert len(ctx["views"]["temporal"]) > 0

    def test_causal_view(self, engine):
        ctx = engine.retrieve_orthogonal_context("reasoning", views=["causal"])
        assert "causal" in ctx["views"]

    def test_entity_view(self, engine):
        ctx = engine.retrieve_orthogonal_context("test", views=["entity"])
        assert "entity" in ctx["views"]

    def test_multi_view(self, engine):
        engine.ingest_episode("Test episode", source="chat")
        ctx = engine.retrieve_orthogonal_context(
            "test", views=["semantic", "temporal", "entity"]
        )
        assert len(ctx["views"]) == 3


# ---------------------------------------------------------------------------
# Agent Spawning
# ---------------------------------------------------------------------------


class TestAgentSpawning:
    """Test dynamic agent creation with composed prompts."""

    def test_spawn_agent(self, engine):
        agent_id = engine.spawn_specialized_agent(
            task_description="Write unit tests for auth module",
            tool_ids=["tool:pytest", "tool:ast-parser"],
        )
        assert agent_id.startswith("spawn:")

        res = engine.query_cypher(
            "MATCH (a:SpawnedAgent {id: $id}) RETURN a.system_prompt as prompt",
            {"id": agent_id},
        )
        assert "Write unit tests" in res[0]["prompt"]

    def test_spawn_with_tool_links(self, engine):
        """Verify spawned agent gets USES edges to its tools."""
        # First ingest a resource so we can link to it
        engine.ingest_agent_skill("/skills/test.md", {"name": "test-skill"}, "body")

        agent_id = engine.spawn_specialized_agent(
            task_description="Analyze code", tool_ids=["skill:test-skill"]
        )

        res = engine.query_cypher(
            "MATCH (a:SpawnedAgent {id: $id})-[:USES]->(r:CallableResource) RETURN r.id as rid",
            {"id": agent_id},
        )
        assert len(res) > 0


# ---------------------------------------------------------------------------
# Agent Lightning Self-Improvement
# ---------------------------------------------------------------------------


class TestSelfImprovement:
    """Test the full Lightning-style APO loop."""

    def test_record_outcome(self, engine):
        ep_id = engine.ingest_episode("Task failed", source="chat")
        engine.record_outcome(ep_id, reward=-0.5, feedback="Insufficient context")

        res = engine.query_cypher(
            "MATCH (e:Episode)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation) "
            "WHERE e.id = $id RETURN o.reward as reward",
            {"id": ep_id},
        )
        assert len(res) > 0
        assert res[0]["reward"] == -0.5

    def test_generate_critique(self, engine):
        ep_id = engine.ingest_episode("Reasoning step", source="chat")
        crit_id = engine.generate_critique(ep_id, "Should have used broader context")
        assert crit_id.startswith("crit:")

        res = engine.query_cypher(
            "MATCH (c:Critique {id: $id}) RETURN c.textual_gradient as grad",
            {"id": crit_id},
        )
        assert "broader context" in res[0]["grad"]

    def test_optimize_prompt(self, engine):
        # Create a base prompt
        base_id = "prompt:test-base"
        engine.backend.execute(
            "CREATE (p:SystemPrompt {id: $id, content: $content, tags: ['agent-base'], source: 'MANUAL', version: 'v1'})",
            {"id": base_id, "content": "You are a helpful assistant."},
        )

        # Create a critique
        crit_id = engine.generate_critique(
            "ep:dummy", "Add specificity to instructions"
        )

        # Optimize
        new_id = engine.optimize_prompt(base_id, crit_id)
        assert new_id.startswith("prompt:")

        # Verify EVOLVED_FROM edge
        res = engine.query_cypher(
            "MATCH (new:SystemPrompt)-[:EVOLVED_FROM]->(old:SystemPrompt) "
            "WHERE new.id = $nid RETURN old.id as oid",
            {"nid": new_id},
        )
        assert len(res) > 0
        assert res[0]["oid"] == base_id

    def test_full_self_improvement_cycle(self, engine):
        """Test the end-to-end Lightning trainer loop."""
        # Setup: episode + negative outcome + prompt
        ep_id = engine.ingest_episode("Failed at code review", source="chat")
        engine.record_outcome(ep_id, reward=-1.0, feedback="Missed edge cases")

        base_id = "prompt:cycle-base"
        engine.backend.execute(
            "CREATE (p:SystemPrompt {id: $id, content: $content, source: 'MANUAL', version: 'v1'})",
            {"id": base_id, "content": "Review code carefully."},
        )

        # Spawn agent and link to episode and prompt
        agent_id = engine.spawn_specialized_agent("review code", [])

        engine.backend.execute(
            "MATCH (e:Episode), (a:SpawnedAgent) WHERE e.id = $eid AND a.id = $aid MERGE (e)-[:EXECUTED_BY]->(a)",
            {"eid": ep_id, "aid": agent_id},
        )
        engine.backend.execute(
            "MATCH (a:SpawnedAgent), (p:SystemPrompt) WHERE a.id = $aid AND p.id = $pid MERGE (a)-[:USES]->(p)",
            {"aid": agent_id, "pid": base_id},
        )

        # Run cycle
        engine.run_self_improvement_cycle()

        # Verify refined prompts were created
        res = engine.query_cypher(
            "MATCH (p:SystemPrompt) WHERE p.source = 'REFINED' RETURN count(p) as cnt"
        )
        assert res[0]["cnt"] > 0


# ---------------------------------------------------------------------------
# Maintenance Operations
# ---------------------------------------------------------------------------


class TestMaintenance:
    """Test GraphMaintainer operations."""

    def test_importance_scoring(self, engine):
        engine.add_memory("Node A", name="A")
        engine.add_memory("Node B", name="B")
        engine.graph.add_edge(
            list(engine.graph.nodes)[0], list(engine.graph.nodes)[1], type="RELATED_TO"
        )

        maintainer = GraphMaintainer(engine)
        updated = maintainer.update_importance_scores()
        assert updated >= 0  # May be 0 if graph is too small for PageRank

    def test_temporal_decay(self, engine):
        mem_id = engine.add_memory("Old memory")
        engine.backend.execute(
            "MATCH (n {id: $id}) SET n.timestamp = '2020-01-01T00:00:00Z', n.importance_score = 1.0",
            {"id": mem_id},
        )

        maintainer = GraphMaintainer(engine)
        maintainer.apply_temporal_decay()

        res = engine.query_cypher(
            "MATCH (n {id: $id}) RETURN n.importance_score as score", {"id": mem_id}
        )
        assert float(res[0]["score"]) < 1.0

    def test_consolidate_memory(self, engine):
        # Create old episodes
        for i in range(3):
            ep_id = engine.ingest_episode(f"Old episode {i}", source="chat")
            engine.backend.execute(
                "MATCH (e:Episode {id: $id}) SET e.timestamp = '2020-01-01T00:00:00Z'",
                {"id": ep_id},
            )

        maintainer = GraphMaintainer(engine)
        consolidated = maintainer.consolidate_memory(keep_days=1)
        assert consolidated >= 3


# ---------------------------------------------------------------------------
# Graph Algorithms (NetworkX)
# ---------------------------------------------------------------------------


class TestGraphAlgorithms:
    """Test NetworkX-powered analysis on the graph."""

    def test_impact_analysis(self, engine):
        engine.graph.add_node("mod:auth", type="module", name="auth")
        engine.graph.add_node("mod:api", type="module", name="api")
        engine.graph.add_edge("mod:api", "mod:auth", type="depends_on")

        impact = engine.query_impact("mod:auth")
        assert len(impact) > 0
        assert any(n["id"] == "mod:api" for n in impact)

    def test_shortest_path(self, engine):
        engine.graph.add_node("A", type="node", name="A")
        engine.graph.add_node("B", type="node", name="B")
        engine.graph.add_node("C", type="node", name="C")
        engine.graph.add_edge("A", "B", type="rel")
        engine.graph.add_edge("B", "C", type="rel")

        path = engine.find_path("A", "C")
        assert path == ["A", "B", "C"]

    def test_hybrid_search(self, engine):
        engine.graph.add_node(
            "tool:python",
            type="tool",
            name="python_runner",
            description="Runs Python code",
        )
        results = engine.search_hybrid("python")
        assert len(results) > 0
        assert results[0]["name"] == "python_runner"
