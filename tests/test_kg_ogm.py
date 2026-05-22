#!/usr/bin/python
from __future__ import annotations

"""Tests for CONCEPT:KG-2.0 — KG Object-Graph Mapper (OGM)."""


from agent_utilities.knowledge_graph.core.ogm import KGMapper, kg_label, resolve_label
from agent_utilities.models.knowledge_graph import (
    AgentNode,
    ProposalNode,
    RegistryNodeType,
    SelfModelNode,  # type: ignore[attr-defined]
    SwarmCoalitionNode,
)


class FakeBackend:
    """Minimal mock backend for OGM tests."""

    def __init__(self):
        self.store: dict[str, dict] = {}
        self.calls: list[tuple[str, dict]] = []

    def execute(self, query: str, params: dict | None = None):
        self.calls.append((query, params or {}))
        # Simulate MATCH ... RETURN n
        if "RETURN n" in query and params and "id" in params:
            node_id = params["id"]
            if node_id in self.store:
                return [{"n": self.store[node_id]}]
        return []


class FakeEngine:
    """Minimal mock engine for OGM tests."""

    def __init__(self, backend=None):
        import networkx as nx

        self.graph = nx.MultiDiGraph()
        self.backend = backend

    def _upsert_node(self, label, node_id, props):
        if self.backend:
            self.backend.store[node_id] = {**props, "id": node_id}

    def link_nodes(self, src, tgt, rel_type, props=None):
        self.graph.add_edge(src, tgt, type=rel_type, **(props or {}))


# ── Label Resolution Tests ────────────────────────────────────────────


class TestResolveLabel:
    def test_snake_case_to_pascal(self):
        assert resolve_label("self_model") == "SelfModel"

    def test_enum_member(self):
        assert resolve_label(RegistryNodeType.SELF_MODEL) == "MemoryRetriever"

    def test_single_word(self):
        assert resolve_label("agent") == "Agent"

    def test_multi_word(self):
        assert (
            resolve_label(RegistryNodeType.KNOWLEDGE_BASE_TOPIC) == "KnowledgeBaseTopic"
        )

    def test_swarm_coalition(self):
        assert resolve_label(RegistryNodeType.SWARM_COALITION) == "SwarmCoalition"

    def test_proposal(self):
        assert resolve_label(RegistryNodeType.PROPOSAL) == "Proposal"


# ── KGMapper CRUD Tests ──────────────────────────────────────────────


class TestKGMapperUpsert:
    def test_upsert_adds_to_networkx(self):
        engine = FakeEngine()
        mapper = KGMapper(engine)

        node = AgentNode(
            id="agent:test",
            name="Test Agent",
            agent_type="mcp",
        )
        result_id = mapper.upsert(node)

        assert result_id == "agent:test"
        assert "agent:test" in engine.graph
        assert engine.graph.nodes["agent:test"]["name"] == "Test Agent"

    def test_upsert_calls_backend(self):
        backend = FakeBackend()
        engine = FakeEngine(backend=backend)
        mapper = KGMapper(engine)

        node = AgentNode(
            id="agent:test",
            name="Test Agent",
            agent_type="mcp",
        )
        mapper.upsert(node)

        assert "agent:test" in backend.store
        assert backend.store["agent:test"]["name"] == "Test Agent"

    def test_upsert_self_model_node(self):
        engine = FakeEngine()
        mapper = KGMapper(engine)

        node = SelfModelNode(
            id="sm:001",
            name="Self-Model v1",
            version=1,
            domain_success_rates={"gitlab": 0.85},
            total_sessions=5,
        )
        mapper.upsert(node)

        assert "sm:001" in engine.graph
        nx_data = engine.graph.nodes["sm:001"]
        assert nx_data["version"] == 1
        assert nx_data["domain_success_rates"] == {"gitlab": 0.85}


class TestKGMapperLoad:
    def test_load_from_networkx(self):
        engine = FakeEngine()
        mapper = KGMapper(engine)

        node = SelfModelNode(
            id="sm:001",
            name="Self-Model v1",
            version=1,
            total_sessions=3,
        )
        mapper.upsert(node)

        loaded = mapper.load("sm:001", SelfModelNode)
        assert loaded is not None
        assert loaded.id == "sm:001"
        assert loaded.version == 1
        assert loaded.total_sessions == 3

    def test_load_returns_none_for_missing(self):
        engine = FakeEngine()
        mapper = KGMapper(engine)

        loaded = mapper.load("nonexistent", SelfModelNode)
        assert loaded is None


class TestKGMapperDelete:
    def test_delete_from_networkx(self):
        engine = FakeEngine()
        mapper = KGMapper(engine)

        node = AgentNode(id="agent:del", name="Delete Me", agent_type="mcp")
        mapper.upsert(node)
        assert "agent:del" in engine.graph

        result = mapper.delete("agent:del")
        assert result is True
        assert "agent:del" not in engine.graph

    def test_delete_nonexistent_returns_false(self):
        engine = FakeEngine()
        mapper = KGMapper(engine)

        result = mapper.delete("nonexistent")
        assert result is False


class TestKGMapperEdge:
    def test_upsert_edge_adds_to_networkx(self):
        engine = FakeEngine()
        mapper = KGMapper(engine)

        engine.graph.add_node("a")
        engine.graph.add_node("b")

        mapper.upsert_edge("a", "b", "VARIANT_OF", {"generation": 1})

        assert engine.graph.has_edge("a", "b")
        edge_data = engine.graph.get_edge_data("a", "b")
        assert edge_data is not None


class TestKGMapperWatch:
    def test_watcher_fires_on_upsert(self):
        engine = FakeEngine()
        mapper = KGMapper(engine)

        events: list[tuple[str, str]] = []

        def on_change(event, node):
            events.append((event, node.id if node else ""))

        mapper.watch("Agent", on_change)

        node = AgentNode(id="agent:watch", name="Watched", agent_type="mcp")
        mapper.upsert(node)

        assert len(events) == 1
        assert events[0] == ("upsert", "agent:watch")


# ── Custom Label Tests ────────────────────────────────────────────────


class TestCustomLabel:
    def test_kg_label_decorator(self):
        @kg_label("CustomLabel")
        class MyNode(AgentNode):
            pass

        engine = FakeEngine()
        mapper = KGMapper(engine)

        node = MyNode(id="custom:1", name="Custom", agent_type="mcp")
        label = mapper._get_label(node)
        assert label == "CustomLabel"


# ── New Node Type Tests ───────────────────────────────────────────────


class TestNewNodeTypes:
    def test_self_model_node_creation(self):
        node = SelfModelNode(
            id="sm:test",
            name="Test Self-Model",
            version=3,
            domain_success_rates={"gitlab": 0.9, "jira": 0.7},
            capability_confidence={"code_review": 0.85},
            tool_proficiency={"search": 0.95},
            total_sessions=10,
            total_tasks_completed=42,
            known_failure_patterns=["timeout on large repos"],
            session_id="sess:abc",
        )
        assert node.type == RegistryNodeType.SELF_MODEL
        assert node.version == 3
        assert node.domain_success_rates["gitlab"] == 0.9

    def test_swarm_coalition_node_creation(self):
        node = SwarmCoalitionNode(
            id="swarm:test",
            name="Test Swarm",
            agents_spawned=5,
            depth_reached=2,
            parallelism_achieved=0.8,
            task_description="Complex multi-domain task",
        )
        assert node.type == RegistryNodeType.SWARM_COALITION
        assert node.agents_spawned == 5

    def test_proposal_node_creation(self):
        node = ProposalNode(
            id="prop:test",
            name="Test Proposal",
            specialist_id="spec:gitlab",
            output="Here are the results...",
            relevance_score=0.9,
            confidence_score=0.8,
            track_record_score=0.85,
            composite_score=0.87,
            selected=True,
        )
        assert node.type == RegistryNodeType.PROPOSAL
        assert node.composite_score == 0.87
