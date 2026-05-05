"""Tests for CONCEPT:AHE-3.3 — TeamConfig & Proven Team Reuse.

Validates:
    - ``TeamConfigNode`` model creation and field defaults
    - ``find_matching_team_config()`` query and ranking
    - ``promote_coalition_to_template()`` lifecycle + cache invalidation
    - ``record_team_outcome()`` EMA updates
    - ``link_prompt_to_agent()`` edge creation
"""

import networkx as nx
import pytest

from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine
from agent_utilities.models.knowledge_graph import (
    AgentCapabilityNode,
    RegistryEdgeType,
    RegistryNodeType,
    SwarmCoalitionNode,
    TeamConfigNode,
)


@pytest.fixture()
def engine():
    """Create a minimal in-memory IntelligenceGraphEngine for testing."""
    g = nx.MultiDiGraph()
    e = IntelligenceGraphEngine(graph=g, db_path=":memory:")
    IntelligenceGraphEngine.set_active(e)
    return e


@pytest.mark.concept("AU-025")
class TestTeamConfigNode:
    """Test suite for the TeamConfigNode model."""

    def test_create_team_config(self):
        """TeamConfigNode should create with correct defaults."""
        tc = TeamConfigNode(
            id="tc:test",
            name="Test Team",
            task_pattern="code audit",
        )
        assert tc.type == RegistryNodeType.TEAM_CONFIG
        assert tc.task_pattern == "code audit"
        assert tc.specialist_ids == []
        assert tc.capability_overrides == {}
        assert tc.success_rate == 0.0
        assert tc.usage_count == 0
        assert tc.reuse_threshold == 0.72

    def test_team_config_with_capability_overrides(self):
        """capability_overrides should correctly store RLM synergy mappings."""
        tc = TeamConfigNode(
            id="tc:rlm",
            name="RLM Team",
            task_pattern="audit the codebase",
            specialist_ids=["code_researcher", "architect"],
            capability_overrides={
                "code_researcher": ["rlm", "navigator"],
                "architect": ["synthesizer"],
            },
        )
        assert "rlm" in tc.capability_overrides["code_researcher"]
        assert "navigator" in tc.capability_overrides["code_researcher"]
        assert "synthesizer" in tc.capability_overrides["architect"]

    def test_team_config_serialization(self):
        """Should round-trip through model_dump/model_validate."""
        tc = TeamConfigNode(
            id="tc:serial",
            name="Serial Test",
            task_pattern="build API client",
            specialist_ids=["api_builder"],
            success_rate=0.85,
        )
        data = tc.model_dump()
        restored = TeamConfigNode.model_validate(data)
        assert restored.id == tc.id
        assert restored.success_rate == 0.85


@pytest.mark.concept("AU-025")
class TestTeamConfigLookup:
    """Test suite for find_matching_team_config()."""

    def test_find_with_keyword_match(self, engine):
        """Should find TeamConfigs that share keywords with the query."""
        # Add a TeamConfig to the graph
        tc = TeamConfigNode(
            id="tc:audit",
            name="Audit Team",
            task_pattern="audit the codebase for security issues",
            specialist_ids=["security_analyst"],
            success_rate=0.9,
        )
        engine.graph.add_node(tc.id, **tc.model_dump())

        results = engine.find_matching_team_config("audit the repository")
        assert len(results) >= 1
        assert results[0].task_pattern == tc.task_pattern

    def test_find_returns_empty_for_no_match(self, engine):
        """Should return empty list when no TeamConfigs match."""
        results = engine.find_matching_team_config("build a spaceship")
        assert results == []

    def test_find_respects_top_k(self, engine):
        """Should limit results to top_k."""
        for i in range(5):
            tc = TeamConfigNode(
                id=f"tc:match_{i}",
                name=f"Match {i}",
                task_pattern=f"deploy service number {i}",
                success_rate=0.5 + i * 0.1,
            )
            engine.graph.add_node(tc.id, **tc.model_dump())

        results = engine.find_matching_team_config("deploy service", top_k=2)
        assert len(results) <= 2


@pytest.mark.concept("AU-025")
class TestPromoteCoalition:
    """Test suite for promote_coalition_to_template()."""

    def test_promote_creates_team_config(self, engine):
        """Promoting a coalition should create a TeamConfig node."""
        # Create a mock coalition
        coalition = SwarmCoalitionNode(
            id="coalition:test",
            name="Test Coalition",
            agents_spawned=3,
            task_description="Analyze repository",
        )
        engine.graph.add_node(coalition.id, **coalition.model_dump())

        result = engine.promote_coalition_to_template(
            coalition_id=coalition.id,
            task_pattern="repository analysis",
        )

        assert "id" in result
        assert result["task_pattern"] == "repository analysis"
        assert result["success_rate"] == 1.0  # Initial promotion

    def test_promote_creates_reused_team_edge(self, engine):
        """Should create a REUSED_TEAM edge from TeamConfig to coalition."""
        coalition = SwarmCoalitionNode(
            id="coalition:edge",
            name="Edge Test Coalition",
            agents_spawned=2,
        )
        engine.graph.add_node(coalition.id, **coalition.model_dump())

        result = engine.promote_coalition_to_template(
            coalition_id=coalition.id,
            task_pattern="edge test",
        )
        tc_id = result["id"]

        # Check edge exists in NetworkX
        assert engine.graph.has_edge(tc_id, coalition.id)


@pytest.mark.concept("AU-025")
class TestRecordTeamOutcome:
    """Test suite for record_team_outcome()."""

    def test_updates_success_rate(self, engine):
        """Should update success_rate using EMA."""
        tc = TeamConfigNode(
            id="tc:outcome",
            name="Outcome Team",
            task_pattern="test outcome",
            success_rate=0.5,
        )
        engine.graph.add_node(tc.id, **tc.model_dump())

        engine.record_team_outcome("tc:outcome", reward=1.0)

        data = engine.graph.nodes["tc:outcome"]
        assert data["success_rate"] > 0.5  # EMA should increase
        assert data["usage_count"] == 1

    def test_increments_usage_count(self, engine):
        """Usage count should increment on each outcome recording."""
        tc = TeamConfigNode(
            id="tc:count",
            name="Count Team",
            task_pattern="test count",
            usage_count=5,
        )
        engine.graph.add_node(tc.id, **tc.model_dump())

        engine.record_team_outcome("tc:count", reward=0.8)

        data = engine.graph.nodes["tc:count"]
        assert data["usage_count"] == 6


@pytest.mark.concept("AU-025")
class TestLinkPromptToAgent:
    """Test suite for link_prompt_to_agent()."""

    def test_creates_uses_prompt_edge(self, engine):
        """Should create a USES_PROMPT edge."""
        engine.graph.add_node("agent:test", type="agent", name="Test Agent")
        engine.graph.add_node("prompt:test", type="prompt", name="Test Prompt")

        engine.link_prompt_to_agent("agent:test", "prompt:test")

        assert engine.graph.has_edge("agent:test", "prompt:test")
        edge_data = engine.graph.get_edge_data("agent:test", "prompt:test")
        assert any(
            e.get("type") == RegistryEdgeType.USES_PROMPT for e in edge_data.values()
        )


@pytest.mark.concept("AU-026")
class TestAgentCapabilityNode:
    """Test suite for the AgentCapabilityNode model."""

    def test_create_capability_node(self):
        """AgentCapabilityNode should create with correct defaults."""
        cap = AgentCapabilityNode(
            id="cap:rlm",
            name="RLM Capability",
            capability_type="rlm",
            handler_module="agent_utilities.rlm.specialist",
            handler_function="run",
            trigger_conditions={"input_chars_gt": 50000},
        )
        assert cap.type == RegistryNodeType.AGENT_CAPABILITY
        assert cap.auto_activate is True
        assert cap.performance_score == 0.5

    def test_capability_with_custom_trigger(self):
        """Should support custom trigger conditions."""
        cap = AgentCapabilityNode(
            id="cap:critic",
            name="Critic",
            capability_type="critic",
            handler_module="agent_utilities.harness.verifier",
            trigger_conditions={"always": True},
            auto_activate=False,
        )
        assert cap.trigger_conditions == {"always": True}
        assert cap.auto_activate is False

    def test_schema_enum_values(self):
        """RegistryNodeType and RegistryEdgeType should have new values."""
        assert RegistryNodeType.TEAM_CONFIG == "team_config"
        assert RegistryNodeType.AGENT_CAPABILITY == "agent_capability"
        assert RegistryEdgeType.HAS_CAPABILITY == "has_capability"
        assert RegistryEdgeType.REUSED_TEAM == "reused_team"
