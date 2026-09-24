"""Tests for CONCEPT:AU-AHE.evaluation.interpretability-tests — TeamConfig & Proven Team Reuse.

Validates:
    - ``TeamConfigNode`` model creation and field defaults
    - ``promote_coalition_to_template()`` lifecycle + cache invalidation
    - no success-rate store or selection on a TeamConfig (ST-7, invariant T5)
    - ``link_prompt_to_agent()`` edge creation
"""

import pytest

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
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
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )

    # A bare EpistemicGraphBackend()/IntelligenceGraphEngine(db_path=...)
    # independently resolves its OWN tenant-routed default graph
    # (resolve_routing_graph(None) -> the shared "tenant__<tenant>____commons__"
    # graph), NOT the per-test isolated graph tests/conftest.py's
    # isolate_graph_compute_engine fixture provisions. Constructing the
    # isolated GraphComputeEngine first and rebinding the backend to it keeps
    # this test on its own graph (same idiom as D-OTR-3/test_kg_native_orchestration.py).
    compute = GraphComputeEngine(backend_type="rust")
    backend = EpistemicGraphBackend()
    backend._graph = compute
    e = IntelligenceGraphEngine(backend=backend)
    IntelligenceGraphEngine._set_active_for_tests(e)
    return e


def _node_kwargs(model) -> dict:
    """``model_dump()`` a RegistryNode subclass into ``add_node``-safe kwargs.

    Every ``RegistryNode`` still carries a Pydantic ``type`` field, but
    ``GraphComputeEngine.add_node`` hard-rejects a stray ``type`` property
    (retired in favor of the canonical ``node_type``) — mirrors
    ``IntelligenceGraphEngine._serialize_node``'s own rename.
    """
    props = model.model_dump()
    props["node_type"] = props.pop("type")
    return props


@pytest.mark.concept("CONCEPT:AU-AHE.evaluation.interpretability-tests")
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
        for counter in ("success_rate", "usage_count", "reuse_threshold"):
            assert counter not in TeamConfigNode.model_fields

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
        )
        data = tc.model_dump()
        restored = TeamConfigNode.model_validate(data)
        assert restored.id == tc.id
        assert restored.specialist_ids == ["api_builder"]


@pytest.mark.concept("CONCEPT:AU-AHE.evaluation.interpretability-tests")
class TestNoSuccessRateSelection:
    """ST-7: the success-rate lookup, EMA and listing are gone."""

    def test_the_success_rate_api_is_gone(self, engine):
        for name in (
            "find_matching_team_config",
            "record_team_outcome",
            "list_team_configs",
        ):
            assert not hasattr(engine, name), name

    def test_an_imported_bundle_drops_its_outcome_counters(self, engine):
        new_id = engine.import_team_config(
            {"config": {"name": "Shared", "success_rate": 0.99, "usage_count": 7}}
        )
        data = engine.graph._get_node_properties(new_id)
        assert "success_rate" not in data and "usage_count" not in data


@pytest.mark.concept("CONCEPT:AU-AHE.evaluation.interpretability-tests")
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
        engine.graph.add_node(coalition.id, **_node_kwargs(coalition))

        result = engine.promote_coalition_to_template(
            coalition_id=coalition.id,
            task_pattern="repository analysis",
        )

        assert "id" in result
        assert result["task_pattern"] == "repository analysis"
        assert "success_rate" not in result

    def test_promote_creates_reused_team_edge(self, engine):
        """Should create a REUSED_TEAM edge from TeamConfig to coalition."""
        coalition = SwarmCoalitionNode(
            id="coalition:edge",
            name="Edge Test Coalition",
            agents_spawned=2,
        )
        engine.graph.add_node(coalition.id, **_node_kwargs(coalition))

        result = engine.promote_coalition_to_template(
            coalition_id=coalition.id,
            task_pattern="edge test",
        )
        tc_id = result["id"]

        # Check edge exists in NetworkX
        assert engine.graph.has_edge(tc_id, coalition.id)


@pytest.mark.concept("CONCEPT:AU-AHE.evaluation.interpretability-tests")
class TestLinkPromptToAgent:
    """Test suite for link_prompt_to_agent()."""

    def test_creates_uses_prompt_edge(self, engine):
        """Should create a USES_PROMPT edge."""
        engine.graph.add_node("agent:test", node_type="agent", name="Test Agent")
        engine.graph.add_node("prompt:test", node_type="prompt", name="Test Prompt")

        engine.link_prompt_to_agent("agent:test", "prompt:test")

        assert engine.graph.has_edge("agent:test", "prompt:test")
        edge_data = engine.graph.get_edge_data("agent:test", "prompt:test")
        # GraphComputeEngine.add_edge stores the relationship type verbatim
        # under the canonical ``relationship`` property key (add_edge hard-
        # rejects the retired ``type``/``rel_type``/... aliases outright), so
        # assert against that contract instead of a nonexistent "rel_type"
        # canonicalization. link_prompt_to_agent (with a backend attached, as
        # here) dispatches through IntelligenceGraphEngine.link_nodes, which
        # unconditionally upper-cases the relationship type to the
        # Cypher/Neo4j convention (engine.py's ``rel_type = rel_type.upper()``)
        # -- so the stored value is the canonical UPPER form, not the
        # lowercase ``RegistryEdgeType`` enum value.
        assert any(
            e.get("relationship") == RegistryEdgeType.USES_PROMPT.value.upper()
            for e in edge_data.values()
        )


@pytest.mark.concept("CONCEPT:AU-ORCH.adapter.hot-cache-invalidation")
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
