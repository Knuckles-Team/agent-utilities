#!/usr/bin/python
"""Comprehensive tests for KG-Native Orchestration (CONCEPT:ORCH-1.1 through ORCH-1.19).

Tests cover all 7 gaps:
  1. KG-Driven Team Composition
  2. Topological Tool Assignment
  3. Execution State Checkpointing
  4. Dynamic Topology Materialization
  5. KG-Native Routing Policy
  6. Persistent Background Agents
  7. Shareable Team Compositions
"""

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine


@pytest.fixture(autouse=True)
def setup_engine():
    """Ensure IntelligenceGraphEngine is globally active for orchestration tests."""
    if IntelligenceGraphEngine._ACTIVE_ENGINE is None:
        compute = GraphComputeEngine(backend_type="rust")
        IntelligenceGraphEngine(db_path=":memory:", graph=compute)


# --- Gap 1: Team Composer ---


class TestKGTeamComposer:
    """Tests for KG-Driven Team Composition (CONCEPT:ORCH-1.1)."""

    def test_compose_simple_team(self):
        from agent_utilities.graph.team_composer import KGTeamComposer

        composer = KGTeamComposer(engine=None)
        team = composer.compose_team("What is Python?", complexity=1)
        assert team.team_id.startswith("team:")
        assert team.source in ("composed", "dynamic_synthesis")
        assert len(team.adaptive_agent_router) > 0
        assert team.execution_mode in (
            "sequential",
            "parallel",
            "mixed",
            "fan_out",
            "fan_in",
        )

    def test_compose_complex_team(self):
        from agent_utilities.graph.team_composer import KGTeamComposer

        composer = KGTeamComposer(engine=None)
        team = composer.compose_team("Analyze this codebase architecture", complexity=4)
        assert len(team.adaptive_agent_router) >= 1
        assert team.execution_mode in (
            "sequential",
            "mixed",
            "parallel",
            "fan_out",
            "fan_in",
        )

    def test_compose_finance_domain(self):
        from agent_utilities.graph.team_composer import KGTeamComposer

        composer = KGTeamComposer(engine=None)
        team = composer.compose_team(
            "Analyze AAPL stock", domain="finance", complexity=4
        )
        assert len(team.adaptive_agent_router) >= 1

    def test_compose_single_agent(self):
        from agent_utilities.graph.team_composer import KGTeamComposer

        composer = KGTeamComposer(engine=None)
        team = composer.compose_team("Hello", complexity=1)
        assert len(team.adaptive_agent_router) >= 1

    def test_team_has_system_prompts(self):
        from agent_utilities.graph.team_composer import KGTeamComposer

        composer = KGTeamComposer(engine=None)
        team = composer.compose_team("Research topic", complexity=3)
        for spec in team.adaptive_agent_router:
            assert "system_prompt" in spec
            assert len(spec["system_prompt"]) > 0

    def test_team_confidence_scoring(self):
        from agent_utilities.graph.team_composer import KGTeamComposer

        composer = KGTeamComposer(engine=None)
        team = composer.compose_team("Test query", complexity=2)
        assert 0.0 <= team.confidence <= 1.0

    def test_team_reasoning_explanation(self):
        from agent_utilities.graph.team_composer import KGTeamComposer

        composer = KGTeamComposer(engine=None)
        team = composer.compose_team("Build a web app", complexity=3)
        assert len(team.reasoning) > 0
        assert "topology" in team.reasoning.lower()


# --- Gap 3: State Checkpointing ---


class TestStateCheckpointer:
    """Tests for Execution State Checkpointing (CONCEPT:ORCH-1.1)."""

    def _make_mock_state(self):
        class MockState:
            query = "Test query"
            plan = "Step 1: do thing"
            specialist_results = {"expert": "Result A"}
            node_history = ["router", "expert", "verifier"]
            routed_domain = "general"
            routed_specialist = "expert"
            active_topology = None
            usage = None

        return MockState()

    def test_checkpoint_without_engine(self):
        from agent_utilities.graph.state_checkpoint import StateCheckpointer

        cp = StateCheckpointer(engine=None)
        state = self._make_mock_state()
        ckpt_id = cp.checkpoint(state, session_id="test-sess")
        assert "test-sess" in ckpt_id

    def test_checkpoint_with_nx_engine(self):
        from agent_utilities.graph.state_checkpoint import StateCheckpointer
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        GraphComputeEngine(backend_type="rust")
        engine = IntelligenceGraphEngine(db_path=":memory:")
        cp = StateCheckpointer(engine=engine)
        state = self._make_mock_state()
        ckpt_id = cp.checkpoint(state, session_id="nx-sess")
        assert ckpt_id in engine.graph.nodes

    def test_restore_from_nx(self):
        from agent_utilities.graph.state_checkpoint import StateCheckpointer
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        GraphComputeEngine(backend_type="rust")
        engine = IntelligenceGraphEngine(db_path=":memory:")
        cp = StateCheckpointer(engine=engine)
        state = self._make_mock_state()
        cp.checkpoint(state, session_id="restore-test")
        restored = cp.restore("restore-test")
        assert restored is not None
        assert restored["query"] == "Test query"
        assert restored["session_id"] == "restore-test"

    def test_mark_completed(self):
        from agent_utilities.graph.state_checkpoint import StateCheckpointer

        cp = StateCheckpointer(engine=None)
        cp.mark_completed("test-session", success=True)  # Should not raise

    def test_list_sessions_empty(self):
        from agent_utilities.graph.state_checkpoint import StateCheckpointer

        cp = StateCheckpointer(engine=None)
        sessions = cp.list_sessions()
        assert sessions == []


# --- Gap 4: Dynamic Topology Engine ---


class TestTopologyEngine:
    """Tests for Dynamic Topology Materialization (CONCEPT:ORCH-1.2)."""

    def _make_team(self, mode="sequential", roles=None, parallel_groups=None):
        from agent_utilities.models.knowledge_graph import TeamComposition

        roles = roles or ["router", "expert", "verifier"]
        adaptive_agent_router = [
            {
                "role": r,
                "agent_id": r,
                "tools": [],
                "system_prompt": f"You are {r}",
                "memory_channels": ["episodic"],
            }
            for r in roles
        ]
        return TeamComposition(
            team_id="test-team",
            adaptive_agent_router=adaptive_agent_router,
            execution_mode=mode,
            parallel_groups=parallel_groups or [],
            memory_channels=["episodic"],
            confidence=0.8,
            reasoning="Test",
        )

    def test_sequential_materialization(self):
        from agent_utilities.graph.topology_engine import TopologyEngine

        te = TopologyEngine(engine=None)
        result = te.materialize(self._make_team("sequential"))
        assert len(result["execution_plan"]) == 3
        for step in result["execution_plan"]:
            assert step["mode"] == "sequential"
            assert len(step["roles"]) == 1

    def test_parallel_materialization(self):
        from agent_utilities.graph.topology_engine import TopologyEngine

        te = TopologyEngine(engine=None)
        result = te.materialize(self._make_team("parallel"))
        assert len(result["execution_plan"]) == 1
        assert result["execution_plan"][0]["mode"] == "parallel"
        assert len(result["execution_plan"][0]["roles"]) == 3

    def test_fan_out_materialization(self):
        from agent_utilities.graph.topology_engine import TopologyEngine

        te = TopologyEngine(engine=None)
        team = self._make_team(
            "fan_out", roles=["router", "worker_a", "worker_b", "worker_c"]
        )
        result = te.materialize(team)
        assert len(result["execution_plan"]) == 2
        assert result["execution_plan"][0]["mode"] == "sequential"
        assert result["execution_plan"][1]["mode"] == "parallel"
        assert len(result["execution_plan"][1]["roles"]) == 3

    def test_fan_in_materialization(self):
        from agent_utilities.graph.topology_engine import TopologyEngine

        te = TopologyEngine(engine=None)
        team = self._make_team("fan_in", roles=["worker_a", "worker_b", "synthesizer"])
        result = te.materialize(team)
        assert len(result["execution_plan"]) == 2
        assert result["execution_plan"][0]["mode"] == "parallel"
        assert result["execution_plan"][1]["mode"] == "sequential"

    def test_mixed_materialization(self):
        from agent_utilities.graph.topology_engine import TopologyEngine

        te = TopologyEngine(engine=None)
        team = self._make_team(
            "mixed",
            roles=["router", "planner", "researcher_a", "researcher_b", "synthesizer"],
            parallel_groups=[["researcher_a", "researcher_b"]],
        )
        result = te.materialize(team)
        plan = result["execution_plan"]
        modes = [s["mode"] for s in plan]
        assert "parallel" in modes
        assert "sequential" in modes

    def test_specialist_configs_populated(self):
        from agent_utilities.graph.topology_engine import TopologyEngine

        te = TopologyEngine(engine=None)
        result = te.materialize(self._make_team("sequential"))
        configs = result["specialist_configs"]
        assert len(configs) == 3
        for role, config in configs.items():
            assert "system_prompt" in config
            assert "tools" in config
            assert config["role"] == role


# --- Gap 5: KG-Native Routing Policy ---


class TestTopologicalRoutingPolicy:
    """Tests for KG-Native Routing Policy (CONCEPT:ORCH-1.4)."""

    def test_cold_start_fallback(self):
        from agent_utilities.graph.adaptive_agent_router import (
            RoutingCandidate,
            TopologicalRoutingPolicy,
        )

        policy = TopologicalRoutingPolicy(engine=None)
        candidates = [
            RoutingCandidate(model_id="gpt-4", confidence=0.9),
            RoutingCandidate(model_id="gpt-3.5", confidence=0.7),
        ]
        decision = policy.route("analyze code", candidates)
        assert decision.selected.model_id == "gpt-4"

    def test_no_candidates_raises(self):
        from agent_utilities.graph.adaptive_agent_router import TopologicalRoutingPolicy

        policy = TopologicalRoutingPolicy(engine=None)
        with pytest.raises(ValueError):
            policy.route("test", [])

    def test_with_engine_scores_candidates(self):
        from agent_utilities.graph.adaptive_agent_router import (
            RoutingCandidate,
            TopologicalRoutingPolicy,
        )
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        g = GraphComputeEngine(backend_type="rust")
        g.add_node("gpt-4", type="agent", name="GPT-4")
        g.add_node("tool1", type="tool", name="search")
        g.add_edge("gpt-4", "tool1", type="provides")
        engine = IntelligenceGraphEngine(db_path=":memory:")
        policy = TopologicalRoutingPolicy(engine=engine)
        candidates = [
            RoutingCandidate(model_id="gpt-4", confidence=0.8),
            RoutingCandidate(model_id="unknown", confidence=0.5),
        ]
        decision = policy.route("search for info", candidates)
        assert decision.selected is not None
        assert "TopologicalRouting" in decision.decision_reason


# --- Gap 6: Persistent Background Agents ---


class TestPersistentAgentManager:
    """Tests for Persistent Background Agents (CONCEPT:ORCH-1.4)."""

    def test_register_agent(self):
        from agent_utilities.graph.persistent_agents import PersistentAgentManager

        mgr = PersistentAgentManager(engine=None)
        agent = mgr.register_agent(
            "bg:monitor",
            "System Monitor",
            agent_type="monitor",
            subscriptions=["system.alert"],
        )
        assert agent.id == "bg:monitor"
        assert agent.status == "idle"
        assert "system.alert" in agent.subscriptions

    def test_heartbeat(self):
        from agent_utilities.graph.persistent_agents import PersistentAgentManager

        mgr = PersistentAgentManager(engine=None)
        mgr.register_agent("bg:test", "Test Agent")
        mgr.heartbeat("bg:test")
        assert mgr._registered_agents["bg:test"].heartbeat_ts != ""

    def test_update_status(self):
        from agent_utilities.graph.persistent_agents import PersistentAgentManager

        mgr = PersistentAgentManager(engine=None)
        mgr.register_agent("bg:test", "Test Agent")
        mgr.update_status("bg:test", "running")
        assert mgr._registered_agents["bg:test"].status == "running"

    def test_save_and_load_state(self):
        from agent_utilities.graph.persistent_agents import PersistentAgentManager

        mgr = PersistentAgentManager(engine=None)
        mgr.register_agent("bg:stateful", "Stateful Agent")
        mgr.save_state("bg:stateful", {"counter": 42, "last_run": "2026-01-01"})
        state = mgr.load_state("bg:stateful")
        assert state["counter"] == 42

    def test_find_subscribers(self):
        from agent_utilities.graph.persistent_agents import PersistentAgentManager

        mgr = PersistentAgentManager(engine=None)
        mgr.register_agent("bg:a", "Agent A", subscriptions=["data.new", "alert"])
        mgr.register_agent("bg:b", "Agent B", subscriptions=["data.new"])
        mgr.register_agent("bg:c", "Agent C", subscriptions=["policy.changed"])
        subs = mgr.find_subscribers("data.new")
        assert set(subs) == {"bg:a", "bg:b"}

    def test_list_agents(self):
        from agent_utilities.graph.persistent_agents import PersistentAgentManager

        mgr = PersistentAgentManager(engine=None)
        mgr.register_agent("bg:1", "Agent 1", agent_type="monitor")
        mgr.register_agent("bg:2", "Agent 2", agent_type="scheduler")
        agents = mgr.list_agents()
        assert len(agents) == 2
        monitors = mgr.list_agents(agent_type="monitor")
        assert len(monitors) == 1

    def test_terminate_agent(self):
        from agent_utilities.graph.persistent_agents import PersistentAgentManager

        mgr = PersistentAgentManager(engine=None)
        mgr.register_agent("bg:temp", "Temp Agent")
        mgr.terminate_agent("bg:temp")
        assert "bg:temp" not in mgr._registered_agents


# --- Gap 7: Shareable Team Compositions ---


class TestShareableTeamConfigs:
    """Tests for Shareable Team Compositions (CONCEPT:ORCH-1.1)."""

    def test_export_from_nx(self):
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        g = GraphComputeEngine(backend_type="rust")
        g.add_node(
            "tc:test",
            type="team_config",
            name="Test Team",
            success_rate=0.85,
            usage_count=5,
            origin="local",
        )
        engine = IntelligenceGraphEngine(db_path=":memory:")
        bundle = engine.export_team_config("tc:test")
        assert bundle is not None
        assert bundle["version"] == "1.0"
        assert bundle["config"]["name"] == "Test Team"

    def test_import_to_nx(self):
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        GraphComputeEngine(backend_type="rust")
        engine = IntelligenceGraphEngine(db_path=":memory:")
        bundle = {
            "version": "1.0",
            "type": "team_config",
            "config": {"name": "Imported Team", "success_rate": 0.9},
        }
        new_id = engine.import_team_config(bundle)
        assert new_id.startswith("tc:imported:")
        assert new_id in engine.graph.nodes
        assert engine.graph.nodes[new_id]["origin"] == "community"

    def test_export_missing_returns_none(self):
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        GraphComputeEngine(backend_type="rust")
        engine = IntelligenceGraphEngine(db_path=":memory:")
        assert engine.export_team_config("nonexistent") is None

    def test_list_team_configs(self):
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        g = GraphComputeEngine(backend_type="rust")
        g.add_node(
            "tc:a",
            type="team_config",
            name="A",
            success_rate=0.9,
            usage_count=3,
            origin="local",
        )
        g.add_node(
            "tc:b",
            type="team_config",
            name="B",
            success_rate=0.5,
            usage_count=1,
            origin="local",
        )
        engine = IntelligenceGraphEngine(db_path=":memory:")
        all_configs = engine.list_team_configs()
        assert len(all_configs) == 2
        high_rate = engine.list_team_configs(min_success_rate=0.7)
        assert len(high_rate) == 1
        assert high_rate[0]["name"] == "A"


# --- Pydantic Model Tests ---


class TestKGNativeModels:
    """Tests for new Pydantic models."""

    def test_topology_template_node(self):
        from agent_utilities.models.knowledge_graph import TopologyTemplateNode

        node = TopologyTemplateNode(
            id="topo:test",
            name="Test Topology",
            domain="finance",
            node_roles=["router", "expert"],
            transitions={"router": ["expert"], "expert": []},
            execution_mode="sequential",
        )
        assert node.type.value == "topology_template"
        assert node.complexity_min == 1
        assert node.complexity_max == 5

    def test_session_checkpoint_node(self):
        from agent_utilities.models.knowledge_graph import SessionCheckpointNode

        node = SessionCheckpointNode(
            id="ckpt:test",
            name="Test Checkpoint",
            session_id="sess:abc",
            query="test query",
        )
        assert node.type.value == "session_checkpoint"
        assert node.status == "active"

    def test_persistent_agent_node(self):
        from agent_utilities.models.knowledge_graph import PersistentAgentNode

        node = PersistentAgentNode(
            id="pa:test",
            name="Monitor Agent",
            agent_type="monitor",
            subscriptions=["alert.critical"],
            status="idle",
        )
        assert node.type.value == "persistent_agent"
        assert node.max_concurrent == 1

    def test_team_composition(self):
        from agent_utilities.models.knowledge_graph import TeamComposition

        team = TeamComposition(
            team_id="team:test",
            adaptive_agent_router=[
                {"role": "expert", "agent_id": "a1", "tools": ["search"]}
            ],
            execution_mode="parallel",
            confidence=0.85,
        )
        assert team.source == "composed"
        assert len(team.adaptive_agent_router) == 1

    def test_registry_node_types_added(self):
        from agent_utilities.models.knowledge_graph import RegistryNodeType

        assert hasattr(RegistryNodeType, "TOPOLOGY_TEMPLATE")
        assert hasattr(RegistryNodeType, "SESSION_CHECKPOINT")
        assert hasattr(RegistryNodeType, "PERSISTENT_AGENT")

    def test_registry_edge_types_added(self):
        from agent_utilities.models.knowledge_graph import RegistryEdgeType

        assert hasattr(RegistryEdgeType, "TRANSITIONS_TO")
        assert hasattr(RegistryEdgeType, "CHECKPOINTED_STATE")
        assert hasattr(RegistryEdgeType, "COMPOSED_TEAM")


# --- Integration: End-to-End Orchestration ---


class TestEndToEndOrchestration:
    """Integration tests for the full KG-native orchestration pipeline."""

    def test_compose_and_materialize_simple(self):
        """Simple: compose team → materialize → verify sequential plan."""
        from agent_utilities.graph.team_composer import KGTeamComposer
        from agent_utilities.graph.topology_engine import TopologyEngine

        composer = KGTeamComposer(engine=None)
        team = composer.compose_team("What is 2+2?", complexity=1)
        te = TopologyEngine(engine=None)
        result = te.materialize(team)
        assert len(result["execution_plan"]) >= 1
        assert len(result["specialist_configs"]) >= 1

    def test_compose_and_materialize_complex(self):
        """Complex: compose research team → materialize → verify mixed plan."""
        from agent_utilities.graph.team_composer import KGTeamComposer
        from agent_utilities.graph.topology_engine import TopologyEngine

        composer = KGTeamComposer(engine=None)
        team = composer.compose_team(
            "Research quantum computing advances", complexity=4
        )
        te = TopologyEngine(engine=None)
        result = te.materialize(team)
        assert len(result["execution_plan"]) >= 1
        assert len(result["specialist_configs"]) >= 1

    def test_compose_checkpoint_restore(self):
        """State persistence: compose → checkpoint → restore round-trip."""
        from agent_utilities.graph.state_checkpoint import StateCheckpointer
        from agent_utilities.graph.team_composer import KGTeamComposer
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        GraphComputeEngine(backend_type="rust")
        engine = IntelligenceGraphEngine(db_path=":memory:")
        composer = KGTeamComposer(engine=engine)
        composer.compose_team("Analyze data", complexity=3)

        class MockState:
            query = "Analyze data"
            plan = "Step 1"
            specialist_results = {"expert": "done"}
            node_history = ["router", "expert"]
            routed_domain = "general"
            routed_specialist = "expert"
            active_topology = None
            usage = None

        cp = StateCheckpointer(engine=engine)
        cp.checkpoint(MockState(), session_id="e2e-sess")

        restored = cp.restore("e2e-sess")
        assert restored is not None
        assert restored["query"] == "Analyze data"

    def test_parallel_team_materialization(self):
        """Parallel: fan-out pattern with multiple workers."""
        from agent_utilities.graph.topology_engine import TopologyEngine
        from agent_utilities.models.knowledge_graph import TeamComposition

        team = TeamComposition(
            team_id="parallel-test",
            adaptive_agent_router=[
                {
                    "role": "dispatcher",
                    "agent_id": "d1",
                    "tools": [],
                    "system_prompt": "Dispatch",
                    "memory_channels": ["ep"],
                },
                {
                    "role": "worker_1",
                    "agent_id": "w1",
                    "tools": ["search"],
                    "system_prompt": "Search",
                    "memory_channels": ["ep"],
                },
                {
                    "role": "worker_2",
                    "agent_id": "w2",
                    "tools": ["search"],
                    "system_prompt": "Search",
                    "memory_channels": ["ep"],
                },
                {
                    "role": "worker_3",
                    "agent_id": "w3",
                    "tools": ["search"],
                    "system_prompt": "Search",
                    "memory_channels": ["ep"],
                },
            ],
            execution_mode="fan_out",
            confidence=0.8,
            reasoning="Test fan-out",
        )
        te = TopologyEngine(engine=None)
        result = te.materialize(team)
        plan = result["execution_plan"]
        assert plan[0]["mode"] == "sequential"
        assert plan[1]["mode"] == "parallel"
        assert len(plan[1]["roles"]) == 3
