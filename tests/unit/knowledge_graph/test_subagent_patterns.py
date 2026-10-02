"""Tests for CONCEPT:AU-ORCH.execution.execution-budget-caps — Subagent Lifecycle Patterns."""

import pytest
from pydantic import ValidationError

from agent_utilities.graph.subagent_patterns import (
    PatternComplexity,
    SubagentPattern,
    SubagentPatternDecision,
    SubagentPatternRouter,
    get_infrastructure_mapping,
)


@pytest.fixture
def mock_engine():
    """Minimal mock engine for pattern router tests."""
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    class _MockEngine:
        def __init__(self):
            self.graph = GraphComputeEngine(backend_type="rust")
            self.backend = None

    return _MockEngine()


@pytest.fixture
def router(mock_engine):
    return SubagentPatternRouter(engine=mock_engine)


@pytest.fixture
def router_no_engine():
    return SubagentPatternRouter(engine=None)


# ── Pattern Selection Logic ────────────────────────────────────────────


class TestPatternSelection:
    """Tests for the pattern selection decision tree."""

    def test_trivial_task_selects_inline(self, router):
        decision = router.select_pattern(
            task_complexity=PatternComplexity.TRIVIAL,
            specialist_count=1,
        )
        assert decision.pattern == SubagentPattern.INLINE_TOOL
        assert decision.confidence > 0.8

    def test_simple_task_selects_inline(self, router):
        decision = router.select_pattern(
            task_complexity=PatternComplexity.SIMPLE,
            specialist_count=1,
        )
        assert decision.pattern == SubagentPattern.INLINE_TOOL

    def test_parallelizable_selects_fan_out(self, router):
        decision = router.select_pattern(
            task_complexity=PatternComplexity.MODERATE,
            parallelizable=True,
            specialist_count=3,
        )
        assert decision.pattern == SubagentPattern.FAN_OUT

    def test_collaboration_selects_agent_pool(self, router):
        decision = router.select_pattern(
            task_complexity=PatternComplexity.MODERATE,
            needs_collaboration=True,
        )
        assert decision.pattern == SubagentPattern.AGENT_POOL

    def test_expert_complexity_selects_teams(self, router):
        decision = router.select_pattern(
            task_complexity=PatternComplexity.EXPERT,
        )
        assert decision.pattern == SubagentPattern.TEAMS

    def test_a2a_peers_selects_teams(self, router):
        decision = router.select_pattern(
            task_complexity=PatternComplexity.MODERATE,
            has_a2a_peers=True,
        )
        assert decision.pattern == SubagentPattern.TEAMS

    def test_decision_has_reasoning(self, router):
        decision = router.select_pattern(task_complexity=PatternComplexity.SIMPLE)
        assert len(decision.reasoning) > 0

    def test_decision_has_timestamp(self, router):
        decision = router.select_pattern()
        assert decision.timestamp


# ── No outcome store ────────────────────────────────────────────────────


class TestNoOutcomeStore:
    """The router neither persists decisions nor learns from self-reported outcomes."""

    def test_selection_writes_nothing_to_the_graph(self, router, mock_engine):
        before = list(mock_engine.graph.nodes(data=True))
        router.select_pattern(task_complexity=PatternComplexity.SIMPLE)
        assert list(mock_engine.graph.nodes(data=True)) == before

    def test_history_in_the_graph_does_not_move_the_confidence(
        self, router, mock_engine
    ):
        for i in range(5):
            mock_engine.graph.add_node(
                f"hist_{i}",
                node_type="subagent_pattern_decision",
                pattern="inline_tool",
                outcome_success=False,
            )
        decision = router.select_pattern(task_complexity=PatternComplexity.SIMPLE)
        assert decision.confidence == 0.9

    def test_the_outcome_api_is_gone(self, router):
        assert not hasattr(router, "record_outcome")


# ── Infrastructure Mapping ─────────────────────────────────────────────


class TestInfrastructureMapping:
    """Tests for the pattern → infrastructure mapping."""

    def test_all_patterns_mapped(self):
        mapping = get_infrastructure_mapping()
        for pattern in SubagentPattern:
            assert pattern in mapping
            assert "module" in mapping[pattern]
            assert "class" in mapping[pattern]

    def test_inline_maps_to_executor(self):
        mapping = get_infrastructure_mapping()
        assert "executor" in mapping[SubagentPattern.INLINE_TOOL]["module"]

    def test_fan_out_maps_to_swarm(self):
        mapping = get_infrastructure_mapping()
        assert "orchestrator" in mapping[SubagentPattern.FAN_OUT]["module"].lower()


# ── Decision Model Validation ──────────────────────────────────────────


class TestDecisionModel:
    """Tests for SubagentPatternDecision Pydantic model."""

    def test_serialization(self):
        decision = SubagentPatternDecision(
            pattern=SubagentPattern.FAN_OUT,
            task_complexity=PatternComplexity.MODERATE,
            parallelizable=True,
            specialist_count=3,
            confidence=0.85,
            reasoning="Test reasoning",
        )
        data = decision.model_dump()
        assert data["pattern"] == "fan_out"
        assert data["task_complexity"] == 3
        assert data["confidence"] == 0.85

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            SubagentPatternDecision(
                pattern=SubagentPattern.INLINE_TOOL,
                task_complexity=PatternComplexity.SIMPLE,
                confidence=1.5,  # Out of bounds
            )
