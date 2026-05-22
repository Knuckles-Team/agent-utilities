"""Tests for CONCEPT:ORCH-1.3 — Subagent Lifecycle Patterns."""

import pytest

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
    import networkx as nx

    class _MockEngine:
        def __init__(self):
            self.graph = nx.MultiDiGraph()
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


# ── Decision Persistence ───────────────────────────────────────────────


class TestDecisionPersistence:
    """Tests for KG persistence of pattern decisions."""

    def test_decision_persisted_to_graph(self, router, mock_engine):
        router.select_pattern(task_complexity=PatternComplexity.SIMPLE)

        # Check that a decision node was added
        decision_nodes = [
            (nid, data)
            for nid, data in mock_engine.graph.nodes(data=True)
            if data.get("type") == "subagent_pattern_decision"
        ]
        assert len(decision_nodes) == 1
        assert decision_nodes[0][1]["pattern"] == SubagentPattern.INLINE_TOOL.value

    def test_no_persistence_without_engine(self, router_no_engine):
        decision = router_no_engine.select_pattern()
        assert decision.pattern  # Should still work, just no persistence


# ── Outcome Recording ──────────────────────────────────────────────────


class TestOutcomeRecording:
    """Tests for pattern outcome learning."""

    def test_record_success_outcome(self, router, mock_engine):
        decision = router.select_pattern(task_complexity=PatternComplexity.SIMPLE)
        router.record_outcome(decision, success=True, duration_ms=150.0)

        # Verify outcome was recorded in graph
        for nid, data in mock_engine.graph.nodes(data=True):
            if data.get("type") == "subagent_pattern_decision":
                assert data["outcome_success"] is True
                assert data["outcome_duration_ms"] == 150.0

    def test_record_failure_outcome(self, router, mock_engine):
        decision = router.select_pattern(task_complexity=PatternComplexity.MODERATE)
        router.record_outcome(decision, success=False, duration_ms=5000.0)

        for nid, data in mock_engine.graph.nodes(data=True):
            if data.get("type") == "subagent_pattern_decision":
                assert data["outcome_success"] is False


# ── Historical Adjustment ──────────────────────────────────────────────


class TestHistoricalAdjustment:
    """Tests for confidence adjustment from historical data."""

    def test_confidence_adjusts_with_history(self, router, mock_engine):
        # Add 5 historical successful inline decisions
        for i in range(5):
            mock_engine.graph.add_node(
                f"hist_{i}",
                type="subagent_pattern_decision",
                pattern="inline_tool",
                outcome_success=True,
            )

        decision = router.select_pattern(task_complexity=PatternComplexity.SIMPLE)
        # Confidence should be adjusted from historical 100% success rate
        assert decision.confidence > 0.85

    def test_low_history_raises_no_error(self, router, mock_engine):
        # Only 1 historical decision (below min_sample_size of 3)
        mock_engine.graph.add_node(
            "hist_0",
            type="subagent_pattern_decision",
            pattern="inline_tool",
            outcome_success=True,
        )
        decision = router.select_pattern(task_complexity=PatternComplexity.SIMPLE)
        assert decision.confidence > 0.0


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
        with pytest.raises(Exception):
            SubagentPatternDecision(
                pattern=SubagentPattern.INLINE_TOOL,
                task_complexity=PatternComplexity.SIMPLE,
                confidence=1.5,  # Out of bounds
            )
