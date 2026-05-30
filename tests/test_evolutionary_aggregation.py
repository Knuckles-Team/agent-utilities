#!/usr/bin/python
from __future__ import annotations

"""Tests for CONCEPT:AHE-3.2 — Evolutionary Aggregation Engine."""


import pytest
from pydantic import ValidationError

from agent_utilities.graph.hierarchical_planner import (
    AggregationStrategy,
    ConvergenceMonitor,
    EvolutionaryAggregator,
    GroupFitness,
)
from agent_utilities.graph.workspace_attention import (
    Proposal,
    WorkspaceAttention,
)


def _make_proposal(
    specialist_id: str = "spec:a",
    output: str = "Answer A",
    confidence: float = 0.8,
    composite: float = 0.7,
) -> Proposal:
    """Build a test proposal."""
    return Proposal(
        specialist_id=specialist_id,
        output=output,
        confidence_score=confidence,
        composite_score=composite,
    )


# ── GroupFitness Model ────────────────────────────────────────────────


class TestGroupFitness:
    """Tests for the GroupFitness Pydantic model."""

    def test_default_values(self):
        gf = GroupFitness(group_id="test")
        assert gf.group_confidence == 0.5
        assert gf.group_diversity == 1
        assert gf.recommended_strategy == AggregationStrategy.LIGHT_MODEL

    def test_custom_values(self):
        gf = GroupFitness(
            group_id="g1",
            specialist_ids=["spec:a", "spec:b"],
            group_confidence=0.9,
            group_diversity=1,
            recommended_strategy=AggregationStrategy.MAJORITY_VOTE,
        )
        assert gf.group_confidence == 0.9
        assert len(gf.specialist_ids) == 2

    def test_timestamp_auto_generated(self):
        gf = GroupFitness(group_id="t")
        assert gf.timestamp  # Should be non-empty

    def test_confidence_clamped(self):
        with pytest.raises(ValidationError):
            GroupFitness(group_id="bad", group_confidence=1.5)


# ── ConvergenceMonitor ────────────────────────────────────────────────


class TestConvergenceMonitor:
    """Tests for the ConvergenceMonitor."""

    def test_no_convergence_with_high_diversity(self):
        monitor = ConvergenceMonitor(convergence_threshold=0.1, patience=3)
        assert monitor.update(0.8) is False
        assert monitor.update(0.7) is False
        assert monitor.update(0.6) is False

    def test_convergence_after_patience(self):
        monitor = ConvergenceMonitor(convergence_threshold=0.2, patience=3)
        assert monitor.update(0.1) is False
        assert monitor.update(0.05) is False
        assert monitor.update(0.08) is True  # 3rd consecutive low → converged

    def test_high_diversity_resets_counter(self):
        monitor = ConvergenceMonitor(convergence_threshold=0.2, patience=3)
        assert monitor.update(0.1) is False
        assert monitor.update(0.05) is False
        assert monitor.update(0.5) is False  # High diversity resets
        assert monitor.update(0.1) is False  # Counter back to 1

    def test_reset_clears_state(self):
        monitor = ConvergenceMonitor(convergence_threshold=0.2, patience=2)
        monitor.update(0.1)
        monitor.update(0.05)
        monitor.reset()
        assert monitor.history == []
        assert monitor.update(0.1) is False  # Counter reset

    def test_history_tracked(self):
        monitor = ConvergenceMonitor()
        monitor.update(0.5)
        monitor.update(0.3)
        monitor.update(0.1)
        assert len(monitor.history) == 3
        assert monitor.history == [0.5, 0.3, 0.1]


# ── EvolutionaryAggregator ───────────────────────────────────────────


class TestEvolutionaryAggregator:
    """Tests for the EvolutionaryAggregator."""

    def test_compute_group_fitness_high_confidence_low_diversity(self):
        """All adaptive_agent_router agree with high confidence → MAJORITY_VOTE."""
        agg = EvolutionaryAggregator(confidence_threshold=0.7, diversity_threshold=2)
        proposals = [
            _make_proposal("spec:a", "Same answer here", 0.9),
            _make_proposal("spec:b", "Same answer here", 0.85),
        ]
        fitness = agg.compute_group_fitness(proposals, "g1")
        assert fitness.group_confidence == pytest.approx(0.875, abs=0.01)
        assert fitness.group_diversity == 1
        assert fitness.recommended_strategy == AggregationStrategy.MAJORITY_VOTE

    def test_compute_group_fitness_high_confidence_some_diversity(self):
        """High confidence but diverse answers → LIGHT_MODEL."""
        agg = EvolutionaryAggregator(confidence_threshold=0.7, diversity_threshold=2)
        proposals = [
            _make_proposal("spec:a", "Answer A about projects", 0.85),
            _make_proposal("spec:b", "Answer B about databases", 0.80),
        ]
        fitness = agg.compute_group_fitness(proposals, "g2")
        assert fitness.group_confidence > 0.7
        assert fitness.group_diversity == 2
        assert fitness.recommended_strategy == AggregationStrategy.LIGHT_MODEL

    def test_compute_group_fitness_low_confidence_high_diversity(self):
        """Low confidence + high diversity → HEAVY_MODEL."""
        agg = EvolutionaryAggregator(confidence_threshold=0.7, diversity_threshold=2)
        proposals = [
            _make_proposal("spec:a", "Answer A about projects", 0.3),
            _make_proposal("spec:b", "Answer B about databases", 0.2),
            _make_proposal("spec:c", "Answer C about security", 0.4),
        ]
        fitness = agg.compute_group_fitness(proposals, "g3")
        assert fitness.group_confidence < 0.7
        assert fitness.group_diversity >= 2
        assert fitness.recommended_strategy == AggregationStrategy.HEAVY_MODEL

    def test_empty_proposals_returns_heavy(self):
        """Empty group → conservative HEAVY_MODEL strategy."""
        agg = EvolutionaryAggregator()
        fitness = agg.compute_group_fitness([], "empty")
        assert fitness.group_confidence == 0.0
        assert fitness.group_diversity == 0
        assert fitness.recommended_strategy == AggregationStrategy.HEAVY_MODEL

    def test_group_proposals_splits_correctly(self):
        """Proposals should be split into groups of group_size."""
        agg = EvolutionaryAggregator(group_size=2, population_size=6)
        proposals = [_make_proposal(f"spec:{i}", f"Output {i}") for i in range(5)]
        groups = agg.group_proposals(proposals)
        assert len(groups) == 3
        assert len(groups[0]) == 2
        assert len(groups[1]) == 2
        assert len(groups[2]) == 1  # Remainder

    def test_population_size_limits_input(self):
        """population_size should cap the number of proposals considered."""
        agg = EvolutionaryAggregator(group_size=2, population_size=4)
        proposals = [_make_proposal(f"spec:{i}", f"Output {i}") for i in range(10)]
        groups = agg.group_proposals(proposals)
        total = sum(len(g) for g in groups)
        assert total == 4  # Only 4 proposals considered

    def test_route_aggregation_returns_strategies(self):
        """route_aggregation should return a map of group_id → strategy."""
        agg = EvolutionaryAggregator()
        groups = [
            GroupFitness(
                group_id="g1",
                group_confidence=0.9,
                group_diversity=1,
                recommended_strategy=AggregationStrategy.MAJORITY_VOTE,
            ),
            GroupFitness(
                group_id="g2",
                group_confidence=0.3,
                group_diversity=3,
                recommended_strategy=AggregationStrategy.HEAVY_MODEL,
            ),
        ]
        routes = agg.route_aggregation(groups)
        assert routes["g1"] == AggregationStrategy.MAJORITY_VOTE
        assert routes["g2"] == AggregationStrategy.HEAVY_MODEL


# ── WorkspaceAttention Group Methods ─────────────────────────────────


class TestGroupConfidence:
    """Tests for WorkspaceAttention.compute_group_confidence()."""

    def test_mean_confidence(self):
        gwt = WorkspaceAttention()
        proposals = [
            _make_proposal("spec:a", "Out A", 0.8),
            _make_proposal("spec:b", "Out B", 0.6),
        ]
        gc = gwt.compute_group_confidence(proposals)
        assert gc == pytest.approx(0.7, abs=0.01)

    def test_empty_returns_neutral(self):
        gwt = WorkspaceAttention()
        assert gwt.compute_group_confidence([]) == 0.5

    def test_single_proposal(self):
        gwt = WorkspaceAttention()
        proposals = [_make_proposal("spec:a", "Out", 0.9)]
        assert gwt.compute_group_confidence(proposals) == 0.9


class TestGroupDiversity:
    """Tests for WorkspaceAttention.compute_group_diversity()."""

    def test_identical_outputs_diversity_one(self):
        gwt = WorkspaceAttention()
        proposals = [
            _make_proposal("spec:a", "Same answer"),
            _make_proposal("spec:b", "Same answer"),
        ]
        assert gwt.compute_group_diversity(proposals) == 1

    def test_different_outputs_counted(self):
        gwt = WorkspaceAttention()
        proposals = [
            _make_proposal("spec:a", "Answer about projects"),
            _make_proposal("spec:b", "Answer about databases"),
            _make_proposal("spec:c", "Answer about security"),
        ]
        assert gwt.compute_group_diversity(proposals) == 3

    def test_empty_proposals(self):
        gwt = WorkspaceAttention()
        assert gwt.compute_group_diversity([]) == 0

    def test_case_insensitive(self):
        gwt = WorkspaceAttention()
        proposals = [
            _make_proposal("spec:a", "Same Answer Here"),
            _make_proposal("spec:b", "same answer here"),
        ]
        assert gwt.compute_group_diversity(proposals) == 1


# ── AggregationStrategy Enum ─────────────────────────────────────────


class TestAggregationStrategy:
    """Tests for the AggregationStrategy enum."""

    def test_values(self):
        assert AggregationStrategy.MAJORITY_VOTE == "majority_vote"
        assert AggregationStrategy.LIGHT_MODEL == "light_model"
        assert AggregationStrategy.HEAVY_MODEL == "heavy_model"

    def test_is_str_enum(self):
        assert isinstance(AggregationStrategy.MAJORITY_VOTE, str)
