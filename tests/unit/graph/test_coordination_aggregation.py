#!/usr/bin/python
"""Tests for the coordination named-aggregation registry + select_winners consensus (b1-02).

CONCEPT:AU-ORCH.execution.execution-budget-caps
"""

import pytest

from agent_utilities.graph.coordination import (
    AggregationOperator,
    CoordinationLayer,
    aggregate_scores,
)
from agent_utilities.graph.workspace_attention import Proposal, WorkspaceAttention

pytestmark = pytest.mark.concept("AU-ORCH.execution.execution-budget-caps")


# --- aggregate_scores registry ---------------------------------------------


def test_aggregate_operators():
    vals = [0.2, 0.4, 0.9]
    assert aggregate_scores(vals, "mean") == pytest.approx(0.5)
    assert aggregate_scores(vals, AggregationOperator.MEDIAN) == pytest.approx(0.4)
    assert aggregate_scores(vals, "max") == 0.9
    assert aggregate_scores(vals, "min") == 0.2
    # log pool (geometric mean) sits below the arithmetic mean
    assert aggregate_scores(vals, "log_pool") < 0.5


def test_aggregate_empty_is_zero():
    assert aggregate_scores([], "mean") == 0.0


# --- CoordinationLayer API --------------------------------------------------


def test_coordination_layer_aggregate():
    layer = CoordinationLayer()
    assert layer.aggregate([1.0, 3.0], "mean") == pytest.approx(2.0)


def test_coordination_layer_rank_delegates_to_selection_registry():
    layer = CoordinationLayer()
    # a beats b, a beats c, b beats c → a should rank first
    ranked = layer.rank(
        ["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")], method="bradley_terry"
    )
    assert ranked[0][0] == "a"


# --- WorkspaceAttention consensus (live) -----------------------------------


def _prop(sid, composite):
    return Proposal(specialist_id=sid, output="o", composite_score=composite)


def test_consensus_score_uses_registry():
    wa = WorkspaceAttention()
    props = [_prop("a", 0.8), _prop("b", 0.4)]
    assert wa.consensus_score(props, "mean") == pytest.approx(0.6)
    assert wa.consensus_score(props, "max") == 0.8


def test_select_winners_still_topk_and_consensus_callable():
    wa = WorkspaceAttention(max_broadcast_slots=2)
    props = [_prop("a", 0.9), _prop("b", 0.5), _prop("c", 0.1)]
    winners = wa.select_winners(props)
    assert [w.specialist_id for w in winners] == ["a", "b"]  # unchanged top-K
    assert wa.consensus_score(winners) == pytest.approx(0.7)
