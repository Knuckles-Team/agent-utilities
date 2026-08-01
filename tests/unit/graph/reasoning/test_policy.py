"""Unit tests for the cost-aware topology escalation policy."""

import pytest

from agent_utilities.graph.reasoning.benchmark import TopologyStats
from agent_utilities.graph.reasoning.policy import (
    DEFAULT_LADDER,
    EscalationPolicy,
)


def test_initial_choice_is_the_cheapest_rung():
    policy = EscalationPolicy()
    decision = policy.choose()
    assert decision.topology == DEFAULT_LADDER[0]
    assert decision.rung == 0
    assert decision.reason == "initial"


def test_needs_tools_routes_to_react_regardless_of_rung():
    policy = EscalationPolicy()
    decision = policy.choose(needs_tools=True, current_rung=3)
    assert decision.topology == "react"
    assert decision.reason == "needs_tools"


def test_low_confidence_escalates_one_rung():
    policy = EscalationPolicy(confidence_floor=0.6)
    decision = policy.choose(prior_confidence=0.3, current_rung=0)
    assert decision.rung == 1
    assert decision.topology == DEFAULT_LADDER[1]
    assert decision.reason == "low_confidence"


def test_adequate_confidence_never_escalates():
    policy = EscalationPolicy(confidence_floor=0.6)
    decision = policy.choose(prior_confidence=0.9, current_rung=0)
    assert decision.rung == 0
    assert decision.reason == "adequate"


def test_prior_failure_escalates_via_reliability():
    policy = EscalationPolicy(reliability_floor=0.6)
    stats = TopologyStats(topology="cot", n=5, reliability=0.2)
    decision = policy.choose(prior_stats=stats, current_rung=0)
    assert decision.rung == 1
    assert decision.reason == "prior_failure"


def test_never_escalates_past_the_top_of_the_ladder():
    policy = EscalationPolicy()
    top_rung = len(DEFAULT_LADDER) - 1
    decision = policy.choose(prior_confidence=0.0, current_rung=top_rung)
    assert decision.rung == top_rung
    assert decision.topology == DEFAULT_LADDER[top_rung]


def test_custom_ladder_is_respected():
    policy = EscalationPolicy(ladder=("cot", "rap"))
    decision = policy.choose()
    assert decision.topology == "cot"
    decision2 = policy.choose(prior_confidence=0.0, current_rung=0)
    assert decision2.topology == "rap"


def test_empty_ladder_rejected():
    with pytest.raises(ValueError):
        EscalationPolicy(ladder=())
