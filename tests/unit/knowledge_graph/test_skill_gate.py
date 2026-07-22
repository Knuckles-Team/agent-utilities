"""Benchmark-gated promotion — the pure ``skill_gate`` decision function (CONCEPT:
AU-AHE.optimization.skillopt-native-reflact). Direct port of SkillOpt's
``skillopt/evaluation/gate.py::evaluate_gate`` comparison (arXiv:2605.23904).

@pytest.mark.concept("AU-AHE.optimization.skillopt-native-reflact")
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.research.skill_gate import (
    evaluate_promotion,
    skill_gate,
)

pytestmark = pytest.mark.concept("AU-AHE.optimization.skillopt-native-reflact")


def test_strictly_better_candidate_promotes():
    assert evaluate_promotion(0.8, 0.6) is True


def test_tie_never_promotes():
    """A tie must NOT promote — matches SkillOpt's strict ``>`` comparison exactly."""
    assert evaluate_promotion(0.7, 0.7) is False


def test_worse_candidate_never_promotes():
    assert evaluate_promotion(0.4, 0.6) is False


def test_skill_gate_accept_new_best():
    result = skill_gate(0.9, incumbent_score=0.6, best_score=0.7)
    assert result.action == "accept_new_best"


def test_skill_gate_accept_not_new_best():
    result = skill_gate(0.75, incumbent_score=0.6, best_score=0.9)
    assert result.action == "accept"


def test_skill_gate_reject():
    result = skill_gate(0.5, incumbent_score=0.6, best_score=0.9)
    assert result.action == "reject"


def test_skill_gate_best_score_defaults_to_incumbent():
    # A win over the incumbent with no separate best-score tracker is accept_new_best.
    result = skill_gate(0.8, incumbent_score=0.6)
    assert result.action == "accept_new_best"
    assert result.best_score == 0.6
