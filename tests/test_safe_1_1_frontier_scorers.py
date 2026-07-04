"""Non-saturating superhuman progress signals (CONCEPT:AU-OS.scaling.non-saturating-compression-scorer).

Relative scorers + a saturation detector that keep producing signal past the
human/known-answer ceiling, so a genuine capability jump is distinguishable from
metric saturation. CompressionScorer rides the reliability suite as an
informational (never-gating) non-saturating metric.
"""

from __future__ import annotations

import pytest

from agent_utilities.harness.frontier_scorers import (
    CompressionScorer,
    build_frontier_suite,
    elo_from_duels,
    saturation_detector,
    setter_solver_gap,
)
from agent_utilities.harness.reliability_scorers import build_reliability_suite

pytestmark = pytest.mark.concept("AU-OS.scaling.non-saturating-compression-scorer")


class TestCompressionScorer:
    def test_rewards_density_and_never_gates(self):
        r = CompressionScorer().score("a" * 100, "b" * 10)
        assert r.passed is True  # informational, never a guardrail
        assert r.metrics["compression_ratio"] == pytest.approx(0.9)
        assert r.score == pytest.approx(0.9)

    def test_empty_output_scores_zero(self):
        assert CompressionScorer().score("input", "").score == 0.0

    def test_registered_in_reliability_suite_live_path(self):
        # Wire-First: the scorer runs wherever build_reliability_suite is used.
        suite = build_reliability_suite()
        assert "compression" in {s.name for s in suite.scorers}
        # and it never flips the suite's pass/fail (always passes).
        result = suite.evaluate("the quick brown fox", "fox", {})
        comp = [r for r in result.results if r.evaluator == "compression"][0]
        assert comp.passed is True


class TestRelativeRanking:
    def test_elo_orders_by_wins(self):
        duels = [("a", "b"), ("a", "c"), ("b", "c")]  # a > b > c
        ratings = elo_from_duels(duels)
        assert ratings["a"] > ratings["b"] > ratings["c"]

    def test_no_duels_empty(self):
        assert elo_from_duels([]) == {}

    def test_setter_solver_gap_rises_with_difficulty(self):
        # solver fails more of the setter's problems ⇒ higher frontier difficulty.
        assert setter_solver_gap(2, 10) == pytest.approx(0.8)
        assert setter_solver_gap(9, 10) == pytest.approx(0.1)
        assert setter_solver_gap(0, 0) == 0.0


class TestSaturationDetector:
    def test_flags_ceiling_collapse(self):
        d = saturation_detector([0.99, 0.985, 0.99], ceiling=0.98, window=3)
        assert d["saturated"] is True and "promote" in d["reason"]

    def test_still_discriminating(self):
        d = saturation_detector([0.7, 0.8, 0.85], ceiling=0.98, window=3)
        assert d["saturated"] is False

    def test_insufficient_history(self):
        d = saturation_detector([0.99], window=3)
        assert d["saturated"] is False and d["reason"] == "insufficient history"


def test_frontier_suite_builds():
    suite = build_frontier_suite()
    assert "compression" in {s.name for s in suite.scorers}
