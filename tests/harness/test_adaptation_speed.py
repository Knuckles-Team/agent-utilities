#!/usr/bin/python
"""Tests for the adaptation-speed metric + SAI task/verifier contract.

CONCEPT:AHE-3.27 / AHE-3.28
"""

from __future__ import annotations

import pytest

from agent_utilities.harness import (
    AdaptationCurve,
    SpecializationTask,
    Verifier,
    VerifierResult,
    marginal_speed_gain,
)


# --------------------------------------------------------------------------- #
# AdaptationCurve (AHE-3.27)
# --------------------------------------------------------------------------- #


def _curve(points: list[tuple[float, int, float]]) -> AdaptationCurve:
    curve = AdaptationCurve(task_id="t")
    for t_wall, n_samples, reward in points:
        curve.record(t_wall, n_samples, reward)
    return curve


def test_time_to_target_and_sample_complexity_on_rising_curve():
    # reward crosses tau=0.8 at the 3rd point (t=3s, 30 samples)
    curve = _curve([(1.0, 10, 0.4), (2.0, 20, 0.6), (3.0, 30, 0.85), (4.0, 40, 0.9)])
    assert curve.reached(0.8) is True
    assert curve.time_to_target(0.8) == 3.0
    assert curve.sample_complexity(0.8) == 30


def test_unreached_target_returns_none():
    curve = _curve([(1.0, 10, 0.1), (2.0, 20, 0.2)])
    assert curve.reached(0.8) is False
    assert curve.time_to_target(0.8) is None
    assert curve.sample_complexity(0.8) is None


def test_best_so_far_makes_noisy_regression_not_unmeet_target():
    # Point 2 reaches 0.85, point 3 regresses to 0.5 — target stays met at point 2.
    curve = _curve([(1.0, 10, 0.5), (2.0, 20, 0.85), (3.0, 30, 0.5)])
    assert curve.time_to_target(0.8) == 2.0
    assert curve.sample_complexity(0.8) == 20


def test_faster_riser_has_higher_learning_auc():
    fast = _curve([(1.0, 10, 0.9), (2.0, 20, 0.9), (3.0, 30, 0.9)])
    slow = _curve([(1.0, 10, 0.1), (2.0, 20, 0.1), (3.0, 30, 0.9)])
    assert fast.learning_auc() > slow.learning_auc()


def test_peak_and_final_reward():
    curve = _curve([(1.0, 10, 0.4), (2.0, 20, 0.95), (3.0, 30, 0.7)])
    assert curve.peak_reward() == pytest.approx(0.95)
    assert curve.final_reward() == pytest.approx(0.7)


def test_empty_curve_is_safe():
    curve = AdaptationCurve(task_id="empty")
    assert curve.learning_auc() == 0.0
    assert curve.peak_reward() == 0.0
    assert curve.final_reward() == 0.0
    assert curve.time_to_target(0.5) is None


def test_single_point_learning_auc_is_the_point_reward():
    curve = _curve([(1.0, 10, 0.42)])
    assert curve.learning_auc() == pytest.approx(0.42)


def test_out_of_order_append_rejected():
    curve = _curve([(2.0, 20, 0.5)])
    with pytest.raises(ValueError):
        curve.record(1.0, 30, 0.6)  # t_wall went backwards
    with pytest.raises(ValueError):
        curve.record(3.0, 10, 0.6)  # n_samples went backwards


def test_metrics_summary_is_serializable():
    curve = _curve([(1.0, 10, 0.4), (3.0, 30, 0.85)])
    m = curve.metrics(0.8)
    assert m["reached"] is True
    assert m["time_to_target_s"] == 3.0
    assert m["sample_complexity"] == 30
    assert m["iterations"] == 2
    # round-trippable through json-ish primitives
    assert set(m) >= {"time_to_target_s", "sample_complexity", "learning_auc", "peak_reward"}


def test_marginal_speed_gain_prefers_the_faster_arm():
    before = _curve([(1.0, 10, 0.2), (2.0, 20, 0.3)])
    after = _curve([(1.0, 10, 0.7), (2.0, 20, 0.8)])
    assert marginal_speed_gain(before, after, tau=0.8) > 0


# --------------------------------------------------------------------------- #
# SpecializationTask + Verifier (AHE-3.28)
# --------------------------------------------------------------------------- #


class _ThresholdVerifier:
    """Toy verifier: reward = fraction of target substring present; passes at 1.0."""

    def __init__(self, target: str) -> None:
        self.target = target

    def verify(self, candidate: str) -> VerifierResult:
        hit = self.target in (candidate or "")
        reward = 1.0 if hit else (0.5 if self.target[: len(self.target) // 2] in candidate else 0.0)
        return VerifierResult(reward=reward, passed=hit, detail={"target": self.target})


def test_verifier_protocol_is_satisfied_structurally():
    v = _ThresholdVerifier("kernel")
    assert isinstance(v, Verifier)


def test_specialization_task_scores_through_verifier():
    task = SpecializationTask(
        task_id="kernel-fused-softmax",
        prompt_corpus=["write a fused softmax kernel"],
        verifier=_ThresholdVerifier("kernel"),
        target_tau=1.0,
        human_baseline=0.7,
    )
    good = task.score("here is a kernel")
    assert good.passed is True
    assert good.reward == pytest.approx(1.0)
    bad = task.score("here is nothing useful")
    assert bad.passed is False
    assert bad.reward == pytest.approx(0.0)


def test_task_rejects_empty_id_and_non_verifier():
    with pytest.raises(ValueError):
        SpecializationTask(
            task_id="",
            prompt_corpus=[],
            verifier=_ThresholdVerifier("x"),
            target_tau=1.0,
        )
    with pytest.raises(TypeError):
        SpecializationTask(
            task_id="t",
            prompt_corpus=[],
            verifier=object(),  # type: ignore[arg-type]
            target_tau=1.0,
        )


def test_curve_consumes_verifier_rewards_end_to_end():
    """Live-path-ish: a verifier's rewards drive an adaptation curve to target."""
    task = SpecializationTask(
        task_id="kernel",
        prompt_corpus=["op"],
        verifier=_ThresholdVerifier("kernel"),
        target_tau=1.0,
    )
    curve = AdaptationCurve(task_id=task.task_id)
    candidates = ["nothing", "a ker piece", "a full kernel now"]  # 0.0, 0.5, 1.0
    for i, cand in enumerate(candidates):
        res = task.score(cand)
        curve.record(t_wall=float(i + 1), n_samples=(i + 1) * 10, reward=res.reward)
    assert curve.reached(task.target_tau)
    assert curve.time_to_target(task.target_tau) == 3.0
    assert curve.sample_complexity(task.target_tau) == 30
