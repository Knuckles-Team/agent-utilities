"""Tests for the pre-run baseline/overfit go/no-go gates (CONCEPT:AU-AHE.assimilation.baseline-overfit-gate)."""

from __future__ import annotations

import math

from agent_utilities.harness.baseline_overfit_gate import (
    GateVerdict,
    PreRunGate,
    baseline_gate,
    overfit_smoke_gate,
)


# --- baseline_gate ----------------------------------------------------------


def test_baseline_gate_passes_on_sufficient_lift() -> None:
    v = baseline_gate(0.90, 0.85, min_lift=0.02)
    assert isinstance(v, GateVerdict)
    assert v.passed is True
    assert v.detail["lift"] == 0.90 - 0.85
    assert "beats baseline" in v.reason


def test_baseline_gate_fails_on_insufficient_lift() -> None:
    v = baseline_gate(0.86, 0.85, min_lift=0.02)
    assert v.passed is False
    assert "insufficient lift" in v.reason


def test_baseline_gate_fails_on_negative_lift() -> None:
    # Candidate is actually worse than the baseline.
    v = baseline_gate(0.80, 0.85, min_lift=0.0)
    assert v.passed is False
    assert v.detail["lift"] < 0


def test_baseline_gate_lower_is_better() -> None:
    # Loss-style metric: candidate 0.30 vs baseline 0.50 is a 0.20 improvement.
    win = baseline_gate(0.30, 0.50, min_lift=0.05, higher_is_better=False)
    assert win.passed is True
    assert win.detail["lift"] == 0.20

    # A higher loss is a regression and must fail when lower is better.
    loss = baseline_gate(0.60, 0.50, min_lift=0.0, higher_is_better=False)
    assert loss.passed is False


def test_baseline_gate_min_lift_boundary_is_inclusive() -> None:
    # Exactly meeting min_lift passes (>=).
    v = baseline_gate(0.87, 0.85, min_lift=0.02)
    assert v.passed is True
    # Just under fails.
    v2 = baseline_gate(0.86999, 0.85, min_lift=0.02)
    assert v2.passed is False


def test_baseline_gate_rejects_negative_min_lift() -> None:
    v = baseline_gate(0.80, 0.85, min_lift=-0.1)
    assert v.passed is False
    assert "non-negative" in v.reason


def test_baseline_gate_rejects_non_finite() -> None:
    v = baseline_gate(math.nan, 0.85)
    assert v.passed is False
    assert "non-finite" in v.reason


# --- overfit_smoke_gate -----------------------------------------------------


def test_overfit_gate_passes_on_decreasing_curve() -> None:
    v = overfit_smoke_gate([1.0, 0.6, 0.3, 0.1], min_drop_frac=0.5)
    assert v.passed is True
    assert v.detail["initial_loss"] == 1.0
    assert v.detail["final_loss"] == 0.1
    assert math.isclose(v.detail["drop_frac"], 0.9)


def test_overfit_gate_drop_frac_boundary_inclusive() -> None:
    # Drops exactly 50%.
    v = overfit_smoke_gate([1.0, 0.5], min_drop_frac=0.5)
    assert v.passed is True
    # Drops just under 50%.
    v2 = overfit_smoke_gate([1.0, 0.51], min_drop_frac=0.5)
    assert v2.passed is False


def test_overfit_gate_fails_on_flat_curve() -> None:
    v = overfit_smoke_gate([1.0, 1.0, 1.0], min_drop_frac=0.5)
    assert v.passed is False
    assert "cannot overfit" in v.reason


def test_overfit_gate_fails_on_rising_curve() -> None:
    v = overfit_smoke_gate([1.0, 1.5, 2.0], min_drop_frac=0.5)
    assert v.passed is False
    assert v.detail["drop_frac"] <= 0.0


def test_overfit_gate_fails_on_empty_curve() -> None:
    v = overfit_smoke_gate([], min_drop_frac=0.5)
    assert v.passed is False
    assert "at least 2 points" in v.reason


def test_overfit_gate_fails_on_single_point() -> None:
    v = overfit_smoke_gate([1.0], min_drop_frac=0.5)
    assert v.passed is False


def test_overfit_gate_fails_on_non_positive_initial() -> None:
    v = overfit_smoke_gate([0.0, -0.5], min_drop_frac=0.5)
    assert v.passed is False
    assert "positive" in v.reason


def test_overfit_gate_fails_on_nan() -> None:
    v = overfit_smoke_gate([1.0, math.nan], min_drop_frac=0.5)
    assert v.passed is False
    assert "non-finite" in v.reason


def test_overfit_gate_rejects_bad_min_drop_frac() -> None:
    assert overfit_smoke_gate([1.0, 0.1], min_drop_frac=1.5).passed is False
    assert overfit_smoke_gate([1.0, 0.1], min_drop_frac=-0.1).passed is False


# --- PreRunGate -------------------------------------------------------------


def test_prerun_gate_requires_both() -> None:
    gate = PreRunGate(min_lift=0.02, min_drop_frac=0.5)

    go = gate.evaluate(
        candidate_score=0.90,
        baseline_score=0.85,
        single_batch_loss_curve=[1.0, 0.4, 0.05],
    )
    assert go.passed is True
    assert go.detail["baseline"]["passed"] is True
    assert go.detail["overfit"]["passed"] is True


def test_prerun_gate_fails_when_baseline_fails() -> None:
    gate = PreRunGate()
    v = gate.evaluate(
        candidate_score=0.851,
        baseline_score=0.85,
        single_batch_loss_curve=[1.0, 0.1],
    )
    assert v.passed is False
    assert v.detail["baseline"]["passed"] is False
    assert v.detail["overfit"]["passed"] is True
    assert "baseline" in v.reason


def test_prerun_gate_fails_when_overfit_fails() -> None:
    gate = PreRunGate()
    v = gate.evaluate(
        candidate_score=0.95,
        baseline_score=0.85,
        single_batch_loss_curve=[1.0, 1.0],
    )
    assert v.passed is False
    assert v.detail["baseline"]["passed"] is True
    assert v.detail["overfit"]["passed"] is False
    assert "overfit" in v.reason


def test_prerun_gate_fails_when_both_fail() -> None:
    gate = PreRunGate()
    v = gate.evaluate(
        candidate_score=0.80,
        baseline_score=0.85,
        single_batch_loss_curve=[1.0, 1.0],
    )
    assert v.passed is False
    assert "and" in v.reason


def test_prerun_gate_honors_higher_is_better_false() -> None:
    gate = PreRunGate(min_lift=0.05)
    v = gate.evaluate(
        candidate_score=0.30,
        baseline_score=0.50,
        single_batch_loss_curve=[1.0, 0.2],
        higher_is_better=False,
    )
    assert v.passed is True


# --- verdict shape & determinism -------------------------------------------


def test_verdict_detail_shape() -> None:
    v = PreRunGate().evaluate(
        candidate_score=0.90,
        baseline_score=0.85,
        single_batch_loss_curve=[1.0, 0.1],
    )
    assert set(v.detail) == {"baseline", "overfit"}
    for sub in v.detail.values():
        assert set(sub) == {"passed", "reason", "detail"}
        assert isinstance(sub["passed"], bool)
        assert isinstance(sub["reason"], str)
        assert isinstance(sub["detail"], dict)


def test_determinism() -> None:
    args = dict(
        candidate_score=0.88,
        baseline_score=0.85,
        single_batch_loss_curve=[1.0, 0.7, 0.3],
    )
    gate = PreRunGate()
    first = gate.evaluate(**args)
    second = gate.evaluate(**args)
    assert first == second
    # Frozen dataclass equality includes nested detail dicts.
    assert first.detail == second.detail
