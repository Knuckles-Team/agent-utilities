"""Unit tests for loop/tool/token/cost/time budgets + classified termination."""

import pytest

from agent_utilities.graph.reasoning.budgets import (
    BudgetExhausted,
    Budgets,
    BudgetTracker,
    TerminationProof,
    TerminationReason,
)


def test_loop_budget_trips_after_exact_count():
    tracker = BudgetTracker(Budgets(loop_budget=3))
    tracker.tick_loop()
    tracker.tick_loop()
    tracker.tick_loop()
    with pytest.raises(BudgetExhausted) as excinfo:
        tracker.tick_loop()
    assert excinfo.value.proof.reason is TerminationReason.LOOP_BUDGET_EXHAUSTED
    assert excinfo.value.proof.degraded is True
    assert excinfo.value.proof.success is False


def test_tool_budget_trips_independently_of_loop_budget():
    tracker = BudgetTracker(Budgets(loop_budget=100, tool_budget=1))
    tracker.tick_tool_call()
    with pytest.raises(BudgetExhausted) as excinfo:
        tracker.tick_tool_call()
    assert excinfo.value.proof.reason is TerminationReason.TOOL_BUDGET_EXHAUSTED


def test_zero_tool_budget_means_no_tools_allowed():
    """``tool_budget=0`` is a REAL ceiling of zero, not "unlimited".

    Regression (waves 1-5 gate): the guard used to be
    ``if self.budgets.tool_budget and ...``, so a falsy 0 short-circuited the
    whole check and a topology declaring "no tools allowed" could make
    unlimited tool calls without ever tripping.
    """
    tracker = BudgetTracker(Budgets(loop_budget=10, tool_budget=0))
    with pytest.raises(BudgetExhausted) as excinfo:
        tracker.tick_tool_call()
    assert excinfo.value.proof.reason is TerminationReason.TOOL_BUDGET_EXHAUSTED
    assert excinfo.value.proof.degraded is True


def test_none_tool_budget_leaves_the_tool_axis_unbounded():
    tracker = BudgetTracker(Budgets(loop_budget=10, tool_budget=None))
    for _ in range(5):
        tracker.tick_tool_call()
    assert tracker.tool_calls_used == 5
    assert Budgets(loop_budget=1, tool_budget=None).declared_axes()["tool"] is False


def test_token_budget_trips_on_caller_reported_usage():
    """``token_budget`` is enforceable — via the caller-reported usage seam.

    Regression (waves 1-5 gate): ``BudgetTracker`` built a private
    ``BudgetGuard`` over a private ``TokenUsageTracker`` that nothing in this
    package ever wrote to, so ``token_budget``/``cost_budget_usd`` were
    structurally inert — ``TOKEN_BUDGET_EXHAUSTED`` was advertised as a
    reachable termination reason but could never fire. ``record_usage`` is the
    seam that makes it real; this proves it trips.
    """
    tracker = BudgetTracker(Budgets(loop_budget=1_000_000, token_budget=100))
    tracker.record_usage(prompt_tokens=40, response_tokens=40)
    with pytest.raises(BudgetExhausted) as excinfo:
        tracker.record_usage(prompt_tokens=40)
    assert excinfo.value.proof.reason is TerminationReason.TOKEN_BUDGET_EXHAUSTED
    assert excinfo.value.proof.degraded is True


def test_cost_budget_trips_on_caller_reported_usage():
    tracker = BudgetTracker(Budgets(loop_budget=1_000_000, cost_budget_usd=0.0001))
    with pytest.raises(BudgetExhausted) as excinfo:
        tracker.record_usage(prompt_tokens=10_000, response_tokens=10_000)
    assert excinfo.value.proof.reason is TerminationReason.COST_BUDGET_EXHAUSTED


def test_declared_axes_reports_which_ceilings_are_actually_declared():
    axes = Budgets(loop_budget=4, token_budget=10).declared_axes()
    assert axes == {
        "loop": True,
        "tool": True,
        "token": True,
        "cost": False,
        "time": False,
    }


def test_time_budget_trips_via_shared_budget_guard():
    tracker = BudgetTracker(Budgets(loop_budget=1_000_000, time_budget_s=0.0))
    with pytest.raises(BudgetExhausted) as excinfo:
        tracker.tick_loop()
    assert excinfo.value.proof.reason is TerminationReason.TIME_BUDGET_EXHAUSTED


def test_termination_proof_as_report_never_reports_degraded_as_success():
    proof = TerminationProof(
        reason=TerminationReason.LOOP_BUDGET_EXHAUSTED,
        success=True,  # the topology THOUGHT it had an answer
        degraded=True,  # but it was budget-halted
        detail="halted",
    )
    report = proof.as_report()
    assert report["success"] is False
    assert report["degraded"] is True


def test_termination_proof_as_report_clean_success():
    proof = TerminationProof(
        reason=TerminationReason.GOAL_REACHED, success=True, degraded=False
    )
    assert proof.as_report()["success"] is True
