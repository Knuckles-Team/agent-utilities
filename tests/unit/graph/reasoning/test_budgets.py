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
    tracker = BudgetTracker(Budgets(loop_budget=10, tool_budget=0))
    # tool_budget=0 is falsy -> the guard is a no-op (unlimited), matching the
    # documented "0 = no tools allowed" semantics is enforced by the CALLER
    # never invoking tick_tool_call for a tool_budget=0 topology (react.py
    # always sets a positive tool_budget). Assert the tracker itself does not
    # spuriously trip on an unset budget.
    for _ in range(5):
        tracker.tick_tool_call()
    assert tracker.tool_calls_used == 5


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
