"""Unit tests for the ReAct thought -> action -> observation topology."""

from agent_utilities.graph.reasoning.budgets import Budgets, TerminationReason
from agent_utilities.graph.reasoning.react import run_react
from agent_utilities.graph.reasoning.state import NodeKind


def test_react_stops_on_is_done_before_another_thought():
    think_calls = {"n": 0}

    def think(_state):
        think_calls["n"] += 1
        return "thinking", "search", {"q": "answer"}

    def act(_action, _args):
        return "found it", True

    calls = {"n": 0}

    def is_done(_state):
        calls["n"] += 1
        return calls["n"] > 1  # done on the SECOND check (after one act/observe)

    state, proof = run_react(
        think,
        act,
        is_done,
        question="find the answer",
        budgets=Budgets(loop_budget=10, tool_budget=10),
    )

    assert proof.reason is TerminationReason.GOAL_REACHED
    assert proof.success is True
    assert proof.degraded is False
    acts = [n for n in state.nodes.values() if n.kind is NodeKind.ACT]
    observes = [n for n in state.nodes.values() if n.kind is NodeKind.OBSERVE]
    assert len(acts) == 1
    assert len(observes) == 1
    assert observes[0].parent_ids == [acts[0].node_id]
    assert state.tool_calls[0].action == "search"
    assert state.tool_calls[0].success is True


def test_react_records_tool_result_and_retry_on_failed_action():
    def think(_state):
        return "retry needed", "flaky_tool", {}

    def act(_action, _args):
        return "error: timeout", False

    def is_done(_state):
        return False

    state, proof = run_react(
        think,
        act,
        is_done,
        question="q",
        budgets=Budgets(loop_budget=10, tool_budget=10),
        max_retries_per_action=1,
    )

    assert proof.reason is TerminationReason.GROUNDING_FAILURE
    assert proof.degraded is True
    assert proof.as_report()["success"] is False
    assert all(not c.success for c in state.tool_calls)
    assert state.tool_calls[-1].retry >= 1


def test_react_halts_truthfully_on_tool_budget_exhaustion():
    def think(_state):
        return "acting", "tool", {"n": 1}

    def act(_action, _args):
        return "ok", True  # never done, always succeeds -- burns the tool budget

    def is_done(_state):
        return False

    state, proof = run_react(
        think,
        act,
        is_done,
        question="q",
        budgets=Budgets(loop_budget=100, tool_budget=2),
    )

    assert proof.reason is TerminationReason.TOOL_BUDGET_EXHAUSTED
    assert proof.degraded is True
    assert len(state.tool_calls) == 2  # partial progress preserved


def test_react_rationale_summaries_are_recorded_each_step():
    def think(_state):
        return "step rationale", "act", {}

    def act(_action, _args):
        return "obs", True

    def is_done(state):
        return len(state.tool_calls) >= 2

    state, _proof = run_react(
        think,
        act,
        is_done,
        question="q",
        budgets=Budgets(loop_budget=10, tool_budget=10),
    )
    assert state.rationale_summary == ["step rationale", "step rationale"]
