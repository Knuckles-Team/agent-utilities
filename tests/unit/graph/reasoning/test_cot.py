"""Unit tests for the CoT and self-consistent CoT topologies."""

from agent_utilities.graph.reasoning.budgets import Budgets, TerminationReason
from agent_utilities.graph.reasoning.cot import run_cot, run_self_consistent_cot
from agent_utilities.graph.reasoning.state import NodeKind


def test_cot_is_a_linear_chain():
    """Each generated node has exactly one parent -- the previous step."""
    steps = iter(
        [
            ("summarized step 1", "intermediate thought", False),
            ("summarized step 2", "final answer: 42", True),
        ]
    )

    def step_fn(_state):
        return next(steps)

    state, proof = run_cot(step_fn, question="what is the answer?")

    assert proof.reason is TerminationReason.GOAL_REACHED
    assert proof.success is True
    assert proof.degraded is False

    generated = [n for n in state.nodes.values() if n.kind is NodeKind.GENERATE]
    assert len(generated) == 2
    for node in generated:
        assert len(node.parent_ids) == 1
    assert generated[-1].content == "final answer: 42"
    assert generated[-1].is_terminal is True


def test_cot_rationale_is_summary_only_never_raw_reasoning():
    def step_fn(_state):
        return "the model's private reasoning was summarized here", "answer", True

    state, _proof = run_cot(step_fn, question="q")
    assert state.rationale_summary == [
        "the model's private reasoning was summarized here"
    ]


def test_cot_halts_truthfully_on_loop_budget_exhaustion():
    def step_fn(_state):
        return "still thinking", "not done yet", False  # never finalizes

    state, proof = run_cot(step_fn, question="q", budgets=Budgets(loop_budget=3))

    assert proof.reason is TerminationReason.LOOP_BUDGET_EXHAUSTED
    assert proof.degraded is True
    assert proof.as_report()["success"] is False
    # partial progress is preserved, not discarded
    generated = [n for n in state.nodes.values() if n.kind is NodeKind.GENERATE]
    assert len(generated) == 3


def test_self_consistent_cot_majority_vote():
    """Three chains agree on 'answer B', two on 'answer A' -> B wins."""
    answers = iter(["answer a", "answer b", "answer b", "answer a", "answer b"])

    def step_fn(_state):
        return "summary", next(answers), True

    state, proof, winner = run_self_consistent_cot(step_fn, question="q", num_samples=5)

    assert winner == "answer b"
    assert proof.success is True
    assert proof.degraded is False
    assert state.candidates[" ".join("answer b".split())].votes == 3
    assert state.candidates[" ".join("answer a".split())].votes == 2


def test_self_consistent_cot_confidence_reflects_vote_share():
    answers = iter(["x", "x", "x", "y"])

    def step_fn(_state):
        return "s", next(answers), True

    state, _proof, winner = run_self_consistent_cot(
        step_fn, question="q", num_samples=4
    )
    assert winner == "x"
    assert state.candidates["x"].confidence == 0.75
    assert state.candidates["y"].confidence == 0.25


def test_self_consistent_cot_degrades_truthfully_on_shared_budget_exhaustion():
    call_count = {"n": 0}

    def step_fn(_state):
        call_count["n"] += 1
        return "s", "partial", False  # never finalizes -> burns the whole budget

    state, proof, winner = run_self_consistent_cot(
        step_fn, question="q", num_samples=5, budgets=Budgets(loop_budget=2)
    )

    assert winner == ""
    assert proof.degraded is True
    assert proof.as_report()["success"] is False
