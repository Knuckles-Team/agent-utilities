"""Unit tests for the Graph-of-Thought topology (generate/transform/aggregate/refine)."""

from agent_utilities.graph.reasoning.budgets import Budgets, TerminationReason
from agent_utilities.graph.reasoning.got import GotOp, run_got
from agent_utilities.graph.reasoning.state import NodeKind


def test_got_generate_then_aggregate_records_merge_provenance():
    plan = [
        GotOp(
            op_id="g1",
            kind=NodeKind.GENERATE,
            parent_ids=["root"],
            fn=lambda q: f"idea-a from {q}",
        ),
        GotOp(
            op_id="g2",
            kind=NodeKind.GENERATE,
            parent_ids=["root"],
            fn=lambda q: f"idea-b from {q}",
        ),
        GotOp(
            op_id="merged",
            kind=NodeKind.AGGREGATE,
            parent_ids=["g1", "g2"],
            fn=lambda contents: " + ".join(contents),
        ),
    ]

    state, proof = run_got("seed question", plan)

    assert proof.reason is TerminationReason.GOAL_REACHED
    assert proof.success is True
    merged = state.nodes["merged"]
    assert merged.kind is NodeKind.AGGREGATE
    # the merge provenance IS the >1 parent_ids -- no separate structure needed.
    assert merged.parent_ids == ["g1", "g2"]
    assert merged.metadata["merged_from"] == ["g1", "g2"]
    assert merged.content == "idea-a from seed question + idea-b from seed question"
    assert merged.is_terminal is True


def test_got_transform_has_single_parent():
    plan = [
        GotOp(
            op_id="t1",
            kind=NodeKind.TRANSFORM,
            parent_ids=["root"],
            fn=lambda q: q.upper(),
        ),
    ]
    state, _proof = run_got("hello", plan)
    node = state.nodes["t1"]
    assert node.parent_ids == ["root"]
    assert node.content == "HELLO"


def test_got_refine_keeps_the_best_scoring_candidate_over_rounds():
    # Round 1 beats the seed, round 2 beats round 1, round 3 REGRESSES --
    # REFINE must keep round 2's candidate, not blindly take the last round.
    content_by_round = {0: "candidate-0", 1: "candidate-1", 2: "candidate-2"}
    score_by_content = {
        "base": 0.0,
        "candidate-0": 0.2,
        "candidate-1": 0.9,
        "candidate-2": 0.1,
    }
    counter = {"i": 0}

    def refine_step(_current: str) -> str:
        content = content_by_round[counter["i"]]
        counter["i"] += 1
        return content

    def score(candidate: str) -> float:
        return score_by_content[candidate]

    plan = [
        GotOp(
            op_id="r1",
            kind=NodeKind.REFINE,
            parent_ids=["root"],
            fn=refine_step,
            score_fn=score,
            refine_rounds=3,
        )
    ]
    state, _proof = run_got("base", plan)
    node = state.nodes["r1"]
    assert node.content == "candidate-1"
    assert node.value == 0.9


def test_got_halts_truthfully_on_loop_budget_exhaustion():
    plan = [
        GotOp(
            op_id=f"g{i}", kind=NodeKind.GENERATE, parent_ids=["root"], fn=lambda q: q
        )
        for i in range(10)
    ]
    state, proof = run_got("q", plan, budgets=Budgets(loop_budget=3))
    assert proof.reason is TerminationReason.LOOP_BUDGET_EXHAUSTED
    assert proof.degraded is True
    assert proof.as_report()["success"] is False
    completed = [n for n in state.nodes.values() if n.node_id.startswith("g")]
    assert len(completed) == 3  # partial progress preserved, not discarded
