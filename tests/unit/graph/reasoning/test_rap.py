"""Unit tests for the RAP topology -- select / expand / simulate / backpropagate.

The critical test here is :func:`test_backpropagation_updates_every_ancestor`:
it proves a reward discovered at a DEPTH-3 leaf is credited to EVERY ancestor
up to the root, not just the immediate parent -- the documented failure mode
of a "simplified" MCTS/RAP port.
"""

from agent_utilities.graph.reasoning.budgets import Budgets, TerminationReason
from agent_utilities.graph.reasoning.rap import (
    _backpropagate,
    _select,
    run_rap,
)
from agent_utilities.graph.reasoning.state import NodeKind, ReasoningState, ThoughtNode


def _build_chain(depth: int) -> ReasoningState:
    """A single unbranched chain root -> n1 -> n2 -> ... -> n{depth}."""
    state = ReasoningState(topology="rap")
    state.add_node(ThoughtNode("root", NodeKind.ROOT, "q", depth=0))
    parent = "root"
    for i in range(1, depth + 1):
        node_id = f"n{i}"
        state.add_node(
            ThoughtNode(
                node_id, NodeKind.GENERATE, f"step {i}", parent_ids=[parent], depth=i
            )
        )
        parent = node_id
    return state


def test_backpropagation_updates_every_ancestor():
    """A reward simulated at a depth-3 leaf must reach the root, not just its parent."""
    state = _build_chain(depth=3)

    _backpropagate(state, "n3", reward=1.0)

    for node_id in ("root", "n1", "n2", "n3"):
        node = state.nodes[node_id]
        assert node.visits == 1, f"{node_id} was not credited"
        assert node.total_reward == 1.0, f"{node_id} reward was not propagated"


def test_backpropagation_accumulates_across_multiple_simulations():
    state = _build_chain(depth=2)

    _backpropagate(state, "n2", reward=1.0)
    _backpropagate(state, "n2", reward=0.0)
    _backpropagate(state, "n1", reward=1.0)  # a second, shallower leaf

    # root is on the path of ALL THREE backprops -> visited 3x, reward 2.0
    assert state.nodes["root"].visits == 3
    assert state.nodes["root"].total_reward == 2.0
    # n1 is on the path of the two n2-backprops PLUS its own direct one -> 3x
    assert state.nodes["n1"].visits == 3
    assert state.nodes["n1"].total_reward == 2.0
    # n2 only credited by its own two direct simulations
    assert state.nodes["n2"].visits == 2
    assert state.nodes["n2"].total_reward == 1.0


def test_select_prefers_the_unvisited_child_first():
    state = _build_chain(depth=1)
    state.add_node(
        ThoughtNode("n1b", NodeKind.GENERATE, "alt", parent_ids=["root"], depth=1)
    )
    # n1 already has stats; n1b is unvisited (uct == inf) and must win selection.
    state.nodes["n1"].visits = 5
    state.nodes["n1"].total_reward = 4.0

    selected = _select(state, "root", c=1.4)
    assert selected == "n1b"


def test_run_rap_expands_a_multi_level_tree_and_reaches_a_goal():
    # expand_fn deterministically branches on the current path length.
    def expand(content: str) -> list[str]:
        depth = content.count(".")
        if depth >= 2:
            return []  # depth cap -- no further children
        return [f"{content}.{i}" for i in range(2)]

    def simulate(content: str) -> float:
        return 1.0 if content.endswith(".1.1") else 0.1

    def is_goal(content: str) -> bool:
        return content == "root.1.1"

    state, proof = run_rap(
        expand,
        simulate,
        is_goal,
        question="root",
        budgets=Budgets(loop_budget=200),
    )

    assert proof.reason is TerminationReason.GOAL_REACHED
    assert proof.success is True
    assert proof.degraded is False
    # the root must show accumulated visits from the many simulations below it.
    root = next(n for n in state.nodes.values() if n.kind is NodeKind.ROOT)
    assert root.visits > 0


def test_run_rap_halts_truthfully_on_loop_budget_exhaustion():
    def expand(content: str) -> list[str]:
        return [
            f"{content}.x"
        ]  # infinite chain -- never reaches the (unreachable) goal

    def simulate(_content: str) -> float:
        return 0.5

    def is_goal(_content: str) -> bool:
        return False

    state, proof = run_rap(
        expand, simulate, is_goal, question="root", budgets=Budgets(loop_budget=5)
    )

    assert proof.reason is TerminationReason.LOOP_BUDGET_EXHAUSTED
    assert proof.degraded is True
    assert proof.as_report()["success"] is False
    # partial search progress (visited/expanded nodes) is preserved, not discarded.
    assert len(state.nodes) > 1
    assert "best=" in proof.detail
