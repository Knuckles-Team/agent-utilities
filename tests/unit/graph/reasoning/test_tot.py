"""Unit tests for the Tree-of-Thought (BFS/DFS) topology."""

from agent_utilities.graph.reasoning.budgets import Budgets, TerminationReason
from agent_utilities.graph.reasoning.state import NodeKind
from agent_utilities.graph.reasoning.tot import PruningPolicy, run_tot

# A tiny deterministic search space: each thought is a path string like
# "0", "0.1", "0.1.9" ... reachable digits are 0/1/9 (never 2), and the ONLY
# goal is the specific depth-3 thought "0.1.9" -- deep enough that BFS must
# actually explore multiple levels to find it.
_SCORES = {
    "0.0": 0.2,
    "0.1": 0.9,
    "0.9": 0.3,
    "0.1.0": 0.3,
    "0.1.1": 0.2,
    "0.1.9": 1.0,
}


def _generate(content: str) -> list[str]:
    return [f"{content}.{d}" for d in (0, 1, 9)]


def _score(content: str) -> float:
    return _SCORES.get(content, 0.1)


def _is_goal(content: str) -> bool:
    return content == "0.1.9"


def test_tot_bfs_reaches_a_deep_goal():
    state, proof = run_tot(
        _generate,
        _score,
        _is_goal,
        question="0",
        strategy="bfs",
        branching_factor=3,
        beam_width=3,
        max_depth=3,
    )
    assert proof.reason is TerminationReason.GOAL_REACHED
    assert proof.success is True
    assert proof.degraded is False
    assert proof.detail in state.nodes
    assert state.nodes[proof.detail].content == "0.1.9"


def test_tot_frontier_and_visited_are_tracked_on_shared_state():
    state, _proof = run_tot(
        _generate, _score, _is_goal, question="0", strategy="bfs", max_depth=2
    )
    assert state.visited  # some nodes were popped and marked visited
    generated = [n for n in state.nodes.values() if n.kind is NodeKind.GENERATE]
    assert generated
    assert all(n.value is not None for n in generated)


def test_tot_top_k_pruning_keeps_only_beam_width_survivors():
    state, _proof = run_tot(
        _generate,
        _score,
        lambda _c: False,  # never goal -- force full exhaustion at max_depth=1
        question="0",
        strategy="bfs",
        branching_factor=3,
        beam_width=1,
        max_depth=1,
        pruning=PruningPolicy.TOP_K,
        budgets=Budgets(loop_budget=10),
    )
    # depth-1 has 3 candidates scored 0.2/0.9/0.3 -- all three are GENERATED,
    # but beam_width=1 only ever pushes the single highest scorer ("0.1",
    # value 0.9) onto the frontier, so only it (plus the root) is ever
    # popped/visited -- the other two are pruned dead ends.
    depth1 = [n for n in state.nodes.values() if n.depth == 1]
    assert len(depth1) == 3
    assert len(state.visited) == 2
    survivor = next(n for n in depth1 if n.node_id in state.visited)
    assert survivor.content == "0.1"
    assert survivor.value == 0.9


def test_tot_dfs_uses_lifo_frontier_order():
    visits: list[str] = []

    def generate(content: str) -> list[str]:
        return [f"{content}.{i}" for i in range(2)]

    def track_goal(content: str) -> bool:
        visits.append(content)
        return len(visits) >= 6  # stop after a handful of pops

    state, proof = run_tot(
        generate,
        lambda _c: 0.5,
        track_goal,
        question="0",
        strategy="dfs",
        branching_factor=2,
        beam_width=2,
        max_depth=10,
        budgets=Budgets(loop_budget=20),
    )
    assert proof.reason is TerminationReason.GOAL_REACHED
    # DFS pops the frontier LIFO, so the LAST-generated child ("0.1") is
    # explored immediately, before backtracking to its sibling ("0.0").
    assert visits[0] == "0"
    assert visits[1] == "0.1"


def test_tot_converges_truthfully_when_no_goal_reachable_within_depth():
    state, proof = run_tot(
        _generate,
        _score,
        lambda _c: False,
        question="0",
        strategy="bfs",
        max_depth=1,
        budgets=Budgets(loop_budget=50),
    )
    assert proof.reason is TerminationReason.CONVERGED
    assert proof.degraded is True  # no node was ever marked terminal
    best = state.nodes[proof.detail]
    generated_values = [
        n.value or 0.0 for n in state.nodes.values() if n.kind is not NodeKind.ROOT
    ]
    assert best.value == max(generated_values)
    # the root's hardcoded value=1.0 must NEVER win this comparison.
    assert best.kind is not NodeKind.ROOT
