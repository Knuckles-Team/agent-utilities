#!/usr/bin/python
from __future__ import annotations

"""RAP topology: select → expand → simulate → backpropagate (real MCTS).

CONCEPT:AU-ORCH.planning.rap-mcts-backpropagation,
CONCEPT:AU-ORCH.planning.reasoning-graph-topologies

Ports "Reasoning with Language Model is Planning with World Model" (RAP,
arXiv:2305.14992): world-model-guided Monte-Carlo Tree Search over reasoning
states, with UCT selection, expansion, rollout/evaluation ("simulation"), and
reward backpropagation.

**Backpropagation is real, not approximated**: :func:`_backpropagate` walks
the FULL parent chain from the simulated node to the root
(:meth:`~.state.ReasoningState.path_to_root`), incrementing ``visits`` and
``total_reward`` on EVERY ancestor — not only the immediate parent, and not
only the simulated leaf. A "simplified" MCTS/RAP port commonly gets exactly
this step wrong (crediting only the local parent, or only the leaf), which
silently breaks UCT everywhere above the first level: a deep, high-value line
never influences selection at the root or mid-tree, so the search degenerates
into flat rollouts. This repo already has one genuinely correct
implementation of this exact mechanism —
:meth:`agent_utilities.harness.graph_search_evolution.GraphSearchEvolver.
_backpropagate` (MLEvolve's code-evolution MCTS) — and this module reuses
that same algorithmic shape (UCT selection + full-path backprop), generalized
off the code-evolution specifics (coder/evaluator callables) to a
general-purpose ``expand``/``simulate`` reasoning search. See
``tests/unit/graph/reasoning/test_rap.py::test_backpropagation_updates_every_ancestor``
for the proof.
"""

import logging
from collections.abc import Callable

from .budgets import (
    BudgetExhausted,
    Budgets,
    BudgetTracker,
    TerminationProof,
    TerminationReason,
)
from .state import NodeKind, ReasoningState, ThoughtNode
from .topology import TopologySpec

logger = logging.getLogger(__name__)

#: ``expand_fn(node_content) -> [child_content, ...]`` proposes next reasoning steps.
ExpandFn = Callable[[str], list[str]]
#: ``simulate_fn(node_content) -> reward in [0, 1]`` evaluates/rolls out a node.
SimulateFn = Callable[[str], float]
IsGoalFn = Callable[[str], bool]

RAP_SPEC = TopologySpec(
    name="rap",
    version="1.0.0",
    node_contracts=(NodeKind.ROOT.value, NodeKind.GENERATE.value),
    loop_budget=64,
    termination_conditions=(
        TerminationReason.GOAL_REACHED.value,
        TerminationReason.CONVERGED.value,
        TerminationReason.LOOP_BUDGET_EXHAUSTED.value,
        TerminationReason.TIME_BUDGET_EXHAUSTED.value,
    ),
)


def _select(state: ReasoningState, root_id: str, c: float) -> str:
    """Descend from the root by UCT, stopping at the first node with no
    expanded children yet (the classic MCTS selection phase)."""
    node_id = root_id
    while True:
        node = state.nodes[node_id]
        if node.is_terminal:
            return node_id
        children = state.children_of(node_id)
        if not children:
            return node_id
        parent_visits = node.visits
        node_id = max(
            children,
            key=lambda cid: (state.nodes[cid].uct(parent_visits, c), cid),
        )


def _expand(
    state: ReasoningState,
    node_id: str,
    expand_fn: ExpandFn,
    counter: list[int],
    branching_factor: int,
) -> list[str]:
    """Materialize this node's children (the expansion phase), capped at
    ``branching_factor``.

    The cap mirrors ToT's (:mod:`.tot`) and is what keeps ONE loop tick's work
    bounded: without it a caller-supplied ``expand_fn`` returning a huge (or
    generated) child list ran a full simulate+backpropagate per child with no
    budget check in between, so a declared ``loop_budget``/``time_budget_s``
    could be overshot by orders of magnitude before the next
    :meth:`BudgetTracker.tick_loop`.
    """
    parent = state.nodes[node_id]
    child_ids: list[str] = []
    for content in list(expand_fn(parent.content))[:branching_factor]:
        counter[0] += 1
        child_id = f"r{counter[0]}"
        child = ThoughtNode(
            node_id=child_id,
            kind=NodeKind.GENERATE,
            content=content,
            parent_ids=[node_id],
            depth=parent.depth + 1,
        )
        state.add_node(child)
        child_ids.append(child_id)
    return child_ids


def _simulate(state: ReasoningState, node_id: str, simulate_fn: SimulateFn) -> float:
    """Evaluate/roll out one node (the simulation phase)."""
    return simulate_fn(state.nodes[node_id].content)


def _backpropagate(state: ReasoningState, leaf_id: str, reward: float) -> None:
    """Propagate ``reward`` to EVERY ancestor of ``leaf_id``, root included.

    This is the hardened step: it walks the whole primary-edge path
    (:meth:`~.state.ReasoningState.path_to_root`), not just the immediate
    parent, so a value discovered deep in the tree genuinely changes UCT at
    every level above it.
    """
    for node_id in state.path_to_root(leaf_id):
        node = state.nodes[node_id]
        node.visits += 1
        node.total_reward += reward


def run_rap(
    expand_fn: ExpandFn,
    simulate_fn: SimulateFn,
    is_goal_fn: IsGoalFn,
    *,
    question: str,
    budgets: Budgets | None = None,
    tracker: BudgetTracker | None = None,
    exploration_c: float = 1.4,
    branching_factor: int = 4,
) -> tuple[ReasoningState, TerminationProof]:
    """Run RAP: select → expand → simulate → backpropagate, one loop-tick per
    iteration, until a goal is hit or the budget is exhausted.

    Each iteration selects a node by UCT; an unvisited selected node is
    simulated directly (standard "simulate before expanding" MCTS —
    :func:`_expand` only runs once a node has already been visited once);
    otherwise it is expanded and every new child is immediately simulated and
    backpropagated. A budget-exhausted run returns the highest-visit node
    found so far (the MCTS-standard "most robust" pick) with ``degraded=True``.
    """
    budgets = budgets or Budgets(loop_budget=RAP_SPEC.loop_budget)
    state = ReasoningState(topology="rap")
    tracker = tracker or BudgetTracker(budgets)
    root = ThoughtNode(node_id="r0", kind=NodeKind.ROOT, content=question, depth=0)
    state.add_node(root)
    counter = [0]

    try:
        while True:
            tracker.tick_loop()
            leaf_id = _select(state, root.node_id, exploration_c)
            leaf = state.nodes[leaf_id]

            if is_goal_fn(leaf.content):
                leaf.is_terminal = True
                _backpropagate(state, leaf_id, 1.0)
                return state, TerminationProof(
                    reason=TerminationReason.GOAL_REACHED,
                    success=True,
                    degraded=False,
                    detail=leaf_id,
                )

            if leaf.visits == 0 and leaf_id != root.node_id:
                reward = _simulate(state, leaf_id, simulate_fn)
                _backpropagate(state, leaf_id, reward)
                continue

            child_ids = _expand(state, leaf_id, expand_fn, counter, branching_factor)
            if not child_ids:
                leaf.is_terminal = True
                reward = _simulate(state, leaf_id, simulate_fn)
                _backpropagate(state, leaf_id, reward)
                continue

            for child_id in child_ids:
                # Re-check the time/token/cost axes BETWEEN children, not only
                # once per outer loop tick: a single expansion fans out up to
                # ``branching_factor`` simulations, and a declared
                # ``time_budget_s`` must be able to trip inside that fan-out.
                tracker.check_time_and_cost()
                reward = _simulate(state, child_id, simulate_fn)
                _backpropagate(state, child_id, reward)
    except BudgetExhausted as exc:
        candidates = [n for n in state.nodes.values() if n.node_id != root.node_id]
        if candidates:
            best = max(candidates, key=lambda n: (n.visits, n.total_reward, n.node_id))
            exc.proof.detail = (
                f"{exc.proof.detail}; best={best.node_id} "
                f"visits={best.visits} mean_reward="
                f"{(best.total_reward / best.visits) if best.visits else 0.0:.3f}"
            )
        return state, exc.proof


__all__ = [
    "ExpandFn",
    "SimulateFn",
    "IsGoalFn",
    "RAP_SPEC",
    "run_rap",
]
