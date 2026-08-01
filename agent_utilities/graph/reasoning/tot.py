#!/usr/bin/python
from __future__ import annotations

"""Tree-of-Thought topology: generate → score → frontier update → branch/loop.

CONCEPT:AU-ORCH.planning.reasoning-graph-topologies

Required state semantics: frontier, visited set, depth, pruning policy — all
carried on the shared :class:`~.state.ReasoningState`. ``strategy="bfs"`` pops
the frontier FIFO (breadth-first: all depth-``d`` thoughts explored before any
depth-``d+1`` thought); ``strategy="dfs"`` pops it LIFO (depth-first: a
promising branch is pushed to its full depth before backtracking). Both
strategies share the same generate/score/prune loop — only the frontier pop
order differs, which is exactly the paper's point: ToT's BFS and DFS variants
are the same topology with a one-line routing-function change, not two
separate frameworks.
"""

import logging
from collections.abc import Callable
from enum import StrEnum
from typing import Literal

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

GenerateFn = Callable[[str], list[str]]
ScoreFn = Callable[[str], float]
IsGoalFn = Callable[[str], bool]


class PruningPolicy(StrEnum):
    """How the frontier is pruned after scoring a generation batch."""

    TOP_K = "top_k"  # keep the ``beam_width`` highest-scoring children
    THRESHOLD = "threshold"  # keep every child scoring >= 0.5


_TOT_TERMINATION = (
    TerminationReason.GOAL_REACHED.value,
    TerminationReason.CONVERGED.value,
    TerminationReason.MAX_DEPTH_REACHED.value,
    TerminationReason.LOOP_BUDGET_EXHAUSTED.value,
    TerminationReason.TIME_BUDGET_EXHAUSTED.value,
)

TOT_BFS_SPEC = TopologySpec(
    name="tot_bfs",
    version="1.0.0",
    node_contracts=(NodeKind.ROOT.value, NodeKind.GENERATE.value),
    loop_budget=64,
    termination_conditions=_TOT_TERMINATION,
)

TOT_DFS_SPEC = TopologySpec(
    name="tot_dfs",
    version="1.0.0",
    node_contracts=(NodeKind.ROOT.value, NodeKind.GENERATE.value),
    loop_budget=64,
    termination_conditions=_TOT_TERMINATION,
)


def run_tot(
    generate_fn: GenerateFn,
    score_fn: ScoreFn,
    is_goal_fn: IsGoalFn,
    *,
    question: str,
    budgets: Budgets | None = None,
    tracker: BudgetTracker | None = None,
    strategy: Literal["bfs", "dfs"] = "bfs",
    branching_factor: int = 3,
    beam_width: int = 3,
    max_depth: int = 5,
    pruning: PruningPolicy = PruningPolicy.TOP_K,
) -> tuple[ReasoningState, TerminationProof]:
    """Run Tree-of-Thought search (BFS or DFS) to ``max_depth``.

    ``generate_fn(parent_content) -> [child_content, ...]`` proposes up to
    ``branching_factor`` next thoughts; ``score_fn(content) -> float in
    [0, 1]`` values one; ``is_goal_fn(content) -> bool`` decides completion.
    A goal hit returns ``GOAL_REACHED``; frontier exhaustion without a goal hit
    returns ``CONVERGED`` on the best-scored node found (``success`` reflects
    whether that best score is usable, ``degraded`` reflects whether it is a
    genuine terminal or just the best partial).
    """
    spec = TOT_BFS_SPEC if strategy == "bfs" else TOT_DFS_SPEC
    budgets = budgets or Budgets(loop_budget=spec.loop_budget)
    state = ReasoningState(topology=spec.name)
    tracker = tracker or BudgetTracker(budgets)
    root = ThoughtNode(
        node_id="n0", kind=NodeKind.ROOT, content=question, depth=0, value=1.0
    )
    state.add_node(root)
    state.frontier = [root.node_id]
    counter = 0

    try:
        while state.frontier:
            tracker.tick_loop()
            node_id = state.frontier.pop(0 if strategy == "bfs" else -1)
            if node_id in state.visited:
                continue
            state.visited.add(node_id)
            node = state.nodes[node_id]

            if is_goal_fn(node.content):
                node.is_terminal = True
                return state, TerminationProof(
                    reason=TerminationReason.GOAL_REACHED,
                    success=True,
                    degraded=False,
                    detail=node_id,
                )
            if node.depth >= max_depth:
                continue

            children_content = generate_fn(node.content)[:branching_factor]
            scored: list[ThoughtNode] = []
            for content in children_content:
                counter += 1
                child_id = f"n{counter}"
                value = score_fn(content)
                child = ThoughtNode(
                    node_id=child_id,
                    kind=NodeKind.GENERATE,
                    content=content,
                    parent_ids=[node_id],
                    depth=node.depth + 1,
                    value=value,
                )
                state.add_node(child)
                scored.append(child)

            if pruning is PruningPolicy.TOP_K:
                scored.sort(key=lambda n: (-(n.value or 0.0), n.node_id))
                survivors = scored[:beam_width]
            else:
                survivors = [n for n in scored if (n.value or 0.0) >= 0.5]
            state.frontier.extend(n.node_id for n in survivors)

        generated = [n for n in state.nodes.values() if n.kind is not NodeKind.ROOT]
        if not generated:
            return state, TerminationProof(
                reason=TerminationReason.CONVERGED,
                success=False,
                degraded=True,
                detail="empty search",
            )
        best = max(generated, key=lambda n: (n.value or 0.0, n.node_id))
        return state, TerminationProof(
            reason=TerminationReason.CONVERGED,
            success=(best.value or 0.0) > 0.0,
            degraded=not best.is_terminal,
            detail=best.node_id,
        )
    except BudgetExhausted as exc:
        return state, exc.proof
