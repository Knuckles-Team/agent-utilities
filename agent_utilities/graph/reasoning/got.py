#!/usr/bin/python
from __future__ import annotations

"""Graph-of-Thought topology: generate / transform / aggregate / refine DAG.

CONCEPT:AU-ORCH.planning.reasoning-graph-topologies

Required state semantics: a typed thought graph + merge provenance. A GoT plan
is an explicit, ordered list of :class:`GotOp` steps executed over the shared
:class:`~.state.ReasoningState`; an ``AGGREGATE`` op naming more than one
``parent_ids`` entry is itself the merge-provenance record (the resulting
node's ``parent_ids`` — no separate structure is needed, per this package's
central thesis that these algorithms share one state shape).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

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

GOT_SPEC = TopologySpec(
    name="got",
    version="1.0.0",
    node_contracts=(
        NodeKind.ROOT.value,
        NodeKind.GENERATE.value,
        NodeKind.TRANSFORM.value,
        NodeKind.AGGREGATE.value,
        NodeKind.REFINE.value,
    ),
    loop_budget=48,
    termination_conditions=(
        TerminationReason.GOAL_REACHED.value,
        TerminationReason.LOOP_BUDGET_EXHAUSTED.value,
        TerminationReason.TIME_BUDGET_EXHAUSTED.value,
    ),
)


@dataclass
class GotOp:
    """One step of an explicit Graph-of-Thought plan.

    ``fn`` semantics depend on ``kind``:
    * ``GENERATE``/``TRANSFORM`` — ``fn(*parent_contents) -> str``.
    * ``AGGREGATE`` — ``fn(parent_contents: list[str]) -> str`` (the merge).
    * ``REFINE`` — ``fn(current_content: str) -> str``, iterated up to
      ``refine_rounds`` times, keeping the highest-``score_fn`` candidate seen.
    """

    op_id: str
    kind: NodeKind
    parent_ids: list[str]
    fn: Callable[..., str]
    score_fn: Callable[[str], float] | None = None
    refine_rounds: int = 1
    metadata: dict[str, object] = field(default_factory=dict)


def run_got(
    question: str,
    plan: list[GotOp],
    *,
    budgets: Budgets | None = None,
    tracker: BudgetTracker | None = None,
) -> tuple[ReasoningState, TerminationProof]:
    """Execute an explicit GoT plan over the shared reasoning state.

    Every op is one loop-budget tick (REFINE additionally ticks once per
    refine round). On success the final plan step's node is marked terminal
    and ``GOAL_REACHED`` is returned; a budget exhaustion mid-plan returns the
    ops completed so far with ``degraded=True``.
    """
    budgets = budgets or Budgets(loop_budget=GOT_SPEC.loop_budget)
    state = ReasoningState(topology="got")
    tracker = tracker or BudgetTracker(budgets)
    root = ThoughtNode(node_id="root", kind=NodeKind.ROOT, content=question, depth=0)
    state.add_node(root)

    last_op_id = root.node_id
    try:
        for op in plan:
            tracker.tick_loop()
            parent_ids = op.parent_ids or [root.node_id]
            parent_contents = [state.nodes[pid].content for pid in parent_ids]
            depth = max(state.nodes[pid].depth for pid in parent_ids) + 1

            if op.kind in (NodeKind.GENERATE, NodeKind.TRANSFORM):
                content = op.fn(*parent_contents)
                node = ThoughtNode(
                    node_id=op.op_id,
                    kind=op.kind,
                    content=content,
                    parent_ids=parent_ids,
                    depth=depth,
                )
            elif op.kind is NodeKind.AGGREGATE:
                content = op.fn(parent_contents)
                node = ThoughtNode(
                    node_id=op.op_id,
                    kind=NodeKind.AGGREGATE,
                    content=content,
                    parent_ids=parent_ids,
                    depth=depth,
                    metadata={"merged_from": list(parent_ids), **op.metadata},
                )
            elif op.kind is NodeKind.REFINE:
                current = parent_contents[0]
                best_score = op.score_fn(current) if op.score_fn else 0.0
                for _round in range(max(1, op.refine_rounds)):
                    tracker.tick_loop()
                    candidate = op.fn(current)
                    score = op.score_fn(candidate) if op.score_fn else best_score + 1.0
                    if op.score_fn is None or score >= best_score:
                        current, best_score = candidate, score
                node = ThoughtNode(
                    node_id=op.op_id,
                    kind=NodeKind.REFINE,
                    content=current,
                    parent_ids=parent_ids,
                    depth=depth,
                    value=best_score,
                )
            else:
                raise ValueError(f"unsupported GotOp.kind: {op.kind}")

            state.add_node(node)
            state.step += 1
            last_op_id = node.node_id

        terminal = state.nodes[last_op_id]
        terminal.is_terminal = True
        return state, TerminationProof(
            reason=TerminationReason.GOAL_REACHED,
            success=True,
            degraded=False,
            detail=last_op_id,
        )
    except BudgetExhausted as exc:
        return state, exc.proof
