#!/usr/bin/python
from __future__ import annotations

"""ReAct topology: thought → action → observation loop.

CONCEPT:AU-ORCH.planning.reasoning-graph-topologies

Required state semantics: tool result, retry, termination, grounding — all on
the shared :class:`~.state.ReasoningState` (``tool_calls``). Grounding /
loop-termination detection reuses the EXISTING
:class:`agent_utilities.security.execution_stability_engine.DoomLoopDetector`
(identical-consecutive-call and repeating-sequence pattern detection) rather
than re-implementing loop-pattern detection — a repeated failed action is
classified as ``GROUNDING_FAILURE``, never silently retried forever.
"""

import logging
from collections.abc import Callable
from typing import Any

from ...security.execution_stability_engine import DoomLoopDetector
from .budgets import (
    BudgetExhausted,
    Budgets,
    BudgetTracker,
    TerminationProof,
    TerminationReason,
)
from .state import NodeKind, ReasoningState, ThoughtNode, ToolCallRecord
from .topology import TopologySpec

logger = logging.getLogger(__name__)

#: ``think_fn(state) -> (rationale_summary, action_name, action_args)``.
ThinkFn = Callable[[ReasoningState], tuple[str, str, dict[str, Any]]]
#: ``act_fn(action_name, action_args) -> (observation, success)``.
ActFn = Callable[[str, dict[str, Any]], tuple[str, bool]]
IsDoneFn = Callable[[ReasoningState], bool]

REACT_SPEC = TopologySpec(
    name="react",
    version="1.0.0",
    node_contracts=(
        NodeKind.ROOT.value,
        NodeKind.ACT.value,
        NodeKind.OBSERVE.value,
    ),
    loop_budget=24,
    tool_budget=24,
    termination_conditions=(
        TerminationReason.GOAL_REACHED.value,
        TerminationReason.GROUNDING_FAILURE.value,
        TerminationReason.LOOP_BUDGET_EXHAUSTED.value,
        TerminationReason.TOOL_BUDGET_EXHAUSTED.value,
        TerminationReason.TIME_BUDGET_EXHAUSTED.value,
    ),
)


def run_react(
    think_fn: ThinkFn,
    act_fn: ActFn,
    is_done_fn: IsDoneFn,
    *,
    question: str,
    budgets: Budgets | None = None,
    max_retries_per_action: int = 2,
    doom_loop_detector: DoomLoopDetector | None = None,
) -> tuple[ReasoningState, TerminationProof]:
    """Run a ReAct thought→action→observation loop.

    ``is_done_fn`` is checked BEFORE each thought (so a caller can terminate
    on the previous step's observation without one extra think/act round). A
    repeated failing action is escalated to ``GROUNDING_FAILURE`` either by
    the local per-action retry counter or by the shared
    :class:`DoomLoopDetector` — whichever trips first.
    """
    budgets = budgets or Budgets(
        loop_budget=REACT_SPEC.loop_budget, tool_budget=REACT_SPEC.tool_budget
    )
    state = ReasoningState(topology="react")
    tracker = BudgetTracker(budgets)
    detector = doom_loop_detector or DoomLoopDetector()
    root = ThoughtNode(node_id="t0", kind=NodeKind.ROOT, content=question, depth=0)
    state.add_node(root)
    retries: dict[str, int] = {}
    last_node_id = root.node_id

    try:
        while True:
            tracker.tick_loop()
            if is_done_fn(state):
                return state, TerminationProof(
                    reason=TerminationReason.GOAL_REACHED,
                    success=True,
                    degraded=False,
                    detail=last_node_id,
                )

            thought_summary, action, args = think_fn(state)
            state.append_rationale(thought_summary)

            tracker.tick_tool_call()
            observation, success = act_fn(action, args)

            retry_key = f"{action}:{sorted(args.items())}"
            retry = retries.get(retry_key, 0)
            if not success:
                retry += 1
                retries[retry_key] = retry

            state.step += 1
            act_node = ThoughtNode(
                node_id=f"a{state.step}",
                kind=NodeKind.ACT,
                content=action,
                parent_ids=[last_node_id],
                depth=state.nodes[last_node_id].depth + 1,
                metadata={"args": args},
            )
            state.add_node(act_node)
            obs_node = ThoughtNode(
                node_id=f"o{state.step}",
                kind=NodeKind.OBSERVE,
                content=observation,
                parent_ids=[act_node.node_id],
                depth=act_node.depth + 1,
                metadata={"success": success, "retry": retry},
            )
            state.add_node(obs_node)
            state.tool_calls.append(
                ToolCallRecord(
                    step=state.step,
                    action=action,
                    observation=observation,
                    success=success,
                    retry=retry,
                )
            )
            last_node_id = obs_node.node_id

            detector.record_call(action, args, observation)
            incident = detector.check()
            if not success and (retry > max_retries_per_action or incident is not None):
                return state, TerminationProof(
                    reason=TerminationReason.GROUNDING_FAILURE,
                    success=False,
                    degraded=True,
                    detail=(
                        f"action '{action}' failed {retry}x"
                        + (" (doom loop detected)" if incident is not None else "")
                    ),
                )
    except BudgetExhausted as exc:
        return state, exc.proof
