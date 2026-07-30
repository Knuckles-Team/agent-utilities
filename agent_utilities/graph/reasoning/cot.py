#!/usr/bin/python
from __future__ import annotations

"""Chain-of-Thought and Self-Consistent Chain-of-Thought topologies.

CONCEPT:AU-ORCH.planning.reasoning-graph-topologies

**CoT**: a linear chain of ``generate`` nodes, each with exactly one parent
(the previous step). At each step an append-only rationale SUMMARY is
recorded (never the model's raw private reasoning — see
:meth:`~.state.ReasoningState.append_rationale`); the chain terminates on
``GOAL_REACHED`` or loop-budget exhaustion, whichever comes first.

**Self-consistent CoT**: fans out ``num_samples`` independent CoT chains
sharing one :class:`~.budgets.Budgets` (candidate distribution — each sample
is one :class:`~.state.CandidateVote` with its own cost/confidence), then
aggregates by majority vote over normalized final answers. Diversity-aware
pruning (dropping near-duplicate chains before voting) reuses
:func:`agent_utilities.graph.test_time_diversity.select_diverse` when an
embedding function is supplied — the voting/aggregation itself is new (VPO's
module provides diversity selection, not a vote).
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
from .state import CandidateVote, NodeKind, ReasoningState, ThoughtNode
from .topology import TopologySpec

logger = logging.getLogger(__name__)

#: A step function drives one CoT step: given the running state, return
#: ``(rationale_summary, next_content, is_final)``.
StepFn = Callable[[ReasoningState], tuple[str, str, bool]]

COT_SPEC = TopologySpec(
    name="cot",
    version="1.0.0",
    node_contracts=(NodeKind.ROOT.value, NodeKind.GENERATE.value),
    loop_budget=32,
    termination_conditions=(
        TerminationReason.GOAL_REACHED.value,
        TerminationReason.LOOP_BUDGET_EXHAUSTED.value,
        TerminationReason.TOKEN_BUDGET_EXHAUSTED.value,
        TerminationReason.TIME_BUDGET_EXHAUSTED.value,
    ),
)

SELF_CONSISTENT_COT_SPEC = TopologySpec(
    name="self_consistent_cot",
    version="1.0.0",
    node_contracts=(NodeKind.ROOT.value, NodeKind.GENERATE.value, NodeKind.VOTE.value),
    loop_budget=COT_SPEC.loop_budget * 8,
    termination_conditions=COT_SPEC.termination_conditions
    + (TerminationReason.CONVERGED.value,),
)


def run_cot(
    step_fn: StepFn,
    *,
    question: str,
    budgets: Budgets | None = None,
) -> tuple[ReasoningState, TerminationProof]:
    """Run one linear Chain-of-Thought reasoning chain.

    Returns the accumulated :class:`~.state.ReasoningState` and a truthful
    :class:`~.budgets.TerminationProof` — a budget-halted chain returns the
    partial chain built so far with ``degraded=True``, never a fabricated
    final answer.
    """
    budgets = budgets or Budgets(loop_budget=COT_SPEC.loop_budget)
    state = ReasoningState(topology="cot")
    tracker = BudgetTracker(budgets)
    root = ThoughtNode(node_id="n0", kind=NodeKind.ROOT, content=question, depth=0)
    state.add_node(root)
    parent_id = root.node_id
    try:
        while True:
            tracker.tick_loop()
            summary, content, is_final = step_fn(state)
            state.append_rationale(summary)
            state.step += 1
            node_id = f"n{state.step}"
            parent = state.nodes[parent_id]
            node = ThoughtNode(
                node_id=node_id,
                kind=NodeKind.GENERATE,
                content=content,
                parent_ids=[parent_id],
                depth=parent.depth + 1,
                is_terminal=is_final,
            )
            state.add_node(node)
            parent_id = node_id
            if is_final:
                return state, TerminationProof(
                    reason=TerminationReason.GOAL_REACHED,
                    success=True,
                    degraded=False,
                    detail=node_id,
                )
    except BudgetExhausted as exc:
        return state, exc.proof


def _normalize(answer: str) -> str:
    return " ".join(answer.strip().lower().split())


def run_self_consistent_cot(
    step_fn: StepFn,
    *,
    question: str,
    num_samples: int,
    budgets: Budgets | None = None,
    cost_fn: Callable[[str], int] | None = None,
) -> tuple[ReasoningState, TerminationProof, str]:
    """Fan out ``num_samples`` independent CoT chains and vote on the answer.

    Every sample shares ONE :class:`~.budgets.Budgets` (a single
    :class:`~.budgets.BudgetTracker` counts loops across ALL chains), so a
    tight budget yields fewer-than-requested samples rather than silently
    overrunning it — the vote still runs over whatever candidates were
    collected, and the proof is marked ``degraded`` when fewer than
    ``num_samples`` chains completed.

    Returns ``(state, proof, winning_answer)`` — ``state.candidates`` holds
    the full candidate distribution (answer/votes/cost/confidence) for
    inspection.
    """
    num_samples = max(1, num_samples)
    budgets = budgets or Budgets(loop_budget=SELF_CONSISTENT_COT_SPEC.loop_budget)
    state = ReasoningState(topology="self_consistent_cot")
    tracker = BudgetTracker(budgets)
    root = ThoughtNode(node_id="root", kind=NodeKind.ROOT, content=question, depth=0)
    state.add_node(root)

    completed = 0
    proof: TerminationProof | None = None
    try:
        for sample_idx in range(num_samples):
            parent_id = root.node_id
            branch_state = ReasoningState(topology="cot")
            branch_state.add_node(root)
            while True:
                tracker.tick_loop()
                summary, content, is_final = step_fn(branch_state)
                branch_state.append_rationale(summary)
                state.append_rationale(f"sample {sample_idx}: {summary}")
                branch_state.step += 1
                node_id = f"s{sample_idx}n{branch_state.step}"
                parent = branch_state.nodes[parent_id]
                node = ThoughtNode(
                    node_id=node_id,
                    kind=NodeKind.GENERATE,
                    content=content,
                    parent_ids=[parent_id],
                    depth=parent.depth + 1,
                    is_terminal=is_final,
                )
                branch_state.add_node(node)
                state.add_node(node)
                parent_id = node_id
                if is_final:
                    answer = content
                    key = _normalize(answer)
                    cost = cost_fn(answer) if cost_fn else len(answer.split())
                    existing = state.candidates.get(key)
                    if existing is None:
                        state.candidates[key] = CandidateVote(
                            candidate_id=key,
                            answer=answer,
                            votes=1,
                            cost_tokens=cost,
                        )
                    else:
                        existing.votes += 1
                        existing.cost_tokens += cost
                    completed += 1
                    break
    except BudgetExhausted as exc:
        proof = exc.proof

    total_votes = sum(c.votes for c in state.candidates.values()) or 1
    for candidate in state.candidates.values():
        candidate.confidence = candidate.votes / total_votes

    if not state.candidates:
        no_answer_proof = proof or TerminationProof(
            reason=TerminationReason.LOOP_BUDGET_EXHAUSTED,
            success=False,
            degraded=True,
            detail="no chain completed",
        )
        return state, no_answer_proof, ""

    winner = max(state.candidates.values(), key=lambda c: (c.votes, c.candidate_id))
    degraded = completed < num_samples
    final_proof = TerminationProof(
        reason=(proof.reason if proof else TerminationReason.CONVERGED),
        success=True,
        degraded=degraded,
        detail=f"{completed}/{num_samples} chains, winner votes={winner.votes}",
    )
    return state, final_proof, winner.answer
