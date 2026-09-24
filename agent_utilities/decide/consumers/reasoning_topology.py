"""EH-474: which reasoning-graph topology runs a task, asked of EG ``Decide``.

``graph_agents(action="reason", topology="auto")`` no longer picks CoT,
self-consistent CoT or Tree-of-Thought from a store the runs report into.
It offers the RUNNABLE topologies to the ``au.reasoning.topology`` decision
point: each option declares its escalation rung (cost order), its declared
loop budget, whether it is the deterministic ladder's pick, and a short shape
description EG scores against the task text (candidate-local BM25). A bound,
calibrated head picks one; otherwise EG abstains and the cheapest-adequate
escalation ladder (:class:`~agent_utilities.graph.reasoning.policy.
EscalationPolicy`) answers. Either way the record is logged, so an INDEPENDENT
evaluation can later be credited to it through ``DecisionLog.evaluate``; the
run never scores itself.

This module is the one call site of the point
(``scripts/check_topology_authority.py`` rule ``second-topology-selector``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from agent_utilities import decide
from agent_utilities.decide.options import Option, text_param
from agent_utilities.graph.reasoning import (
    COT_SPEC,
    SELF_CONSISTENT_COT_SPEC,
    TOT_BFS_SPEC,
    TOT_DFS_SPEC,
    EscalationPolicy,
    TopologySpec,
)

QUESTION = "au.reasoning.topology"
#: Longest task text sent as the BM25 parameter (a prefix is enough to score shape).
MAX_TASK_CHARS = 2_000

#: The topologies the live ``reason`` path can run, cheapest first.
RUNNABLE: tuple[TopologySpec, ...] = (
    COT_SPEC,
    SELF_CONSISTENT_COT_SPEC,
    TOT_BFS_SPEC,
    TOT_DFS_SPEC,
)

#: What each topology is good for, in words a task statement shares.
SHAPES: dict[str, str] = {
    "cot": "direct question single linear chain step by step short answer",
    "self_consistent_cot": (
        "ambiguous question several independent answers majority vote agreement"
    ),
    "tot_bfs": (
        "search explore alternatives compare candidate plans puzzle breadth options"
    ),
    "tot_dfs": "deep search backtrack long plan proof dead end depth",
}


@dataclass(frozen=True, slots=True)
class ReasoningSelection:
    """The topology to run and how it was chosen."""

    topology: str
    decided: bool
    reason: str
    record_id: str | None = None

    def report(self) -> dict[str, object]:
        return {
            "by": "decide" if self.decided else "fallback",
            "reason": self.reason,
            "record_id": self.record_id,
        }


def _ladder(runnable: Sequence[TopologySpec]) -> EscalationPolicy:
    return EscalationPolicy(ladder=tuple(spec.name for spec in runnable))


def reasoning_options(runnable: Sequence[TopologySpec], fallback: str) -> list[Option]:
    """One option per runnable topology, keyed like the published schema."""
    return [
        Option(
            spec.name,
            {
                "rung": float(rung),
                "loop_budget": float(spec.loop_budget),
                "heuristic": 1.0 if spec.name == fallback else 0.0,
            },
            texts={"shape": SHAPES.get(spec.name, spec.name.replace("_", " "))},
        )
        for rung, spec in enumerate(runnable)
    ]


async def select_reasoning_topology(
    task: str, runnable: Sequence[TopologySpec] = RUNNABLE
) -> ReasoningSelection:
    """EG's choice among ``runnable`` for ``task``; the ladder's first rung otherwise."""
    fallback = _ladder(runnable).choose().topology
    choice = await decide.achoose(
        QUESTION,
        reasoning_options(runnable, fallback),
        lambda: fallback,
        params=[text_param("task", task[:MAX_TASK_CHARS])],
    )
    return ReasoningSelection(
        topology=str(choice.option_id or fallback),
        decided=choice.decided,
        reason=choice.reason,
        record_id=choice.record_id,
    )


__all__ = [
    "MAX_TASK_CHARS",
    "QUESTION",
    "RUNNABLE",
    "SHAPES",
    "ReasoningSelection",
    "reasoning_options",
    "select_reasoning_topology",
]
