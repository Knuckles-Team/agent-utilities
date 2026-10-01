"""The synthetic equal-budget swarm-topology benchmark (AU-CONTROL-R020).

Every policy answers the same synthetic tasks under the SAME lease budget:
a single agent; hermes-like 3 children at depth 1; goose-like 5-wide; a fixed
3-member council; AU's old family tree; and EG's certified plan (any
:data:`Planner`, e.g. :func:`eg_planner` over an installed assembler).

A task's ground truth is BY CONSTRUCTION: the acceptable topology classes and
the width its independent subtasks need. An answer succeeds when its class
is acceptable, it covers the needed width and it fits the budget; an
abstention is counted separately (abstaining is not failing). Every rate
carries a Wilson 95% interval.

:func:`width_buys_quality` is the promotion gate: exploration over width
stays at ZERO until the wider answers' success interval lies strictly above
the one-wide answers' -- on SYNTHETIC evidence, recorded as such.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

#: The benchmark's lease budget per task (concurrent agents).
EQUAL_BUDGET = 6
_Z = 1.96


@dataclass(frozen=True, slots=True)
class Task:
    """One synthetic task and its ground truth by construction."""

    name: str
    shape: str
    subtasks: int
    acceptable: frozenset[str]


@dataclass(frozen=True, slots=True)
class Answer:
    """A policy's answer: a topology class and width, or an abstention."""

    topology: str | None
    width: int = 1
    tokens: int = 0
    makespan_ms: int = 0


Planner = Callable[[Task, int], Answer]

TASKS: tuple[Task, ...] = (
    Task("atomic", "TaskShape", 1, frozenset({"Single"})),
    Task(
        "pair", "IndependentSubtasks", 2, frozenset({"FanOutJoin", "SupervisorWorkers"})
    ),
    Task(
        "survey-4",
        "IndependentSubtasks",
        4,
        frozenset({"FanOutJoin", "SupervisorWorkers"}),
    ),
    Task(
        "survey-8",
        "IndependentSubtasks",
        8,
        frozenset({"FanOutJoin", "SupervisorWorkers"}),
    ),
    Task("chain", "SequentialDependency", 1, frozenset({"Pipeline", "Single"})),
    Task("negotiate", "NeedsNegotiation", 3, frozenset({"Council", "Debate"})),
    Task("checked", "NeedsIndependentCheck", 1, frozenset({"CritiqueLoop"})),
)

_TURN_TOKENS = 4_000
_TURN_MS = 30_000


def _answer(topology: str, width: int, subtasks: int) -> Answer:
    turns = -(-subtasks // width)
    return Answer(topology, width, width * turns * _TURN_TOKENS, turns * _TURN_MS)


def au_family_tree(task: Task, budget: int) -> Answer:
    """AU's former cost-ordered family tree, projected onto classes."""
    if task.shape == "IndependentSubtasks":
        return _answer("FanOutJoin", min(task.subtasks, budget), task.subtasks)
    if task.shape == "NeedsNegotiation":
        return _answer("PeerTeam", min(3, budget), task.subtasks)
    return _answer("Single", 1, task.subtasks)


BASELINES: dict[str, Planner] = {
    # A single agent for every query.
    "single": lambda task, budget: _answer("Single", 1, task.subtasks),
    # ``_DEFAULT_MAX_CONCURRENT_CHILDREN=3``, ``MAX_DEPTH=1``.
    "hermes-3x1": lambda task, budget: _answer(
        "SupervisorWorkers", min(3, budget), task.subtasks
    ),
    # "Parallelize freely" up to ``GOOSE_MAX_BACKGROUND_TASKS=5``.
    "goose-5": lambda task, budget: _answer(
        "FanOutJoin", min(5, budget), task.subtasks
    ),
    # A 3-member council for every query.
    "council-3": lambda task, budget: _answer(
        "Council", min(3, budget), task.subtasks
    ),
    "au-tree": au_family_tree,
}


def succeeded(task: Task, answer: Answer, budget: int) -> bool:
    fan_out = task.shape == "IndependentSubtasks"
    covers = not fan_out or answer.width >= min(task.subtasks, budget)
    return answer.topology in task.acceptable and covers and answer.width <= budget


def wilson(successes: int, n: int) -> tuple[float, float]:
    """Wilson 95% interval of a rate; ``(0, 1)`` with no trials."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = (p + _Z * _Z / (2 * n)) / (1 + _Z * _Z / n)
    half = _Z * math.sqrt(p * (1 - p) / n + _Z * _Z / (4 * n * n)) / (1 + _Z * _Z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True, slots=True)
class Row:
    """One policy's results over the suite."""

    policy: str
    answered: int
    successes: int
    abstentions: int
    tokens: int
    makespan_ms: int
    success_interval: tuple[float, float]


def run_policy(
    name: str,
    planner: Planner,
    *,
    tasks: Sequence[Task] = TASKS,
    budget: int = EQUAL_BUDGET,
) -> Row:
    answers = [(task, planner(task, budget)) for task in tasks]
    answered = [(t, a) for t, a in answers if a.topology is not None]
    successes = sum(1 for t, a in answered if succeeded(t, a, budget))
    return Row(
        policy=name,
        answered=len(answered),
        successes=successes,
        abstentions=len(answers) - len(answered),
        tokens=sum(a.tokens for _, a in answered),
        makespan_ms=sum(a.makespan_ms for _, a in answered),
        success_interval=wilson(successes, len(answered)),
    )


def run_suite(policies: dict[str, Planner]) -> list[Row]:
    """Every policy over the suite under the equal budget."""
    return [run_policy(name, planner) for name, planner in policies.items()]


def eg_planner(
    assembler: Any, run: Callable[[Any], Any], templates: Callable[[], Sequence[Any]]
) -> Planner:
    """EG's certified plan as a benchmark policy: the task's shape class and
    subtasks asked of ``AgentAssemble`` under the benchmark budget as the
    width cap; an abstention is an abstention."""
    from agent_utilities.decide.consumers.topology import (
        TopologyAsk,
        ask_topology,
        plan_of,
    )
    from agent_utilities.decide.topology.schema_source import SWARM_NS

    def plan(task: Task, budget: int) -> Answer:
        ask = TopologyAsk(
            task_classes=(SWARM_NS + task.shape,),
            subtasks=task.subtasks,
            max_width=budget,
        )
        decided = plan_of(run(ask_topology(assembler, ask, templates())).result)
        if decided is None:
            return Answer(None)
        slots = list(decided.get("slots") or ())
        return Answer(
            str(decided["class_iri"]).removeprefix(SWARM_NS),
            max((int(s["width"]) for s in slots), default=1),
            sum(int(s.get("tokens") or 0) for s in slots),
            int(decided.get("makespan_ms") or 0),
        )

    return plan


def width_buys_quality(
    planner: Planner, tasks: Sequence[Task] = TASKS, budget: int = EQUAL_BUDGET
) -> bool:
    """Q2 gate: do the planner's WIDER answers succeed with an interval strictly
    above its one-wide answers' (synthetic evidence)?"""
    wide = narrow = wide_ok = narrow_ok = 0
    for task in tasks:
        answer = planner(task, budget)
        if answer.topology is None:
            continue
        ok = int(succeeded(task, answer, budget))
        if answer.width > 1:
            wide, wide_ok = wide + 1, wide_ok + ok
        else:
            narrow, narrow_ok = narrow + 1, narrow_ok + ok
    return wilson(wide_ok, wide)[0] > wilson(narrow_ok, narrow)[1]


__all__ = [
    "BASELINES",
    "EQUAL_BUDGET",
    "TASKS",
    "Answer",
    "Planner",
    "Row",
    "Task",
    "run_policy",
    "run_suite",
    "eg_planner",
    "succeeded",
    "width_buys_quality",
    "wilson",
]
