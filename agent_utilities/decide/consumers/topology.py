"""EH-048: swarm topology through EG ``Decide`` (DESIGN §4).

The topology family -- inline tool, fan-out, agent pool, team -- is a
``route`` decision over declared options, each carrying what it costs and
what it offers for THIS task: agents it runs, whether it gives the agents a
messaging channel, and the task's own shape (complexity, parallelism,
collaboration, A2A peers). The existing cost-ordered decision tree is the
deterministic fallback and a declared fact (``heuristic``), so a head fitted
on outcomes can learn when to overrule it. Filling a chosen topology's slots
with concrete agents is an assembly over the operator-published multi-agent
templates (``graphos.route_a2a_task`` / ``graph.assemble()``); outcomes are
credited to that whole slate (EH-012), never per edge.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_utilities import decide
from agent_utilities.decide.options import Option

#: Topology -> (agents it runs as a function of the specialist count, messaging).
_SHAPES: dict[str, tuple[int, float]] = {
    "inline_tool": (0, 0.0),
    "fan_out": (1, 0.0),
    "agent_pool": (1, 1.0),
    "teams": (2, 1.0),
}


@dataclass(frozen=True, slots=True)
class TaskShape:
    """The task facts the topology tree reads."""

    complexity: int
    parallelizable: bool
    needs_collaboration: bool
    specialist_count: int
    has_a2a_peers: bool


def _agents(extra: int, specialists: int) -> float:
    return 1.0 if extra == 0 else float(specialists + extra - 1)


def _options(shape: TaskShape, heuristic: str) -> list[Option]:
    task = {
        "complexity": float(shape.complexity),
        "parallelizable": 1.0 if shape.parallelizable else 0.0,
        "collaboration": 1.0 if shape.needs_collaboration else 0.0,
        "a2a_peers": 1.0 if shape.has_a2a_peers else 0.0,
    }
    return [
        Option(
            name,
            {
                **task,
                "agents": _agents(extra, max(1, shape.specialist_count)),
                "messaging": messaging,
                "heuristic": 1.0 if name == heuristic else 0.0,
            },
        )
        for name, (extra, messaging) in _SHAPES.items()
    ]


def choose_topology(shape: TaskShape, heuristic: str) -> decide.Choice:
    """The topology for one task; ``heuristic`` (the tree's pick) is the fallback."""
    return decide.choose(
        "au.swarm.topology", _options(shape, heuristic), lambda: heuristic
    )


def decided_topology(pattern: str, reasoning: str, shape: TaskShape) -> tuple[str, str]:
    """``(pattern, reasoning)`` after EG; the reasoning names who decided and the record."""
    choice = choose_topology(shape, pattern)
    if not choice.decided or choice.option_id is None:
        return pattern, reasoning
    return (
        choice.option_id,
        f"{reasoning} [EG decided {choice.option_id}: {choice.record_id}]",
    )


__all__ = ["TaskShape", "choose_topology", "decided_topology"]
