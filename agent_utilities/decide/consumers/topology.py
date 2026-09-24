"""EH-048 / ST-7: the swarm topology question, asked of EG's assembly decision.

SWARM-TOPOLOGY-DECIDE-DESIGN §1: *how should this task be split across agents,
how wide and deep, on what resources, and when does it stop?* is ONE question,
answered by EG ``AgentAssemble`` over the published topology templates with a
``requirements.topology`` section. The answer is a certified ``TopologyPlan``
(per-slot width and rounds, stop rule, lease plan, sub-agent allowances) or a
typed abstention; there is no AU-side selector, learning store or success rate
(invariants T4/T5). The old cost-ordered family tree stays only as the
deterministic fallback for a caller no topology asker is installed for.

The plan PROPOSES capacity; graph-os acquires it (``AcquireCapacity``) after
commit, and the harness admission is built from the committed plan
(:func:`agent_utilities.graph.topology_engine.ElasticTopologyAdmission.from_plan`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_utilities.decide.consumers.assembly import (
    Assembled,
    Assembler,
    AssemblyBudget,
    assembly_request,
)
from agent_utilities.decide.topology.schema_source import SWARM_NS

logger = logging.getLogger(__name__)

#: Swarm task-shape classes (the vocabulary's local names).
INDEPENDENT_SUBTASKS = SWARM_NS + "IndependentSubtasks"
NEEDS_NEGOTIATION = SWARM_NS + "NeedsNegotiation"
TASK_SHAPE = SWARM_NS + "TaskShape"


@dataclass(frozen=True, slots=True)
class TopologyAsk:
    """The typed topology requirements of one task (``TopologyRequirements``).

    Caps may only TIGHTEN the decision policy's (EG refuses a loosening with
    ``POLICY_LOOSENING``); ``cells`` are the capacity cells the plan must fit,
    read at ``priority``.
    """

    task_classes: tuple[str, ...]
    subtasks: int = 1
    per_agent_subtasks: int = 1
    deadline_ms: int | None = None
    max_width: int = 16
    max_depth: int = 8
    max_rounds: int = 8
    max_tokens: int | None = None
    cells: tuple[str, ...] = ()
    priority: str = "orchestration"

    def requirements(self) -> dict[str, Any]:
        caps: dict[str, Any] = {
            "max_width": self.max_width,
            "max_depth": self.max_depth,
            "max_rounds": self.max_rounds,
        }
        if self.max_tokens is not None:
            caps["max_tokens"] = self.max_tokens
        body: dict[str, Any] = {
            "task_classes": list(self.task_classes),
            "subtasks": self.subtasks,
            "per_agent_subtasks": self.per_agent_subtasks,
            "caps": caps,
            "capacity": {"cells": list(self.cells), "priority": self.priority},
        }
        if self.deadline_ms is not None:
            body["deadline_ms"] = self.deadline_ms
        return body


def topology_request(
    tenant: str,
    ask: TopologyAsk,
    templates: Sequence[Mapping[str, Any]],
    *,
    task_iris: Sequence[str] = (),
    capabilities: Sequence[str] = (),
) -> dict[str, Any]:
    """The ``AssemblyRequest`` asking the topology question over ``templates``."""
    request = assembly_request(
        tenant,
        task_iris=task_iris,
        capabilities=capabilities,
        budget=AssemblyBudget(templates=tuple(templates)),
    )
    request["requirements"]["topology"] = ask.requirements()
    return request


def plan_of(result: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The ``TopologyPlan`` of a solved topology record, else ``None``."""
    record = (result or {}).get("record") or {}
    outcome = record.get("outcome") if isinstance(record, Mapping) else None
    if not isinstance(outcome, Mapping) or outcome.get("outcome") != "solved":
        return None
    plan = outcome.get("topology")
    return plan if isinstance(plan, Mapping) else None


async def ask_topology(
    assembler: Assembler,
    ask: TopologyAsk,
    templates: Sequence[Mapping[str, Any]],
    *,
    task_iris: Sequence[str] = (),
    capabilities: Sequence[str] = (),
) -> Assembled:
    """Ask EG the topology question; commit a solved plan when the assembler
    carries a commit context (graph-os), evaluate-only otherwise (work-market
    pricing). An abstention is returned, never replaced by a guess."""
    request = topology_request(
        assembler.tenant,
        ask,
        templates,
        task_iris=task_iris,
        capabilities=capabilities,
    )
    return await assembler.assemble(request, lambda reasons: None)


#: A decided topology class -> the AU executor family that runs it. A
#: projection of EG's answer onto AU's executors, never a selection.
_FAMILY: dict[str, str] = {
    "Single": "inline_tool",
    "Pipeline": "agent_pool",
    "FanOutJoin": "fan_out",
    "SupervisorWorkers": "fan_out",
    "PeerTeam": "agent_pool",
    "Debate": "agent_pool",
    "Council": "agent_pool",
    "CritiqueLoop": "agent_pool",
    "Hierarchy": "teams",
}


def family_of(plan: Mapping[str, Any]) -> str | None:
    """The executor family for a plan's class; ``None`` for an extension class."""
    class_iri = str(plan.get("class_iri") or "")
    if not class_iri.startswith(SWARM_NS):
        return None
    return _FAMILY.get(class_iri[len(SWARM_NS) :])


@dataclass(frozen=True, slots=True)
class TaskShape:
    """The task facts the family tree (the fallback) reads."""

    complexity: int
    parallelizable: bool
    needs_collaboration: bool
    specialist_count: int
    has_a2a_peers: bool

    def ask(self) -> TopologyAsk:
        """The typed topology requirements these facts state."""
        classes = [TASK_SHAPE]
        if self.parallelizable:
            classes.append(INDEPENDENT_SUBTASKS)
        if self.needs_collaboration:
            classes.append(NEEDS_NEGOTIATION)
        return TopologyAsk(
            task_classes=tuple(classes), subtasks=max(1, self.specialist_count)
        )


Templates = Callable[[], Sequence[Mapping[str, Any]]]
_INSTALLED: list[tuple[Assembler, Callable[[Any], Any], Templates] | None] = [None]


def install_topology(
    assembler: Assembler | None,
    run: Callable[[Any], Any] | None = None,
    templates: Templates | None = None,
) -> None:
    """Install the process topology asker: assembler, sync driver, templates."""
    if assembler is None or run is None or templates is None:
        _INSTALLED[0] = None
        return
    _INSTALLED[0] = (assembler, run, templates)


def planned_topology(shape: TaskShape) -> tuple[Mapping[str, Any], str] | None:
    """``(plan, record_id)`` EG decided for ``shape``, or ``None`` (no asker,
    abstention or unavailable engine: the caller's tree is the fallback)."""
    installed = _INSTALLED[0]
    if installed is None:
        return None
    assembler, run, templates = installed
    try:
        answer = run(ask_topology(assembler, shape.ask(), templates()))
    except (RuntimeError, ConnectionError, TimeoutError, ValueError) as exc:
        logger.warning("topology asker unavailable: %s", type(exc).__name__)
        return None
    plan = plan_of(answer.result)
    record = (answer.result or {}).get("record") or {}
    if plan is None:
        return None
    return plan, str(record.get("record_id") or "")


def decided_topology(pattern: str, reasoning: str, shape: TaskShape) -> tuple[str, str]:
    """``(family, reasoning)``: EG's plan when one was decided, else the tree's."""
    decided = planned_topology(shape)
    family = None if decided is None else family_of(decided[0])
    if decided is None or family is None:
        return pattern, reasoning
    return family, f"{reasoning} [EG topology plan {decided[1]}]"


__all__ = [
    "TaskShape",
    "TopologyAsk",
    "ask_topology",
    "decided_topology",
    "family_of",
    "install_topology",
    "plan_of",
    "planned_topology",
    "topology_request",
]
