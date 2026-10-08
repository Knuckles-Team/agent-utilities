"""Answer "How can I execute task X?" as one structured, provenanced plan.

The planner composes existing authorities; it decides nothing itself:

* **tasks** -- a mapper proposes native task IRIs. The mapping is a labelled
  claim (AU-CONTROL-R008), never a proof.
* **agent components** -- EG ``AgentAssemble`` returns skills, tools, prompt
  and model with a decision record (AU-CONTROL-R003, EG-DECISION-ENGINE-R036).
* **topology and DAG** -- EG decides the topology over the published
  reference templates (AU-CONTROL-R007); the template supplies the DAG.
* **reuse** -- a capability search finds Agent Library templates and
  external A2A agents that already serve a step.
* **guardrails** -- the control-plane invariants every run obeys, plus the
  policy rules a guardrail source returns for the tasks.
* **workflow** -- a compiled workflow for the tasks when one exists.

Every element carries provenance: ``decision`` (an EG record), ``claim``,
``catalog`` (a capability-search hit), ``policy``, ``requirement`` or
``fallback``. A missing port is a named gap in the plan, never a guess.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.decide.consumers.assembly import (
    NATIVE_TASKS,
    Assembled,
    Assembler,
    TaskMapper,
    abstain_reasons,
    installed_assembler,
    spec_fields,
    text_digest,
)
from agent_utilities.decide.consumers.topology import (
    TASK_SHAPE,
    TopologyAsk,
    ask_topology,
    installed_templates,
    plan_of,
)
from agent_utilities.decide.topology.templates import REFERENCE_TEMPLATES, TemplateSpec

_PLANNING = re.compile(
    r"^\s*(?:how\s+(?:can|could|do|should)\s+(?:i|we)\b"
    r"|what\s+(?:are\s+the\s+)?steps\b"
    r"|which\s+agents\s+(?:should|do\s+(?:i|we)\s+need|are\s+needed|can)\b"
    r"|plan\s+(?:to|for|how)\b)",
    re.IGNORECASE,
)


def is_planning_question(text: str) -> bool:
    """True for "how can I ...", "what steps ...", "which agents should ..."."""
    return bool(_PLANNING.match(text or ""))


#: Native task IRIs in plan order, with the word stems that suggest each.
_TASK_STEMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "eg:task/research",
        ("research", "find", "investigat", "analy", "compar", "discover"),
    ),
    (
        "eg:task/implement",
        ("implement", "build", "writ", "code", "creat", "fix", "migrat"),
    ),
    ("eg:task/review", ("review", "audit", "verif", "check", "test", "validat")),
    (
        "eg:task/operate",
        ("deploy", "operat", "run", "restart", "monitor", "provision", "execut"),
    ),
    (
        "eg:task/communicate",
        ("notif", "email", "announc", "summar", "report", "messag"),
    ),
)
_ORDER = {iri: i for i, (iri, _) in enumerate(_TASK_STEMS)}
_WORD = re.compile(r"[a-z]+")


def keyword_task_mapper(goal: str) -> list[str]:
    """A deterministic claim: the native tasks whose stems occur in ``goal``."""
    words = _WORD.findall(goal.lower())
    return [
        iri
        for iri, stems in _TASK_STEMS
        if any(word.startswith(stem) for word in words for stem in stems)
    ]


#: Control-plane invariants every plan carries, each with its requirement.
BASE_GUARDRAILS: tuple[Mapping[str, str], ...] = (
    {
        "id": "claim-not-proof",
        "requirement": "AU-CONTROL-R008",
        "rule": "The task mapping is a labelled claim and authorizes nothing.",
    },
    {
        "id": "committed-decision",
        "requirement": "AU-CONTROL-R002",
        "rule": "Run components come from one committed EG decision.",
    },
    {
        "id": "preview-before-mutation",
        "requirement": "AU intent surface plan_ref",
        "rule": "A mutating step runs only from a reviewed preview plan_ref.",
    },
    {
        "id": "capacity-admission",
        "requirement": "AU-CONTROL-R017",
        "rule": "graph-os admits capacity first; one re-decision on denial.",
    },
    {
        "id": "narrow-only-continuation",
        "requirement": "AU-CONTROL-R019",
        "rule": "A running swarm may stop or narrow and never widens.",
    },
)

Lookup = Callable[[Sequence[str]], Awaitable[Sequence[Mapping[str, Any]]]]
WorkflowLookup = Callable[[Sequence[str]], Awaitable[Mapping[str, Any] | None]]
Templates = Callable[[], Sequence[Mapping[str, Any]]]


@dataclass
class TaskPlan:
    """The structured answer; ``to_dict`` is the wire form."""

    goal: str
    tasks: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    agents: list[dict[str, Any]] = field(default_factory=list)
    topology: dict[str, Any] = field(default_factory=dict)
    workflow: dict[str, Any] = field(default_factory=dict)
    guardrails: list[dict[str, Any]] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    def cite(self, element: str, evidence: str, **facts: Any) -> None:
        self.provenance.append({"element": element, "evidence": evidence, **facts})

    def agent_count(self) -> dict[str, int]:
        return {
            "min": sum(int(a["min_width"]) for a in self.agents),
            "max": sum(int(a["max_width"]) for a in self.agents),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "task_plan",
            "goal_digest": text_digest(self.goal),
            "tasks": self.tasks,
            "steps": self.steps,
            "agent_count": self.agent_count(),
            "agents": self.agents,
            "topology": self.topology,
            "workflow": self.workflow,
            "guardrails": self.guardrails,
            "provenance": self.provenance,
            "gaps": self.gaps,
        }


def _template(class_iri: str) -> TemplateSpec | None:
    return next((t for t in REFERENCE_TEMPLATES if t.class_iri == class_iri), None)


def _fallback_template(tasks: Sequence[str]) -> TemplateSpec:
    wanted = "swarm:single" if len(tasks) < 2 else "swarm:pipeline"
    return next(t for t in REFERENCE_TEMPLATES if t.graph_id == wanted)


def _record_id(answer: Assembled | None) -> str:
    record = ((answer.result if answer else None) or {}).get("record") or {}
    return str(record.get("record_id") or "") if isinstance(record, Mapping) else ""


def _slot_widths(spec: TemplateSpec, decided: Mapping[str, Any] | None) -> dict:
    widths = {s.node_id: (s.widths[0], s.widths[1]) for s in spec.slots}
    for slot in (decided or {}).get("slots") or ():
        width = int(slot.get("width") or 1)
        widths[str(slot.get("node_id"))] = (width, width)
    return widths


@dataclass
class TaskPlanner:
    """Composes one :class:`TaskPlan` from the installed ports."""

    mapper: TaskMapper = keyword_task_mapper
    assembler: Assembler | None = None
    templates: Templates | None = None
    capability_search: Lookup | None = None
    guardrail_source: Lookup | None = None
    workflows: WorkflowLookup | None = None
    #: The sync driver the installed assembler's coroutines run under; ``None``
    #: awaits them on the caller's loop.
    driver: Callable[[Any], Any] | None = None

    async def _eg(self, coro: Any) -> Any:
        if self.driver is None:
            return await coro
        return await asyncio.to_thread(self.driver, coro)

    async def plan(self, goal: str) -> TaskPlan:
        plan = TaskPlan(goal)
        plan.tasks = sorted(
            {t for t in self.mapper(goal) if t in NATIVE_TASKS}, key=_ORDER.get
        )
        if not plan.tasks:
            plan.gaps.append("unmapped_task: the goal names no native task")
            return plan
        plan.cite("tasks", "claim", producer=getattr(self.mapper, "__name__", "mapper"))
        components = await self._components(plan)
        spec, decided = await self._topology(plan)
        await self._agents_and_steps(plan, spec, decided, components)
        await self._guardrails(plan)
        await self._workflow(plan, spec)
        return plan

    async def _components(self, plan: TaskPlan) -> dict[str, Any]:
        if self.assembler is None:
            plan.gaps.append("assembler_not_installed: no EG agent assembly")
            return {}
        answer = await self._eg(self.assembler.assemble_mapped(plan.goal, plan.tasks))
        if answer.agent is None:
            reasons = abstain_reasons(answer.result or {}) or [answer.reason]
            plan.gaps.append("assembly_abstained: " + ",".join(reasons))
            return {}
        plan.cite("agents.components", "decision", record_id=_record_id(answer))
        return spec_fields(answer.agent)

    async def _topology(
        self, plan: TaskPlan
    ) -> tuple[TemplateSpec, Mapping[str, Any] | None]:
        decided = await self._decided_topology(plan)
        spec = _template(str((decided or {}).get("class_iri") or ""))
        if decided is None or spec is None:
            spec = _fallback_template(plan.tasks)
            plan.cite("topology", "fallback", template=spec.graph_id)
            decided = None
        plan.topology = {
            "template": spec.graph_id,
            "class_iri": spec.class_iri,
            "decided": decided is not None,
            "stop": dict((decided or {}).get("stop") or spec.stop),
            "nodes": [s.node_id for s in spec.slots] + [c[0] for c in spec.control],
            "edges": [list(edge) for edge in spec.edges],
            "entry": spec.entry,
        }
        return spec, decided

    async def _decided_topology(self, plan: TaskPlan) -> Mapping[str, Any] | None:
        if self.assembler is None or self.templates is None:
            plan.gaps.append("topology_asker_not_installed: no EG topology decision")
            return None
        ask = TopologyAsk(task_classes=(TASK_SHAPE,), subtasks=len(plan.tasks))
        answer = await self._eg(
            ask_topology(self.assembler, ask, self.templates(), task_iris=plan.tasks)
        )
        decided = plan_of(answer.result)
        if decided is None:
            reasons = abstain_reasons(answer.result or {}) or [answer.reason]
            plan.gaps.append("topology_abstained: " + ",".join(reasons))
            return None
        plan.cite("topology", "decision", record_id=_record_id(answer))
        return decided

    async def _reuse(self, plan: TaskPlan, task: str) -> list[dict[str, Any]]:
        if self.capability_search is None:
            return []
        hits = [dict(h) for h in await self.capability_search([task])][:3]
        for hit in hits:
            plan.cite(f"steps.{task}.reuse", "catalog", hit=hit.get("id"))
        return hits

    async def _agents_and_steps(
        self,
        plan: TaskPlan,
        spec: TemplateSpec,
        decided: Mapping[str, Any] | None,
        components: Mapping[str, Any],
    ) -> None:
        if self.capability_search is None:
            plan.gaps.append("capability_search_not_installed: no reuse lookup")
        slots = list(spec.slots)
        reuse: dict[str, list[dict[str, Any]]] = {}
        for index, task in enumerate(plan.tasks):
            slot = slots[min(index, len(slots) - 1)].node_id
            hits = await self._reuse(plan, task)
            reuse.setdefault(slot, []).extend(hits)
            after = [plan.steps[-1]["id"]] if plan.steps else []
            plan.steps.append(
                {
                    "id": f"step-{index + 1}",
                    "task": task,
                    "slot": slot,
                    "depends_on": after,
                    "reuse": hits,
                }
            )
        widths = _slot_widths(spec, decided)
        plan.agents = [
            _agent(
                slot.node_id,
                slot.role,
                widths[slot.node_id],
                components,
                reuse.get(slot.node_id, []),
            )
            for slot in slots
        ]

    async def _guardrails(self, plan: TaskPlan) -> None:
        plan.guardrails = [dict(g, evidence="requirement") for g in BASE_GUARDRAILS]
        if self.guardrail_source is None:
            plan.gaps.append("guardrail_source_not_installed: base guardrails only")
            return
        for rule in await self.guardrail_source(plan.tasks):
            plan.guardrails.append(dict(rule, evidence="policy"))

    async def _workflow(self, plan: TaskPlan, spec: TemplateSpec) -> None:
        compiled = None if self.workflows is None else await self.workflows(plan.tasks)
        if compiled is not None:
            plan.workflow = {"source": "compiled_workflow", "plan": dict(compiled)}
            plan.cite("workflow", "catalog", workflow=compiled.get("id"))
            return
        plan.workflow = {
            "source": "topology_template",
            "template": spec.graph_id,
            "order": [step["id"] for step in plan.steps],
        }


def _agent(
    slot: str,
    role: str,
    width: tuple[int, int],
    components: Mapping[str, Any],
    hits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reused = next(iter(hits), None)
    source = "assembled" if components else "unassigned"
    if reused is not None:
        source = "a2a" if reused.get("kind") == "a2a_agent" else "agent_library"
    return {
        "slot": slot,
        "role": role,
        "min_width": width[0],
        "max_width": width[1],
        "source": source,
        "reuses": None if reused is None else reused.get("id"),
        "skills": list(components.get("skills") or ()),
        "tools": list(components.get("tools") or ()),
        "system_prompt": components.get("system_prompt") or None,
        "model": components.get("model") or None,
    }


_INSTALLED: list[TaskPlanner | None] = [None]


def install_task_planner(planner: TaskPlanner | None) -> None:
    """Install the process planner (graph-os binds its search/policy ports)."""
    _INSTALLED[0] = planner


def process_task_planner() -> TaskPlanner:
    """The installed planner, else one over the installed EG assembler."""
    if _INSTALLED[0] is not None:
        return _INSTALLED[0]
    installed = installed_assembler()
    if installed is None:
        return TaskPlanner()
    assembler, driver = installed
    return TaskPlanner(
        assembler=assembler, templates=installed_templates(), driver=driver
    )


__all__ = [
    "BASE_GUARDRAILS",
    "TaskPlan",
    "TaskPlanner",
    "install_task_planner",
    "is_planning_question",
    "keyword_task_mapper",
    "process_task_planner",
]
