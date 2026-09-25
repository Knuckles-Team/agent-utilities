"""EH-036 / EH-037: ``graph.assemble()`` in place of LLM agent composition.

RF-ADR-010 §5 as amended (DECISIONS 2026-09-17): an agent is ASSEMBLED by EG's
decision ladder over typed requirements -- ``AgentAssemble`` returns a
one-agent graph draft with coverage derivations and an optimality
certificate, or a typed abstention naming what is missing. An LLM composes
only when EG abstains.

Free text is never a requirement. Turning a goal into native task IRIs is a
classification step (DECIDE-LAYER-DESIGN §7.1); when AU asks a model for that
mapping, the mapping re-enters EG as a ``ClaimedTaskMapping`` -- a CLAIM
premise carrying its producer, so the record's evidence class is at most
``claim`` (EH-037's assembly half; statistical abstentions re-enter through
``DecisionLog.resolve``, see :mod:`agent_utilities.decide.runner`).

The record is always returned; it is made durable (``DecisionCommit``) only
through a mutation-context provider, which the process that owns the
policy gate (graph-os) binds -- AU never mints a mutation context itself.

EH-394 (topology -> skill): the skills EG PROVED for the requested task
classes -- the ``skill_ref`` of their judged-successful retrieval paths
(``decision_proven_paths``) -- are pinned into the request at their current
Agent Library revision, so the assembled agent carries the skill its proven
retrieval topology ran under.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: The Agent Library kinds an assembly draws from.
ASSEMBLY_KINDS = ("model_profile", "skill", "system_prompt", "tool")

#: The most proven skills one assembly pins.
MAX_PROVEN_SKILL_PINS = 4
_PROVEN_SKILLS_SQL = (
    "SELECT skill_ref, successes, failures FROM decision_proven_paths "
    "WHERE task_class = {task} ORDER BY successes DESC, template_digest"
)

#: goal text -> native task IRIs (a model's proposal, recorded as a claim).
TaskMapper = Callable[[str], Sequence[str]]
CommitContext = Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class AssemblyBudget:
    """The candidate kinds, templates and constraints one assembly runs under."""

    kinds: tuple[str, ...] = ASSEMBLY_KINDS
    templates: tuple[Mapping[str, Any], ...] = ()
    context_budget_tokens: int | None = None
    require_tools: bool = False

    def constraints(self) -> dict[str, Any]:
        out: dict[str, Any] = {"require_tools": self.require_tools}
        if self.context_budget_tokens is not None:
            out["context_budget_tokens"] = int(self.context_budget_tokens)
        return out


def text_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def assembly_request(
    tenant: str,
    *,
    task_iris: Sequence[str] = (),
    capabilities: Sequence[str] = (),
    goal: str | None = None,
    mapped: Sequence[str] = (),
    producer: str = "au-task-mapper",
    budget: AssemblyBudget | None = None,
    pins: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """An ``AssemblyRequest`` -- the ONE builder every AU/graph-os assembly uses.

    Free text never travels: a goal with a model's ``mapped`` task IRIs becomes
    a ``ClaimedTaskMapping`` (a claim premise); a goal with no IRIs at all is
    sent only as its digest in ``unmapped_task_digests``, which EG answers with
    an explicit ``unmapped_task`` abstention rather than a guess.
    """
    shape = budget or AssemblyBudget()
    digest = None if goal is None else text_digest(goal)
    mappings = []
    if digest is not None and mapped:
        mappings.append(
            {
                "text_digest": digest,
                "task_iris": list(mapped),
                "provenance": {"producer": producer},
            }
        )
    typed = bool(task_iris or capabilities or mappings)
    return {
        "tenant_id": tenant,
        "requirements": {
            "tasks": sorted(set(task_iris)),
            "capabilities": sorted(set(capabilities)),
            "task_mappings": mappings,
            "unmapped_task_digests": [] if typed or digest is None else [digest],
            "constraints": shape.constraints(),
            "pins": [dict(pin) for pin in pins],
        },
        "candidates": {"kinds": list(shape.kinds)},
        "templates": [dict(t) for t in shape.templates],
        "policy": {"policy": "default"},
    }


def _outcome(result: Mapping[str, Any]) -> Mapping[str, Any]:
    record = result.get("record") or {}
    outcome = record.get("outcome") if isinstance(record, Mapping) else None
    return outcome if isinstance(outcome, Mapping) else {}


def _proven(row: Mapping[str, Any]) -> bool:
    wins, losses = int(row.get("successes") or 0), int(row.get("failures") or 0)
    return bool(row.get("skill_ref")) and wins > losses


async def proven_skills(task_iris: Sequence[str]) -> list[str]:
    """The skills EG proved for ``task_iris``: the ``skill_ref`` of their
    proven paths that won more judged runs than they lost."""
    from agent_utilities.decide.learning.ops import literal
    from agent_utilities.decide.learning.session import current_session

    session = current_session()
    if session is None:
        return []
    refs: list[str] = []
    for task in sorted(set(task_iris)):
        rows = await session.aquery(_PROVEN_SKILLS_SQL.format(task=literal(task)))
        refs.extend(str(row["skill_ref"]) for row in rows if _proven(row))
    return list(dict.fromkeys(refs))[:MAX_PROVEN_SKILL_PINS]


def _dependency(entry: Any) -> dict[str, Any]:
    kind = getattr(entry.kind, "value", entry.kind)
    return {
        "component_id": str(entry.component_id),
        "kind": str(kind),
        "definition_digest": str(entry.definition_digest),
    }


def solved(result: Mapping[str, Any]) -> bool:
    return str(_outcome(result).get("outcome")) == "solved" and bool(
        result.get("agents")
    )


def abstain_reasons(result: Mapping[str, Any]) -> list[str]:
    reasons = _outcome(result).get("reasons") or []
    return [str(r.get("reason")) for r in reasons if isinstance(r, Mapping)]


@dataclass(frozen=True, slots=True)
class Assembled:
    """What assembly concluded: the EG result, or the LLM fallback's answer."""

    result: Mapping[str, Any] | None
    fallback: Any = None
    reason: str = "solved"
    committed: Mapping[str, Any] | None = None

    @property
    def agent(self) -> Mapping[str, Any] | None:
        agents = (self.result or {}).get("agents") or []
        return agents[0] if agents else None


@dataclass
class Assembler:
    """``graph.assemble()`` bound to one tenant and one L3 client."""

    graphs: Any
    tenant: str
    commit_context: CommitContext | None = None
    #: The L1 component client pins resolve through; ``None``: the one bound to
    #: ``graphs``' client and session.
    components: Any = None

    async def skill_pins(self, task_iris: Sequence[str]) -> list[dict[str, Any]]:
        """The proven skills of ``task_iris`` at their current Agent Library
        revision (a skill not in the library is not pinned). Unreadable ->
        no pins (logged): assembly runs as it would without them."""
        try:
            refs = await proven_skills(task_iris)
            return [pin for ref in refs if (pin := await self._current(ref))]
        except Exception as exc:
            logger.warning("proven skill pins unavailable: %s", exc)
            return []

    async def _current(self, component_id: str) -> dict[str, Any] | None:
        from agent_utilities.layers.clients import ComponentClient

        components = self.components or ComponentClient(
            self.graphs.client, self.graphs.session
        )
        op = {"op": "current", "component_id": component_id, "tenant_id": self.tenant}
        entry = await components.current(op)
        return None if entry is None else _dependency(entry)

    async def assemble_mapped(self, goal: str, mapped: Sequence[str]) -> Assembled:
        """Assemble for ``goal``'s claimed task mapping, pinning the skills EG
        proved for those tasks (EH-394)."""
        pins = await self.skill_pins(mapped)
        request = assembly_request(self.tenant, goal=goal, mapped=mapped, pins=pins)
        return await self.assemble(request, lambda reasons: None)

    async def _commit(self, result: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if self.commit_context is None:
            return None
        record = result["record"]
        request = await self.commit_context(record)
        return await self.graphs.commit_decision(request)

    async def publish_routed(
        self,
        answer: Assembled,
        context: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> Any:
        """Publish the routed graph with its committed record as evidence (EH-044).

        The graph's ``synthesis_evidence`` pins the committed ``DecisionRecord``
        component, so a delegation of this graph resolves to why it was chosen
        (DECIDE-LAYER-DESIGN §4.6). Refuses an answer that was not solved AND
        committed: a graph without its decision has no evidence to carry. The
        assembled agents the graph pins must already be published.
        """
        graph = (answer.result or {}).get("graph")
        if not isinstance(graph, Mapping) or answer.committed is None:
            raise ValueError("only a solved, committed assembly is published")
        return await self.graphs.publish_graph(
            graph,
            context,
            evidence=decision_evidence(answer.committed),
            idempotency_key=idempotency_key,
        )

    async def assemble(
        self,
        request: Mapping[str, Any],
        fallback: Callable[[list[str]], Any],
    ) -> Assembled:
        """Assemble; on abstention or failure the fallback composes (named reasons)."""
        try:
            result = await self.graphs.assemble(request)
        except Exception as exc:  # noqa: BLE001 — EG unavailable costs only the fallback; cause kept in reason
            logger.warning("assembly unavailable: %s", exc)
            return Assembled(None, fallback([]), f"unavailable: {exc}")
        payload = getattr(result, "payload", result)
        if not solved(payload):
            reasons = abstain_reasons(payload)
            return Assembled(
                payload, fallback(reasons), "abstained: " + ",".join(reasons)
            )
        return Assembled(payload, None, "solved", await self._commit(payload))


def decision_evidence(committed: Mapping[str, Any]) -> dict[str, Any]:
    """The ``ComponentDependency`` pinning a ``DecisionCommitResult``'s record."""
    committed = getattr(committed, "payload", committed)
    component = (committed.get("component") or {}).get("component") or {}
    return {
        "component_id": str(committed["record_id"]),
        "kind": str(component.get("kind") or "decision_record"),
        "definition_digest": str(component["definition_digest"]),
    }


#: EG's native task vocabulary (``agent_ontology``): the only IRIs a mapping may name.
NATIVE_TASKS = (
    "eg:task/communicate",
    "eg:task/implement",
    "eg:task/operate",
    "eg:task/research",
    "eg:task/review",
)

_MAP_PROMPT = """Which of these task IRIs does the goal need? Goal:
{goal}
Task IRIs:
{tasks}
Output ONLY a JSON array of task IRIs from the list. No other text."""


def llm_task_mapper(llm_fn: Callable[[str], str]) -> TaskMapper:
    """A model mapping goal text onto native task IRIs; unknown IRIs are dropped."""
    import json

    def mapper(goal: str) -> list[str]:
        raw = llm_fn(_MAP_PROMPT.format(goal=goal, tasks="\n".join(NATIVE_TASKS)))
        try:
            proposed = json.loads(raw)
        except (TypeError, ValueError):
            return []
        items = proposed if isinstance(proposed, list) else []
        return sorted({str(i) for i in items} & set(NATIVE_TASKS))

    return mapper


_INSTALLED: list[tuple[Assembler, Callable[[Any], Any]] | None] = [None]


def install_assembler(
    assembler: Assembler | None, run: Callable[[Any], Any] | None = None
) -> None:
    """Install the process assembler and the sync driver for its coroutines."""
    _INSTALLED[0] = None if assembler is None or run is None else (assembler, run)


def assemble_goal(goal: str, mapper: TaskMapper) -> Assembled | None:
    """Assemble an agent for free-text ``goal`` from a sync call site.

    ``None`` when no assembler is installed; otherwise the :class:`Assembled`
    answer, whose ``agent`` is ``None`` when EG abstained or was unavailable
    (the caller's own composition is then the fallback). The goal reaches EG
    only as a claim-premised mapping, never as text.
    """
    installed = _INSTALLED[0]
    if installed is None:
        return None
    assembler, run = installed
    mapped = list(mapper(goal))
    if not mapped:
        return Assembled(None, None, "unmapped: the goal names no native task")
    try:
        return run(assembler.assemble_mapped(goal, mapped))
    except Exception as exc:  # noqa: BLE001 — no loop / timeout costs only the fallback; cause kept in reason
        logger.warning("assembly transport unavailable: %s", exc)
        return Assembled(None, None, f"unavailable: {exc}")


def spec_fields(agent: Mapping[str, Any]) -> dict[str, Any]:
    """``AgentSpec`` fields from an assembled ``AgentLibraryEntryDraft``."""

    def ids(key: str) -> list[str]:
        return [str(d.get("component_id")) for d in agent.get(key) or []]

    prompt = agent.get("system_prompt") or {}
    return {
        "name": str(agent.get("agent_id") or ""),
        "tools": ids("tools"),
        "skills": ids("skills"),
        "model": str(agent.get("model_identity") or ""),
        "system_prompt": f"component:{prompt.get('component_id', '')}",
    }


__all__ = [
    "ASSEMBLY_KINDS",
    "AssemblyBudget",
    "decision_evidence",
    "NATIVE_TASKS",
    "assemble_goal",
    "install_assembler",
    "llm_task_mapper",
    "Assembled",
    "Assembler",
    "CommitContext",
    "TaskMapper",
    "MAX_PROVEN_SKILL_PINS",
    "abstain_reasons",
    "assembly_request",
    "proven_skills",
    "solved",
    "spec_fields",
    "text_digest",
]
