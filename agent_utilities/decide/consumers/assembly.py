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

#: goal text -> native task IRIs (a model's proposal, recorded as a claim).
TaskMapper = Callable[[str], Sequence[str]]
CommitContext = Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]]


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
) -> dict[str, Any]:
    """An ``AssemblyRequest``; a mapped goal travels as a claim, never as text."""
    mappings = []
    if goal is not None and mapped:
        mappings.append(
            {
                "text_digest": text_digest(goal),
                "task_iris": list(mapped),
                "provenance": {"producer": producer},
            }
        )
    return {
        "tenant_id": tenant,
        "requirements": {
            "tasks": list(task_iris),
            "capabilities": list(capabilities),
            "task_mappings": mappings,
        },
        "candidates": {"kinds": list(ASSEMBLY_KINDS)},
        "templates": [],
        "policy": {"policy": "default"},
    }


def _outcome(result: Mapping[str, Any]) -> Mapping[str, Any]:
    record = result.get("record") or {}
    outcome = record.get("outcome") if isinstance(record, Mapping) else None
    return outcome if isinstance(outcome, Mapping) else {}


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

    async def _commit(self, result: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if self.commit_context is None:
            return None
        record = result["record"]
        request = await self.commit_context(record)
        return await self.graphs.commit_decision(request)

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
    request = assembly_request(assembler.tenant, goal=goal, mapped=mapped)
    try:
        return run(assembler.assemble(request, lambda reasons: None))
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
    "NATIVE_TASKS",
    "assemble_goal",
    "install_assembler",
    "llm_task_mapper",
    "Assembled",
    "Assembler",
    "CommitContext",
    "TaskMapper",
    "abstain_reasons",
    "assembly_request",
    "solved",
    "spec_fields",
    "text_digest",
]
