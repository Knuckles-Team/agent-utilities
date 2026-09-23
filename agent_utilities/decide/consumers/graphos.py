"""EH-044 / EH-045: graph-os-side decisions, as ``graph.assemble()`` calls graph-os makes.

graph-os owns both surfaces (the A2A inbound endpoint and the multiplexer's
tool exposure, W5); these are the decision halves it calls, so the choice is
EG's assembly -- typed requirements, a verifiable certificate or a typed
abstention -- and graph-os keeps its current behaviour as the fallback.

* EH-044 A2A inbound task -> agent graph: the task's skills, mapped to native
  capability IRIs, are the requirements; the operator-published agent-graph
  templates are the options. Abstention -> graph-os's current routing.
* EH-045 multiplexer tool exposure: the smallest tool subset covering the
  task's capabilities within a context-token budget -- the assembly objective
  (uncovered, then fewest components) is exactly that. EVALUATE-ONLY: the
  record is returned for sampling, never committed per request (§4.5).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_utilities.decide.consumers.assembly import (
    Assembled,
    Assembler,
    AssemblyBudget,
    assembly_request,
)

#: A tool subset request draws only tools.
TOOL_KINDS = ("tool",)


async def route_a2a_task(
    assembler: Assembler,
    current: Callable[[list[str]], Any],
    *,
    task_iris: Sequence[str] = (),
    capabilities: Sequence[str] = (),
    context_budget_tokens: int | None = None,
    templates: Sequence[Mapping[str, Any]] = (),
    text: str | None = None,
) -> Assembled:
    """The agent graph for one inbound A2A task; ``current`` routes on abstention.

    Requirements are typed (task / capability IRIs) under an optional context
    budget over the published agent-graph ``templates``; ``text`` is never
    sent -- without IRIs only its digest goes, and EG abstains with
    ``unmapped_task``. Publish a solved answer with
    :meth:`Assembler.publish_routed`.
    """
    request = assembly_request(
        assembler.tenant,
        task_iris=task_iris,
        capabilities=capabilities,
        goal=text,
        budget=AssemblyBudget(
            templates=tuple(templates), context_budget_tokens=context_budget_tokens
        ),
    )
    return await assembler.assemble(request, current)


async def tool_subset(
    assembler: Assembler,
    current: Callable[[list[str]], Sequence[str]],
    *,
    context_budget_tokens: int,
    task_iris: Sequence[str] = (),
    capabilities: Sequence[str] = (),
    text: str | None = None,
) -> tuple[list[str], Assembled]:
    """The smallest covering tool subset within the budget, or ``current``'s tools.

    Evaluate-only (§4.5): the record is returned for sampling, never committed.
    """
    evaluate_only = Assembler(assembler.graphs, assembler.tenant, commit_context=None)
    request = assembly_request(
        assembler.tenant,
        task_iris=task_iris,
        capabilities=capabilities,
        goal=text,
        budget=AssemblyBudget(
            kinds=TOOL_KINDS,
            context_budget_tokens=context_budget_tokens,
            require_tools=True,
        ),
    )
    answer = await evaluate_only.assemble(request, current)
    if answer.agent is None:
        return list(answer.fallback or []), answer
    tools = [str(t.get("component_id")) for t in answer.agent.get("tools") or []]
    return tools, answer


__all__ = ["TOOL_KINDS", "route_a2a_task", "tool_subset"]
