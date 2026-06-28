"""Synthesis → live execution bridge (CONCEPT:KG-2.10).

Wires the KG-driven synthesis layer to the production orchestration:
* :func:`make_capability_search` backs ``synthesize_*`` with the real
  ``CapabilityIndex.designate`` (Plan-08 retrieval) instead of a demo embed-store.
* :func:`execute_agent_spec` / :func:`execute_team_spec` run synthesized agents/
  teams through ``orchestration.agent_runner.run_agent`` (the
  ``graph_orchestrate(action='execute_agent')`` entry point) against the live LLM.
* :func:`persist_as_runnable` writes a synthesized agent as a resolvable KG node
  so ``run_agent`` can find it with its tools.

The runner + facade + embed_fn are injectable so this is unit-testable without a
live model or daemon; the defaults use the real components.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .models import EnrichmentEdge, ExtractionBatch, GraphNode
from .orchestration import AgentSpec, TeamSpec
from .registry import write_batch

logger = logging.getLogger(__name__)

# (query, k) -> list of capability candidates {id,type,name,score,capabilities}
CapabilitySearchFn = Callable[[str, int], list[dict[str, Any]]]
# (agent_name, task) -> result string
RunnerFn = Callable[..., Awaitable[str]]


def make_capability_search(
    facade: Any, embed_fn: Callable[[list[str]], list[list[float]]], default_k: int = 8
) -> CapabilitySearchFn:
    """Production capability search: embed the query, then KG ``designate()``.

    Returns the standard candidate shape ``synthesize_*`` expects. Entity ``type``/
    ``name`` are parsed from the id prefix (``tool:foo`` → Tool/foo).
    """

    def search(query: str, k: int = default_k) -> list[dict[str, Any]]:
        vec = embed_fn([query])[0]
        designations = facade.designate(vec, k=k or default_k)
        out: list[dict[str, Any]] = []
        for d in designations:
            nid = getattr(d, "id", None) or (
                d.get("id") if isinstance(d, dict) else None
            )
            if not nid:
                continue
            prefix, _, rest = str(nid).partition(":")
            out.append(
                {
                    "id": nid,
                    "type": prefix.capitalize() if rest else "",
                    "name": rest or nid,
                    "score": float(getattr(d, "score", 0.0) or 0.0),
                    "capabilities": sorted(getattr(d, "capabilities", []) or []),
                }
            )
        return out

    return search


def persist_as_runnable(backend: Any, spec: AgentSpec) -> tuple[int, int]:
    """Write a synthesized agent as a CallableResource node ``run_agent`` resolves.

    ``run_agent`` matches CallableResource nodes by name (resource_type
    AGENT_SKILL) and binds their tools — so a synthesized agent becomes runnable.
    """
    nodes = [
        GraphNode(
            id=f"resource:{spec.id}",
            type="CallableResource",
            props={
                "name": spec.name,
                "resource_type": "AGENT_SKILL",
                "description": spec.description or spec.goal,
                "system_prompt": spec.system_prompt[:8000],
            },
        )
    ]
    edges = [
        EnrichmentEdge(
            source=f"resource:{spec.id}", target=f"tool:{t}", rel_type="USES_TOOL"
        )
        for t in spec.tools
    ]
    return write_batch(
        backend, ExtractionBatch(category="orchestration", nodes=nodes, edges=edges)
    )


def persist_skill_as_runnable(
    backend: Any,
    *,
    skill_id: str,
    name: str,
    system_prompt: str,
    description: str = "",
    tools: list[str] | None = None,
) -> tuple[int, int]:
    """Bind an ingested ``:Skill`` node into a runnable ``CallableResource``.

    CONCEPT:ORCH-1.96 — the dispatch half of skill ingestion. An ingested atomic
    skill (whether a bare ``:Skill`` node written by ``skill_workflow_ingest`` or
    an ``AGENT_SKILL`` CallableResource that only carries a ``skill_code_path``) is
    *search corpus* until something makes it executable. This reuses the
    :func:`persist_as_runnable` shape — it upserts the SAME ``CallableResource``
    node id ``run_agent`` resolves (resource_type ``AGENT_SKILL``), now carrying the
    skill's instruction body as ``system_prompt`` plus ``USES_TOOL`` edges to any
    declared tools — so "pick an ingested skill, run it on the local LLM" has a
    live, idempotent path. Writing onto the same node id keeps it a single object
    (No-Legacy): the skill *becomes* its runnable resource, it is not duplicated.
    """
    nodes = [
        GraphNode(
            id=skill_id,
            type="CallableResource",
            props={
                "name": name,
                "resource_type": "AGENT_SKILL",
                "description": description or name,
                "system_prompt": (system_prompt or "")[:8000],
                "runnable_bound": True,
            },
        )
    ]
    edges = [
        EnrichmentEdge(source=skill_id, target=f"tool:{t}", rel_type="USES_TOOL")
        for t in (tools or [])
        if t
    ]
    return write_batch(
        backend, ExtractionBatch(category="orchestration", nodes=nodes, edges=edges)
    )


def _runner() -> RunnerFn:
    from agent_utilities.orchestration.agent_runner import run_agent

    return run_agent


async def execute_agent_spec(
    spec: AgentSpec,
    task: str,
    runner: RunnerFn | None = None,
    max_steps: int = 30,
) -> str:
    """Execute a synthesized agent via the live KG-to-LLM runner.

    Brackets execution with the background throttle's foreground flag so the KG
    daemons yield the GPU to this interactive run. (CONCEPT:KG-2.7/2.10)
    """
    run = runner or _runner()
    logger.info("[KG-2.10] executing synthesized agent %s", spec.name)
    try:
        from agent_utilities.core.background_throttle import get_throttle

        with get_throttle().foreground():
            return await run(spec.name, task, max_steps=max_steps)
    except ImportError:  # pragma: no cover
        return await run(spec.name, task, max_steps=max_steps)


async def execute_team_spec(
    team: TeamSpec,
    members: list[AgentSpec],
    task: str,
    runner: RunnerFn | None = None,
    concurrent: bool = True,
) -> dict[str, str]:
    """Execute a synthesized team: each member runs its part; results collected.

    Members run on their own goal (or the team task). Concurrent by default; the
    lead's result is keyed under its name. Returns {member_name: result}.
    """
    run = runner or _runner()
    targets = members or []

    async def _one(m: AgentSpec) -> tuple[str, str]:
        member_task = f"{task}\n\nYour role: {m.goal or m.description or m.name}"
        try:
            res = await run(m.name, member_task)
        except Exception as e:  # pragma: no cover - live runner failure
            res = f"ERROR: {e}"
        return m.name, res

    if concurrent:
        pairs = await asyncio.gather(*[_one(m) for m in targets])
    else:
        pairs = [await _one(m) for m in targets]
    return dict(pairs)
