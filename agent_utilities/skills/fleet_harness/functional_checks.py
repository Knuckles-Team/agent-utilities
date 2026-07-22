"""Functional (live) validation layer — a client of a reachable graph-os MCP.

This layer never starts graph-os, the engine, or any other service — it is
purely a client, exactly like `agent_utilities.skills.runtime_validation`
(the existing bundled-skill certification harness this pattern is modeled
on). Every network operation is wrapped in a short, explicit timeout so an
unreachable or hung endpoint degrades to ``SKIPPED-unreachable`` instead of
blocking the harness or reporting a false PASS.

What it actually exercises today: every skill whose body names a live
graph-os tool (a backtick ``graph_*``/``engine_*``/``ontology_*``/``object_*``
span) is checked against the MCP server's OWN advertised ``list_tools()``
result — catching the real-world failure mode of a skill documenting a tool
that was renamed or removed. A skill with no such references is not
graph-os-routed and is marked ``SKIPPED-not-applicable`` (not a failure).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

from agent_utilities.skills.fleet_harness.discovery import SkillRecord
from agent_utilities.skills.fleet_harness.static_checks import GRAPHOS_TOOL_REF_RE

FunctionalStatus = Literal[
    "PASS", "FAIL", "SKIPPED-unreachable", "SKIPPED-not-applicable"
]

#: Deployment-varying, not auto-detectable, no universal default — a real
#: reachable graph-os endpoint differs per environment (local dev, k8s
#: ingress, ...). Highest priority: an explicit SSE/HTTP URL.
_URL_ENV = "SKILL_HARNESS_GRAPH_OS_URL"
#: Fallback: a stdio command to spawn, e.g. the local console script
#: (`graph-os`) resolved from PATH — passed straight to fastmcp's `Client`,
#: which infers a stdio transport from a bare command string. Still just a
#: *client* — the process this spawns is graph-os itself acting as an MCP
#: server, not a helper the harness owns.
_COMMAND_ENV = "SKILL_HARNESS_GRAPH_OS_COMMAND"
_DEFAULT_COMMAND = "graph-os"

_CONNECT_TIMEOUT_SECONDS = float(os.environ.get("SKILL_HARNESS_CONNECT_TIMEOUT", "8"))
_CALL_TIMEOUT_SECONDS = float(os.environ.get("SKILL_HARNESS_CALL_TIMEOUT", "10"))

#: A zero-arg factory returning an async context manager yielding an MCP
#: client (what `@asynccontextmanager`-decorated `default_client` produces).
ClientFactory = Callable[[], AbstractAsyncContextManager[Any]]


@asynccontextmanager
async def default_client() -> AsyncIterator[Any]:
    """The default graph-os MCP client: URL first, then a spawned stdio command.

    Import of ``fastmcp`` is deliberately lazy — the static layer must import
    cleanly with no MCP/network deps installed.
    """
    from fastmcp import Client

    url = os.environ.get(_URL_ENV)
    if url:
        async with Client(url) as client:
            yield client
        return
    command = os.environ.get(_COMMAND_ENV, _DEFAULT_COMMAND)
    async with Client(command) as client:
        yield client


@dataclass
class FunctionalResult:
    skill: str
    status: FunctionalStatus
    detail: str
    referenced_tools: tuple[str, ...] = ()


def referenced_graphos_tools(body: str) -> list[str]:
    """Every ``graph_*``/``engine_*``/``ontology_*``/``object_*`` backtick
    reference in a skill body, in first-seen order, de-duplicated."""
    seen: dict[str, None] = {}
    for match in GRAPHOS_TOOL_REF_RE.findall(body):
        seen.setdefault(match, None)
    return list(seen)


async def probe_reachable(
    client_factory: ClientFactory = default_client,
) -> tuple[bool, frozenset[str], str]:
    """Return ``(reachable, live_tool_names, detail)``.

    Bounded by ``_CONNECT_TIMEOUT_SECONDS`` + ``_CALL_TIMEOUT_SECONDS`` total
    — this function ALWAYS returns; it never raises and never hangs.
    """
    try:
        async with asyncio.timeout(_CONNECT_TIMEOUT_SECONDS + _CALL_TIMEOUT_SECONDS):
            async with client_factory() as client:
                tools = await client.list_tools()
                names = frozenset(getattr(t, "name", None) or t["name"] for t in tools)
                return True, names, f"connected, {len(names)} live tool(s) advertised"
    except TimeoutError:
        return (
            False,
            frozenset(),
            (
                f"unreachable: no response within {_CONNECT_TIMEOUT_SECONDS + _CALL_TIMEOUT_SECONDS:.0f}s"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - reachability probe must degrade, never raise
        return False, frozenset(), f"unreachable: {type(exc).__name__}: {exc}"


async def run_functional_checks(
    records: list[SkillRecord],
    client_factory: ClientFactory = default_client,
) -> list[FunctionalResult]:
    """Run the functional layer over every discovered skill.

    One reachability probe for the whole run (not per-skill — a per-skill
    connect would multiply the unreachable case by hundreds of timeouts).
    """
    reachable, live_tools, why = await probe_reachable(client_factory)
    results: list[FunctionalResult] = []
    for record in records:
        body = record.skill_md.read_text(encoding="utf-8", errors="replace")
        tools = referenced_graphos_tools(body)
        if not tools:
            results.append(
                FunctionalResult(
                    skill=record.relative_path,
                    status="SKIPPED-not-applicable",
                    detail="no graph-os tool references in this skill's body",
                )
            )
            continue
        if not reachable:
            results.append(
                FunctionalResult(
                    skill=record.relative_path,
                    status="SKIPPED-unreachable",
                    detail=why,
                    referenced_tools=tuple(tools),
                )
            )
            continue
        missing = [t for t in tools if t not in live_tools]
        if missing:
            results.append(
                FunctionalResult(
                    skill=record.relative_path,
                    status="FAIL",
                    detail=f"tool(s) referenced but absent from the live graph-os tool surface: {missing}",
                    referenced_tools=tuple(tools),
                )
            )
        else:
            results.append(
                FunctionalResult(
                    skill=record.relative_path,
                    status="PASS",
                    detail=f"all {len(tools)} referenced tool(s) present on the live surface",
                    referenced_tools=tuple(tools),
                )
            )
    return results
