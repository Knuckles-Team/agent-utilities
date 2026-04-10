#!/usr/bin/python
"""
HSM/BT infrastructure - entry/exit hooks. invariants, preconditions, regions.
"""

from __future__ import annotations

import logging
import asyncio
import time
from typing import Any, Optional, Callable

from pydantic_ai import Agent
from ..models import MCPServerHealth
from .config_helpers import emit_graph_event

logger = logging.getLogger(__name__)

# Hook registry

_on_enter_hooks: list = []
_on_exit_hooks: list = []


def register_on_enter_hook(hook: Callable) -> None:
    """
    Register a plugin callback that fires on every specialist entry.
    Hook signature: async def hook(deps, state, agent_name, server_name) -> None
    """
    _on_enter_hooks.append(hook)


def register_on_exit_hook(hook: Callable) -> None:
    """
    Register a plugin callback that fires on every specialist exit.
    Hook signature: async def hook(deps, state, agent_name, success, server_name, duration) -> None
    """
    _on_exit_hooks.append(hook)


async def on_enter_specialist(
    ctx_deps: Any, ctx_state: Any, agent_name: str, server_name: str = ""
) -> None:
    """HSM entry action: guaranteed to fire when entering any specialist superstate."""
    ctx_state.node_history.append(agent_name)
    emit_graph_event(
        ctx_deps.event_queue,
        event_type="specialist_enter",
        agent=agent_name,
        server=server_name,
        timestamp=time.time(),
    )
    # Store entry time for duration tracking on exit
    if not hasattr(ctx_deps, "_entry_times"):
        ctx_deps._entry_times = {}
    ctx_deps._entry_times[agent_name] = time.time()
    logger.info(f"HSM Enter: {agent_name}")

    # Fire registered plugin hooks
    for hook in _on_enter_hooks:
        try:
            await hook(ctx_deps, ctx_state, agent_name, server_name)
        except Exception as e:
            logger.warning(f"Entry hook error: {e}")


async def on_exit_specialist(
    ctx_deps: Any, ctx_state: Any, agent_name: str, success: bool, server_name: str = ""
) -> None:
    """HSM exit action: guaranteed to fire when exiting specialist superstate."""
    entry_time = getattr(ctx_deps, "_entry_times", {}).get(agent_name, time.time())
    duration = time.time() - entry_time

    emit_graph_event(
        ctx_deps.event_queue,
        event_type="specialist_exit",
        agent=agent_name,
        server=server_name,
        success=success,
        duration_ms=round(duration * 1000),
    )

    # Update circuit breaker
    if server_name:
        health = ctx_deps.server_health.get(server_name)
        if not health:
            health = MCPServerHealth(server_name=server_name)
            ctx_deps.server_health[server_name] = health
        if success:
            health.record_success()
        else:
            health.record_failure()

    logger.info(
        f"HSM Exit: {agent_name} ({'OK' if success else 'FAIL'})"
        f"duration={duration:.2f}s"
    )

    # Fire registered plugin hooks
    for hook in _on_exit_hooks:
        try:
            await hook(ctx_deps, ctx_state, agent_name, success, server_name, duration)
        except Exception as e:
            logger.warning(f"Exit hook error: {e}")


async def run_orthogonal_regions(
    agent: Agent,
    queries: list[str],
    deps: Any = None,
    timeout: float = 120.0,
    event_queue: Optional[asyncio.Queue] = None,
    agent_name: str = "",
) -> dict[str, str]:
    """Execute multiple independent sub-tasks concurrently within a single specialist

    This implements HSM orthogonal regions: multiple independent sub-state-machines
    running in parllel within one superstate. Each region runs the same agent with
    a different query, and results are merged.

    Args:
        agent: The specialist Pydantic AI Agent.
        queries: List of sub-task queries to run concurrently.
        deps: Dependencies to pass to agent.run().
        timeout: Per-region timeout in seconds.
        event_queue: Optional queue for streaming events.
        agent_name: Name for event tagging.

    Returns:
        Dict mapping query->result for each region.
    """

    if len(queries) <= 1:
        # Single query - no need for orthogonal regions
        res = await asyncio.wait_for(agent.run(queries[0], deps=deps), timeout=timeout)
        return {queries[0]: str(res.output)}

    emit_graph_event(
        event_queue,
        event_type="orthogonal_regions_start",
        agent=agent_name,
        region_count=len(queries),
    )

    async def run_region(query: str, region_id: int) -> tuple[str, str]:
        try:
            emit_graph_event(
                event_queue,
                event_type="region_start",
                agent=agent_name,
                region_id=region_id,
                query=query[:200],
            )
            res = await asyncio.wait_for(agent.run(query, deps=deps), timeout=timeout)
            emit_graph_event(
                event_queue,
                event_type="region_complete",
                agent=agent_name,
                region_id=region_id,
                success=True,
            )
            return (query, str(res.output))
        except Exception as e:
            logger.warning(f"Region {region_id} failed: {e}")
            emit_graph_event(
                event_queue,
                event_type="region_complete",
                agent=agent_name,
                region_id=region_id,
                success=False,
                error=str(e),
            )
            return (query, f"Error: {e}")

    # Run all regions concurrently
    tasks = [run_region(q, i) for i, q in enumerate(queries)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    merged = {}

    for result in results:
        if isinstance(result, tuple):
            merged[result[0]] = result[1]
        elif isinstance(result, Exception):
            merged[f"error_{id(result)}"] = str(result)

    emit_graph_event(
        event_queue,
        event_type="orthogonal_regions_complete",
        agent=agent_name,
        region_count=len(queries),
        success_count=sum(1 for v in merged.values() if not v.startswith("Error:")),
    )

    return merged


class StateInvariantError(Exception):
    """Raised when a graph state invariant is violated at a transition boundary"""

    pass


def assert_state_valid(state: Any, transition: str) -> None:
    """HSM state invariant assertions - validate at every transition boundary.
    Catches corruption early: empty queries, cursor overflows, infinite loops
    """

    if not state.query:
        raise StateInvariantError(f"Empty query at {transition}")
    if state.step_cursor < 0:
        raise StateInvariantError(
            f"Negative step cursor ({state.step_cursor}) at {transition}"
        )
    if state.global_research_loops > 5:
        raise StateInvariantError(
            f"Infinite re-plan loop research loops ({state.global_research_loops}) at {transition}"
        )
    if state.verification_attempts > 3:
        raise StateInvariantError(
            f"Verification loop ({state.verification_attempts}) at {transition}"
        )
    if state.plan and state.plan.steps:
        if state.step_cursor > len(state.plan.steps) + 1:
            raise StateInvariantError(
                f"Step cursor overflow ({state.step_cursor}/{len(state.plan.steps)}) at {transition}"
            )


class NodeResult:
    """Behavior Tree tri-state result for graph node execution.

    Every specialist node implicitly returns one of these states:
        - SUCCESS: Node completed its task. Transition to next.
        - FAILURE: Node failed. Trigger fallback/recovery.
        - RUNNING: Node is paused (e.g. waiting for human approval)
    """

    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


def check_specialist_preconditions(
    agent_info: Any,
    deps: Any,
) -> tuple[bool, str]:
    """Behavior Tree precondition check before entering a specialist superstate.

    Validates that the specialist has the required resources to execute:
    - MCP server is available (circuit breaker not open)
    - At least once MCP toolset is bound for this server

    Returns:
        (can_proceed, reason) - True if preconditions met, False with explanation if not
    """

    server_name = getattr(agent_info, "mcp_server", "")

    # Check circuit breaker
    if server_name and hasattr(agent_info, "server_health"):
        health = deps.server_health.get(server_name)
        if health and not health.is_available():
            return (
                False,
                f"Server '{server_name}' circuit is OPEN ({health.failures} failures)",
            )

        # Check that at least one toolset matches this server
        if server_name and hasattr(deps, "mcp_toolsets"):
            has_tools = any(
                getattr(ts, "id", getattr(ts, "name", None)) == server_name
                for ts in deps.mcp_toolsets
            )
            if not has_tools:
                return (False, f"No MCP toolset bound for server '{server_name}'")

        return (True, "")

    # Default to true if no specific guards triggered
    return (True, "")


def static_route_query(query: str, available_specialists: dict[str, str]) -> str | None:
    """HSM junction pseudostate: static keyword-based routing before LLM
    Returns specialist node_id if a strong keyword match is found, else None.
    This saves an LLM call for obvious queries like "list gitlab projects".
    """
    query_lower = query.lower()
    # Build keyword->specialist index from specialist descriptions
    for node_id, description in available_specialists.items():
        # Check if the node_id itself appears as a word in the query
        tag_clean = node_id.replace("_", " ").replace("-", " ")
        for keyword in tag_clean.split():
            if len(keyword) > 3 and keyword in query_lower:
                return node_id
    return None
