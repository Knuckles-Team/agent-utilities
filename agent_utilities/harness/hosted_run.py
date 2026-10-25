"""Public host composition seam for a client-only coding run."""

from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import replace
from typing import Any

from agent_utilities.graph import RunRequest
from agent_utilities.graph.executor import (
    ModelStreamObserver,
    iterate_run,
    resolve_plan,
)
from agent_utilities.host_ports import ClientCapabilityTransport


async def iterate_client_run(
    request: RunRequest,
    transport: ClientCapabilityTransport,
    *,
    stream_observer: ModelStreamObserver | None = None,
) -> AsyncGenerator[dict[str, Any]]:
    """Preserve the admitted request while binding a client-only received plan."""
    if not isinstance(request, RunRequest):
        raise TypeError("client runs require an admitted RunRequest")
    plan = await resolve_plan(request, {})
    if plan.tool_names or plan.toolsets:
        raise PermissionError("Client runs require client-only tool bindings")
    plan = replace(plan, client_transport=transport)
    async with aclosing(
        iterate_run(request, plan=plan, stream_observer=stream_observer)
    ) as execution:
        async for event in execution:
            yield event
