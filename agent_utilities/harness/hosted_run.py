"""Public host composition seam for a client-only coding run."""

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from agent_utilities.graph import RunRequest
from agent_utilities.graph.executor import iterate_run, resolve_plan
from agent_utilities.host_ports import ClientCapabilityTransport


async def iterate_client_run(
    text: str, transport: ClientCapabilityTransport
) -> AsyncIterator[dict[str, Any]]:
    """Resolve the host plan, reject server tools, and bind the client transport."""
    request = RunRequest(task=text)
    plan = await resolve_plan(request, {})
    if plan.tool_names or plan.toolsets:
        raise PermissionError("Client runs require client-only tool bindings")
    plan = replace(plan, client_transport=transport)
    async for event in iterate_run(request, plan=plan):
        yield event
