"""Wire-level compatibility guard for the 2026 MCP Tasks extension.

GraphOS mounts a native, WorkItem-backed `io.modelcontextprotocol/tasks`
extension (`agent_utilities/mcp/tasks_extension.py`,
CONCEPT:AU-ECO.mcp.tasks-workitem-bridge) -- NOT the separate `fastmcp-tasks`
package (SEP-2663), whose execution engine is hard-wired to Docket/Redis, a
second job system this codebase's one `WorkItem` state machine (AU-P1-1)
forbids duplicating. This exercises real FastMCP initialization metadata and
handler registration rather than a local capability model.

Until 2026-07-31 this test asserted the OPPOSITE (GraphOS must NOT advertise
the extension until wired up) -- see `docs/architecture/mcp_v2_gateway.md`
and the isolated `mcp_v2_gateway` sidecar's own Tasks↔WorkItem mapping, which
landed first. Left passing unchanged, that negative assertion would have hidden
the feature actually shipping (a Wire-First violation), so it is updated here
to the real, current contract instead.

CONCEPT:AU-ECO.mcp.protocol-compat-bridge — MCP SDK v2's `LowLevelServer` (the
`mcp._mcp_server` this test drives) renamed the request-handler store from a
public `request_handlers` dict keyed by request-model CLASS to a private
`_request_handlers` dict keyed by wire-protocol METHOD STRING (e.g. `"tools/list"`),
reached through the public `get_request_handler(method: str) -> HandlerEntry | None`
accessor.
"""

from __future__ import annotations

import pytest


def test_graphos_advertises_and_serves_the_workitem_backed_tasks_extension() -> None:
    from agent_utilities.mcp.server_factory import create_mcp_server
    from agent_utilities.mcp.tasks_extension import (
        TASKS_EXTENSION_ID,
        WorkItemTasksExtension,
    )

    _args, mcp, _middlewares = create_mcp_server(
        name="graph-os-test",
        command_args=[],
    )

    capabilities = mcp._mcp_server.create_initialization_options().capabilities
    assert TASKS_EXTENSION_ID in (capabilities.extensions or {})

    # The tasks extension registers exactly these three methods when mounted,
    # each bound to WorkItemTasksExtension's own handler -- reachable, and
    # backed by the native WorkItem authority, not fastmcp-tasks' Docket engine.
    for method in ("tasks/get", "tasks/update", "tasks/cancel"):
        entry = mcp._mcp_server.get_request_handler(method)
        assert entry is not None
        # The bound params type is this module's own wire model, not
        # fastmcp_tasks' -- proof it's WorkItemTasksExtension that answered,
        # not some other tasks-extension implementation.
        assert entry.params_type.__module__ == WorkItemTasksExtension.__module__


@pytest.mark.asyncio
async def test_graph_jobs_cancel_uses_the_dispatched_work_item(monkeypatch) -> None:
    import json

    import agent_utilities.mcp.kg_server as kg
    from agent_utilities.mcp.tools.job_tools import register_job_tools
    from agent_utilities.orchestration.work_item import orchestrator_work_item_id

    class _MCP:
        def tool(self, **_kwargs):
            return lambda function: function

    job_id = "job:mcp-task-cancel"
    item_id = orchestrator_work_item_id(job_id)

    class _Authority:
        def __init__(self) -> None:
            self.node = {
                "id": item_id,
                "tenant": "tenant-test",
                "status": "ready",
                "depends_on": [],
                "downstream_ids": [],
                "metadata": {},
            }
            self.cancel_requests: list[dict] = []

        def query_cypher(self, _query: str, params: dict | None = None):
            return [self.node] if (params or {}).get("id") == item_id else []

        def cancel_work_item(self, request: dict):
            self.cancel_requests.append(request)
            self.node["status"] = "cancelled"
            return {"status": "cancelled"}

    authority = _Authority()

    class _Engine:
        _work_item_engine = authority

    engine = _Engine()
    register_job_tools(_MCP())
    monkeypatch.setattr(kg, "_get_engine", lambda: engine)

    cancelled = json.loads(
        await kg._execute_tool("graph_jobs", action="cancel", job_id=job_id)
    )

    assert cancelled == {"status": "cancelled", "job_id": job_id}
    assert authority.node["status"] == "cancelled"
    assert authority.cancel_requests[0]["work_item_id"] == item_id
