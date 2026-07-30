"""Wire-level compatibility guard for the 2026 MCP Tasks extension.

GraphOS must not advertise the MCP Tasks extension (`io.modelcontextprotocol/tasks`,
served over `tasks/get` / `tasks/update` / `tasks/cancel` by the separate
`fastmcp-tasks` package, SEP-2663) until it is actually wired up. This exercises
real FastMCP initialization metadata and handler registration rather than a local
capability model.

CONCEPT:AU-ECO.mcp.protocol-compat-bridge — MCP SDK v2's `LowLevelServer` (the
`mcp._mcp_server` this test drives) renamed the request-handler store from a
public `request_handlers` dict keyed by request-model CLASS to a private
`_request_handlers` dict keyed by wire-protocol METHOD STRING (e.g. `"tools/list"`),
reached through the public `get_request_handler(method: str) -> HandlerEntry | None`
accessor. The behavior under test — GraphOS doesn't register the tasks-extension
handlers — still holds; only the introspection mechanism needed to update, so this
asserts the public accessor by the extension's real method names instead of poking
the private class-keyed dict.
"""

from __future__ import annotations

import pytest


def test_graphos_does_not_advertise_unimplemented_2026_tasks_extension() -> None:
    from agent_utilities.mcp.server_factory import create_mcp_server

    _args, mcp, _middlewares = create_mcp_server(
        name="graph-os-test",
        command_args=[],
    )

    capabilities = mcp._mcp_server.create_initialization_options().capabilities
    extensions = (capabilities.model_extra or {}).get("extensions", {})
    assert "io.modelcontextprotocol/tasks" not in extensions

    # The tasks extension (fastmcp-tasks, SEP-2663) registers exactly these three
    # methods when mounted; none should be reachable on a GraphOS server that never
    # mounted it.
    for method in ("tasks/get", "tasks/update", "tasks/cancel"):
        assert mcp._mcp_server.get_request_handler(method) is None


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
