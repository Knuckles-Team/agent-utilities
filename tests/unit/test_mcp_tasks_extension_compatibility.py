"""Wire-level compatibility guard for the 2026 MCP Tasks extension.

FastMCP 3.4.5 currently carries the 2025-11-25 experimental Tasks types, so
GraphOS must not advertise the newer extension until it can register all of its
handlers.  This exercises the real FastMCP initialization metadata, rather than
testing a local capability model.
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

    handlers = mcp._mcp_server.request_handlers
    assert not any(
        request_type.__name__ in {"TasksGetRequest", "TasksUpdateRequest"}
        for request_type in handlers
    )


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
