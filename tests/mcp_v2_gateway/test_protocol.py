"""Conformance-style JSON-RPC checks for the isolated MCP v2 gateway."""

from __future__ import annotations

import pytest

from mcp_v2_gateway.gateway import (
    MCP_V2_PROTOCOL_VERSION,
    TASKS_EXTENSION,
    GraphOSClient,
    GraphOSV2Gateway,
)


class RecordingGraphOS(GraphOSClient):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], str]] = []
        self.status = "queued"

    async def list_tools(self, authorization: str) -> dict[str, object]:
        self.calls.append(("tools/list", {}, authorization))
        return {"tools": [{"name": "graph_jobs"}]}

    async def call_tool(
        self, name: str, arguments: dict[str, object], authorization: str
    ) -> dict[str, object]:
        self.calls.append((name, arguments, authorization))
        if name != "graph_jobs":
            return {"content": [{"type": "text", "text": "ok"}], "isError": False}
        if arguments["action"] == "dispatch":
            return {"job_id": "job:opaque-work-item"}
        if arguments["action"] == "status":
            return {"status": self.status, "created_at": "2026-07-30T00:00:00Z"}
        if arguments["action"] == "cancel":
            self.status = "cancelled"
            return {"status": "cancelled"}
        raise AssertionError(arguments)


def _meta(*, tasks: bool = False) -> dict[str, object]:
    capabilities: dict[str, object] = {}
    if tasks:
        capabilities["extensions"] = {TASKS_EXTENSION: {}}
    return {
        "io.modelcontextprotocol/protocolVersion": MCP_V2_PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientCapabilities": capabilities,
    }


def _request(
    method: str, params: dict[str, object], request_id: int = 1
) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


@pytest.mark.asyncio
async def test_discovery_is_stateless_but_authorization_filtered() -> None:
    downstream = RecordingGraphOS()
    gateway = GraphOSV2Gateway(downstream, clock=lambda: 0)

    response = await gateway.dispatch(
        _request("server/discover", {"_meta": _meta()}),
        authorization="Bearer tenant-token",
    )

    result = response["result"]
    assert result["supportedVersions"] == [MCP_V2_PROTOCOL_VERSION]
    assert result["capabilities"]["extensions"] == {TASKS_EXTENSION: {}}
    assert result["tools"] == [{"name": "graph_jobs"}]
    assert downstream.calls == [("tools/list", {}, "Bearer tenant-token")]


@pytest.mark.asyncio
async def test_durable_graph_jobs_dispatch_projects_one_work_item_as_a_task() -> None:
    downstream = RecordingGraphOS()
    gateway = GraphOSV2Gateway(downstream, clock=lambda: 0)
    response = await gateway.dispatch(
        _request(
            "tools/call",
            {
                "name": "graph_jobs",
                "arguments": {"action": "dispatch", "task": "write report"},
                "_meta": _meta(tasks=True),
            },
        ),
        authorization="Bearer tenant-token",
    )

    task = response["result"]
    assert task["resultType"] == "task"
    assert task["taskId"] == "job:opaque-work-item"
    assert task["status"] == "working"
    # Dispatch is followed by the same tenant-scoped status read before the
    # handle is returned, satisfying the task creation durability requirement.
    assert [call[1]["action"] for call in downstream.calls] == ["dispatch", "status"]
    assert all(call[2] == "Bearer tenant-token" for call in downstream.calls)


@pytest.mark.asyncio
async def test_task_lifecycle_uses_graph_jobs_without_a_gateway_store() -> None:
    downstream = RecordingGraphOS()
    gateway = GraphOSV2Gateway(downstream, clock=lambda: 0)
    common = {"taskId": "job:opaque-work-item", "_meta": _meta(tasks=True)}

    working = await gateway.dispatch(
        _request("tasks/get", common), authorization="Bearer tenant-token"
    )
    assert working["result"]["resultType"] == "complete"
    assert working["result"]["status"] == "working"

    updated = await gateway.dispatch(
        _request("tasks/update", common | {"inputResponses": {}}),
        authorization="Bearer tenant-token",
    )
    assert updated["result"] == {"resultType": "complete"}

    cancelled = await gateway.dispatch(
        _request("tasks/cancel", common), authorization="Bearer tenant-token"
    )
    assert cancelled["result"] == {"resultType": "complete"}
    observed = await gateway.dispatch(
        _request("tasks/get", common), authorization="Bearer tenant-token"
    )
    assert observed["result"]["status"] == "cancelled"
    assert all(call[0] == "graph_jobs" for call in downstream.calls)


@pytest.mark.asyncio
async def test_tasks_require_current_request_capability_and_bearer() -> None:
    gateway = GraphOSV2Gateway(RecordingGraphOS())
    missing_capability = await gateway.dispatch(
        _request("tasks/get", {"taskId": "job:opaque", "_meta": _meta()}),
        authorization="Bearer tenant-token",
    )
    assert missing_capability["error"]["code"] == -32021
    assert missing_capability["error"]["data"]["requiredCapabilities"] == {
        "extensions": {TASKS_EXTENSION: {}}
    }

    missing_bearer = await gateway.dispatch(
        _request("tools/list", {"_meta": _meta()}), authorization=None
    )
    assert missing_bearer["error"]["code"] == -32001


@pytest.mark.asyncio
async def test_non_task_tool_remains_a_complete_graphos_result() -> None:
    gateway = GraphOSV2Gateway(RecordingGraphOS())
    response = await gateway.dispatch(
        _request(
            "tools/call", {"name": "graph_query", "arguments": {}, "_meta": _meta()}
        ),
        authorization="Bearer tenant-token",
    )
    assert response["result"] == {
        "content": [{"type": "text", "text": "ok"}],
        "isError": False,
        "resultType": "complete",
    }


def test_graphos_json_text_result_is_decoded_only_for_workitem_projection() -> None:
    assert GraphOSV2Gateway._tool_object(  # noqa: SLF001 - wire adapter seam
        {"content": [{"type": "text", "text": '{"job_id":"job:opaque"}'}]}
    ) == {"job_id": "job:opaque"}
