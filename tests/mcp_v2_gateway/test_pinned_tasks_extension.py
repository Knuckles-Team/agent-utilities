"""Offline conformance checks for the pinned experimental Tasks extension.

The source contract is modelcontextprotocol/experimental-ext-tasks at
2c1425d9a288b9b1f489430fe1e00bb392b47e48.  Its draft schema had SHA-256
b17cb4a2534379c214b17770bd5d3d54f69fde16a953bfb542c58235a61274bb when
this reduced, result-focused validator was transcribed.  Keep this test
offline: MCP Python SDK 2.0's task models describe the older wire shape.
"""

from __future__ import annotations

from typing import Any

import pytest
from jsonschema import Draft202012Validator

from mcp_v2_gateway.gateway import (
    TASKS_EXTENSION_REVISION,
    GatewayRequestContext,
    GraphOSV2Gateway,
)

_PINNED_TASKS_REVISION = "2c1425d9a288b9b1f489430fe1e00bb392b47e48"
_PINNED_SCHEMA_SHA256 = (
    "b17cb4a2534379c214b17770bd5d3d54f69fde16a953bfb542c58235a61274bb"
)

_TASK_PROPERTIES: dict[str, Any] = {
    "taskId": {"type": "string"},
    "status": {
        "enum": ["working", "input_required", "completed", "failed", "cancelled"]
    },
    "statusMessage": {"type": "string"},
    "createdAt": {"type": "string"},
    "lastUpdatedAt": {"type": "string"},
    "ttlMs": {"type": ["number", "null"]},
    "pollIntervalMs": {"type": "number"},
}

_TASK_REQUIRED = ["taskId", "status", "createdAt", "lastUpdatedAt", "ttlMs"]
_CREATE_TASK_SCHEMA = {
    "type": "object",
    "properties": {**_TASK_PROPERTIES, "resultType": {"const": "task"}},
    "required": [*_TASK_REQUIRED, "resultType"],
    "additionalProperties": False,
}
_GET_TASK_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {**_TASK_PROPERTIES, "resultType": {"const": "complete"}},
            "required": [*_TASK_REQUIRED, "resultType"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                **_TASK_PROPERTIES,
                "resultType": {"const": "complete"},
                "result": {"type": "object"},
            },
            "required": [*_TASK_REQUIRED, "resultType", "result"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                **_TASK_PROPERTIES,
                "resultType": {"const": "complete"},
                "error": {"type": "object"},
            },
            "required": [*_TASK_REQUIRED, "resultType", "error"],
            "additionalProperties": False,
        },
    ]
}
_UPDATE_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "jsonrpc": {"const": "2.0"},
        "id": {"type": ["string", "integer"]},
        "method": {"const": "tasks/update"},
        "params": {
            "type": "object",
            "properties": {
                "taskId": {"type": "string"},
                "inputResponses": {"type": "object"},
            },
            "required": ["taskId", "inputResponses"],
            "additionalProperties": False,
        },
    },
    "required": ["jsonrpc", "id", "method", "params"],
    "additionalProperties": False,
}


class _TasksDownstream:
    def __init__(self, status: str) -> None:
        self.status = status

    async def list_tools(self, _context: GatewayRequestContext) -> dict[str, object]:
        return {
            "tools": [
                {
                    "name": "graph_jobs",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ]
        }

    async def call_tool(
        self,
        _name: str,
        arguments: dict[str, object],
        _context: GatewayRequestContext,
    ) -> dict[str, object]:
        if arguments["action"] == "status":
            return {
                "status": self.status,
                "created_at": 0,
                "updated_at": 1,
                "result": {"content": [{"type": "text", "text": "done"}]},
            }
        if arguments["action"] == "cancel":
            return {"status": "cancelled"}
        raise AssertionError(arguments)


def _meta() -> dict[str, object]:
    return {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {
            "extensions": {"io.modelcontextprotocol/tasks": {}}
        },
    }


def _request(method: str, params: dict[str, object]) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}


@pytest.mark.asyncio
async def test_tasks_results_conform_to_the_pinned_offline_extension_contract() -> None:
    assert TASKS_EXTENSION_REVISION == _PINNED_TASKS_REVISION
    assert len(_PINNED_SCHEMA_SHA256) == 64
    context = GatewayRequestContext(authorization="Bearer synthetic-token")

    for downstream_status in ("queued", "completed", "failed", "cancelled"):
        gateway = GraphOSV2Gateway(_TasksDownstream(downstream_status))
        polled = await gateway.dispatch(
            _request("tasks/get", {"taskId": "job:opaque", "_meta": _meta()}),
            context=context,
        )
        Draft202012Validator(_GET_TASK_SCHEMA).validate(polled["result"])
        assert polled["result"]["ttlMs"] is None

    created = GraphOSV2Gateway(_TasksDownstream("queued"))._task_from_status(
        "job:opaque",
        {"status": "queued", "created_at": 0, "updated_at": 1},
        result_type="task",
    )
    Draft202012Validator(_CREATE_TASK_SCHEMA).validate(created)


@pytest.mark.asyncio
async def test_tasks_update_uses_the_pinned_input_responses_object_shape() -> None:
    context = GatewayRequestContext(authorization="Bearer synthetic-token")
    request = _request(
        "tasks/update",
        {
            "taskId": "job:opaque",
            "inputResponses": {"approval": {"action": "accept"}},
            "_meta": _meta(),
        },
    )
    params = request["params"]
    assert isinstance(params, dict)
    extension_request = {
        **request,
        "params": {key: value for key, value in params.items() if key != "_meta"},
    }
    Draft202012Validator(_UPDATE_REQUEST_SCHEMA).validate(extension_request)

    result = await GraphOSV2Gateway(_TasksDownstream("queued")).dispatch(
        request,
        context=context,
    )
    assert result["result"] == {"resultType": "complete"}
