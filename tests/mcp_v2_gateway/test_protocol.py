"""Conformance-style JSON-RPC checks for the isolated MCP v2 gateway."""

from __future__ import annotations

import asyncio
import json as jsonlib
from collections.abc import Mapping

import anyio
import httpx
import pytest

from mcp_v2_gateway.gateway import (
    MCP_V2_PROTOCOL_VERSION,
    MISSING_REQUIRED_EXTENSION_CAPABILITY,
    TASKS_EXTENSION,
    GatewayProtocolError,
    GatewayRequestContext,
    GraphOSClient,
    GraphOSV2Gateway,
    StreamableHTTPGateway,
    StreamableHTTPGraphOSClient,
)


class RecordingGraphOS(GraphOSClient):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object], GatewayRequestContext]] = []
        self.status = "queued"
        self.tools: list[dict[str, object]] = [
            {
                "name": "graph_jobs",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "task": {"type": "string"},
                        "job_id": {"type": "string"},
                    },
                },
            },
            {
                "name": "graph_query",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    async def list_tools(self, context: GatewayRequestContext) -> dict[str, object]:
        self.calls.append(("tools/list", {}, context))
        return {"tools": self.tools}

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
        context: GatewayRequestContext,
    ) -> dict[str, object]:
        self.calls.append((name, arguments, context))
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


def _meta(*, tasks: bool = False, **extra: object) -> dict[str, object]:
    capabilities: dict[str, object] = {}
    if tasks:
        capabilities["extensions"] = {TASKS_EXTENSION: {}}
    return {
        "io.modelcontextprotocol/protocolVersion": MCP_V2_PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientCapabilities": capabilities,
        **extra,
    }


def _request(
    method: str, params: Mapping[str, object], request_id: int = 1
) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": dict(params),
    }


def _context(
    *,
    headers: dict[str, str] | None = None,
    authorization: str = "Bearer tenant-token",
) -> GatewayRequestContext:
    return GatewayRequestContext(
        authorization=authorization,
        headers=headers or {},
    )


def _http_headers(
    method: str,
    *,
    name: str | None = None,
    origin: str | None = None,
    extra: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    headers = [
        ("Content-Type", "application/json"),
        ("Accept", "application/json, text/event-stream"),
        ("Authorization", "Bearer tenant-token"),
        ("MCP-Protocol-Version", MCP_V2_PROTOCOL_VERSION),
        ("Mcp-Method", method),
    ]
    if name is not None:
        headers.append(("Mcp-Name", name))
    if origin is not None:
        headers.append(("Origin", origin))
    if extra:
        headers.extend(extra.items())
    return headers


class StatefulLegacyHTTPClient:
    """Synthetic legacy endpoint with session-scoped dynamic visibility."""

    def __init__(
        self,
        *,
        expose_graph_jobs: bool = True,
        yield_between_requests: bool = False,
        **_kwargs: object,
    ) -> None:
        self.expose_graph_jobs = expose_graph_jobs
        self.yield_between_requests = yield_between_requests
        self.posts: list[tuple[dict[str, str], dict[str, object]]] = []
        self.deletes: list[dict[str, str]] = []
        self.session_count = 0
        self.graph_jobs_loaded: set[str] = set()
        self.auto_unload: set[str] = set()
        self.session_headers: dict[str, dict[str, str]] = {}

    async def __aenter__(self) -> StatefulLegacyHTTPClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
    ) -> httpx.Response:
        if self.yield_between_requests:
            await asyncio.sleep(0)
        self.posts.append((dict(headers), json))
        method = json["method"]
        if method == "notifications/initialized":
            return httpx.Response(202, request=httpx.Request("POST", url))

        request_id = json["id"]
        if method == "initialize":
            self.session_count += 1
            session_id = f"synthetic-session-{self.session_count}"
            self.session_headers[session_id] = dict(headers)
            result: dict[str, object] = {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
            }
        else:
            session_id = headers["Mcp-Session-Id"]
            params = json.get("params")
            assert isinstance(params, dict)
            if method == "tools/list":
                tools: list[dict[str, object]] = [{"name": "graph_query"}]
                if session_id in self.graph_jobs_loaded:
                    tools.append(
                        {
                            "name": "graph_jobs",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string"},
                                    "task": {"type": "string"},
                                    "job_id": {"type": "string"},
                                },
                            },
                        }
                    )
                result = {"tools": tools}
            else:
                name = params.get("name")
                arguments = params.get("arguments")
                assert isinstance(arguments, dict)
                if name == "load_tools":
                    assert arguments == {
                        "tools": ["graph_jobs"],
                        "auto_unload": True,
                    }
                    if self.expose_graph_jobs:
                        self.graph_jobs_loaded.add(session_id)
                        self.auto_unload.add(session_id)
                    result = {
                        "content": [{"type": "text", "text": "{}"}],
                        "isError": False,
                    }
                elif name == "unload_tools":
                    assert arguments == {"tools": ["graph_jobs"]}
                    self.graph_jobs_loaded.discard(session_id)
                    self.auto_unload.discard(session_id)
                    result = {
                        "content": [{"type": "text", "text": "{}"}],
                        "isError": False,
                    }
                elif name == "graph_jobs":
                    assert session_id in self.graph_jobs_loaded
                    action = arguments.get("action")
                    if action == "dispatch":
                        value: dict[str, object] = {"job_id": "job:opaque-work-item"}
                    elif action == "status":
                        value = {
                            "status": "queued",
                            "created_at": "2026-07-30T00:00:00Z",
                        }
                    elif action == "cancel":
                        value = {"status": "cancelled"}
                    else:
                        raise AssertionError(arguments)
                    result = {
                        "content": [
                            {
                                "type": "text",
                                "text": jsonlib.dumps(value, separators=(",", ":")),
                            }
                        ],
                        "isError": False,
                    }
                    if session_id in self.auto_unload:
                        self.graph_jobs_loaded.discard(session_id)
                        self.auto_unload.discard(session_id)
                else:
                    result = {
                        "content": [{"type": "text", "text": "ok"}],
                        "isError": False,
                    }
        frames = "\n".join(
            [
                'data: {"jsonrpc":"2.0","method":"notifications/progress"}',
                f'data: {{"jsonrpc":"2.0","id":"{request_id}",'
                f'"result":{jsonlib.dumps(result)}}}',
                "",
            ]
        )
        response_headers = {"Content-Type": "text/event-stream"}
        if method == "initialize":
            response_headers["Mcp-Session-Id"] = session_id
        return httpx.Response(
            200,
            headers=response_headers,
            text=frames,
            request=httpx.Request("POST", url),
        )

    async def delete(self, url: str, *, headers: dict[str, str]) -> httpx.Response:
        if self.yield_between_requests:
            await asyncio.sleep(0)
        self.deletes.append(dict(headers))
        session_id = headers["Mcp-Session-Id"]
        self.graph_jobs_loaded.discard(session_id)
        self.auto_unload.discard(session_id)
        return httpx.Response(200, request=httpx.Request("DELETE", url))


class CancellingLegacyHTTPClient(StatefulLegacyHTTPClient):
    """Hold a live call or unload so cancellation can hit either window."""

    def __init__(
        self,
        *,
        block_unload: bool = False,
        block_delete: bool = False,
    ) -> None:
        super().__init__()
        self.block_unload = block_unload
        self.block_delete = block_delete
        self.graph_jobs_call_started = anyio.Event()
        self.unload_started = anyio.Event()
        self.release_unload = anyio.Event()
        self.delete_started = anyio.Event()
        self.release_delete = anyio.Event()
        self.unload_attempts = 0

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
    ) -> httpx.Response:
        params = json.get("params")
        tool_name = params.get("name") if isinstance(params, dict) else None
        if json.get("method") == "tools/call" and tool_name == "graph_jobs":
            self.graph_jobs_call_started.set()
            await anyio.sleep_forever()
        if json.get("method") == "tools/call" and tool_name == "unload_tools":
            self.unload_attempts += 1
            self.unload_started.set()
            if self.block_unload:
                await self.release_unload.wait()
        return await super().post(url, headers=headers, json=json)

    async def delete(self, url: str, *, headers: dict[str, str]) -> httpx.Response:
        self.delete_started.set()
        if self.block_delete:
            await self.release_delete.wait()
        return await super().delete(url, headers=headers)


def _assert_every_session_step_preserves_headers(
    client: StatefulLegacyHTTPClient,
) -> None:
    expected_names = (
        "Authorization",
        "MCP-Protocol-Version",
        "mcp-param-tenant",
        "traceparent",
        "tracestate",
        "baggage",
    )
    for headers, request in client.posts:
        if request["method"] == "initialize":
            continue
        session_id = headers["Mcp-Session-Id"]
        initialized_headers = client.session_headers[session_id]
        assert all(
            headers[name] == initialized_headers[name] for name in expected_names
        )
    for headers in client.deletes:
        session_id = headers["Mcp-Session-Id"]
        initialized_headers = client.session_headers[session_id]
        assert all(
            headers[name] == initialized_headers[name] for name in expected_names
        )


@pytest.mark.asyncio
async def test_discovery_is_stateless_but_authorization_filtered() -> None:
    downstream = RecordingGraphOS()
    gateway = GraphOSV2Gateway(downstream, clock=lambda: 0)

    response = await gateway.dispatch(
        _request("server/discover", {"_meta": _meta()}),
        context=_context(),
    )

    result = response["result"]
    assert result["supportedVersions"] == [MCP_V2_PROTOCOL_VERSION]
    assert result["capabilities"]["extensions"] == {TASKS_EXTENSION: {}}
    assert result["capabilities"]["tools"] == {}
    assert "listChanged" not in result["capabilities"]["tools"]
    assert "tools" not in result
    assert downstream.calls[0][:2] == ("tools/list", {})
    assert downstream.calls[0][2].authorization == "Bearer tenant-token"


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
        context=_context(),
    )

    task = response["result"]
    assert task["resultType"] == "task"
    assert task["taskId"] == "job:opaque-work-item"
    assert task["status"] == "working"
    # Dispatch is followed by the same tenant-scoped status read before the
    # handle is returned, satisfying the task creation durability requirement.
    assert [call[1].get("action") for call in downstream.calls] == [
        None,
        "dispatch",
        "status",
    ]
    assert all(
        call[2].authorization == "Bearer tenant-token" for call in downstream.calls
    )


@pytest.mark.asyncio
async def test_task_lifecycle_uses_graph_jobs_without_a_gateway_store() -> None:
    downstream = RecordingGraphOS()
    gateway = GraphOSV2Gateway(downstream, clock=lambda: 0)
    common = {"taskId": "job:opaque-work-item", "_meta": _meta(tasks=True)}

    working = await gateway.dispatch(_request("tasks/get", common), context=_context())
    assert working["result"]["resultType"] == "complete"
    assert working["result"]["status"] == "working"

    updated = await gateway.dispatch(
        _request("tasks/update", common | {"inputResponses": {}}),
        context=_context(),
    )
    assert updated["result"] == {"resultType": "complete"}
    invalid_update = await gateway.dispatch(
        _request("tasks/update", common),
        context=_context(),
    )
    assert invalid_update["error"]["code"] == -32602

    cancelled = await gateway.dispatch(
        _request("tasks/cancel", common), context=_context()
    )
    assert cancelled["result"] == {"resultType": "complete"}
    observed = await gateway.dispatch(_request("tasks/get", common), context=_context())
    assert observed["result"]["status"] == "cancelled"
    assert all(call[0] == "graph_jobs" for call in downstream.calls)


@pytest.mark.asyncio
async def test_create_task_result_stays_base_task_even_if_work_finished() -> None:
    downstream = RecordingGraphOS()
    downstream.status = "completed"
    gateway = GraphOSV2Gateway(downstream, clock=lambda: 0)

    response = await gateway.dispatch(
        _request(
            "tools/call",
            {
                "name": "graph_jobs",
                "arguments": {"action": "dispatch", "task": "small task"},
                "_meta": _meta(tasks=True),
            },
        ),
        context=_context(),
    )

    task = response["result"]
    assert task["resultType"] == "task"
    assert task["status"] == "completed"
    assert "result" not in task


@pytest.mark.asyncio
async def test_task_timestamp_must_be_timezone_aware_iso8601() -> None:
    class InvalidTimestampGraphOS(RecordingGraphOS):
        async def call_tool(
            self,
            name: str,
            arguments: dict[str, object],
            context: GatewayRequestContext,
        ) -> dict[str, object]:
            result = await super().call_tool(name, arguments, context)
            if name == "graph_jobs" and arguments.get("action") == "status":
                result["created_at"] = "not-a-timestamp"
            return result

    gateway = GraphOSV2Gateway(InvalidTimestampGraphOS())
    response = await gateway.dispatch(
        _request(
            "tasks/get",
            {"taskId": "job:opaque-work-item", "_meta": _meta(tasks=True)},
        ),
        context=_context(),
    )

    assert response["error"]["code"] == -32603


@pytest.mark.asyncio
async def test_task_timestamp_normalizes_native_workitem_unix_times() -> None:
    class NumericTimestampGraphOS(RecordingGraphOS):
        async def call_tool(
            self,
            name: str,
            arguments: dict[str, object],
            context: GatewayRequestContext,
        ) -> dict[str, object]:
            result = await super().call_tool(name, arguments, context)
            if name == "graph_jobs" and arguments.get("action") == "status":
                result["created_at"] = 0.0
                result["updated_at"] = 1.0
            return result

    response = await GraphOSV2Gateway(NumericTimestampGraphOS()).dispatch(
        _request(
            "tasks/get",
            {"taskId": "job:opaque-work-item", "_meta": _meta(tasks=True)},
        ),
        context=_context(),
    )

    assert response["result"]["createdAt"] == "1970-01-01T00:00:00.000Z"
    assert response["result"]["lastUpdatedAt"] == "1970-01-01T00:00:01.000Z"


@pytest.mark.asyncio
async def test_task_timestamp_rejects_an_update_before_creation() -> None:
    class ReversedTimestampGraphOS(RecordingGraphOS):
        async def call_tool(
            self,
            name: str,
            arguments: dict[str, object],
            context: GatewayRequestContext,
        ) -> dict[str, object]:
            result = await super().call_tool(name, arguments, context)
            if name == "graph_jobs" and arguments.get("action") == "status":
                result["created_at"] = 1
                result["updated_at"] = 0
            return result

    response = await GraphOSV2Gateway(ReversedTimestampGraphOS()).dispatch(
        _request(
            "tasks/get",
            {"taskId": "job:opaque-work-item", "_meta": _meta(tasks=True)},
        ),
        context=_context(),
    )

    assert response["error"]["code"] == -32603


@pytest.mark.asyncio
async def test_task_projection_fails_closed_if_workitem_retention_becomes_bounded() -> (
    None
):
    class RetainedWorkItemGraphOS(RecordingGraphOS):
        async def call_tool(
            self,
            name: str,
            arguments: dict[str, object],
            context: GatewayRequestContext,
        ) -> dict[str, object]:
            result = await super().call_tool(name, arguments, context)
            if name == "graph_jobs" and arguments.get("action") == "status":
                result["retention_ttl_ms"] = 60_000
            return result

    response = await GraphOSV2Gateway(RetainedWorkItemGraphOS()).dispatch(
        _request(
            "tasks/get",
            {"taskId": "job:opaque-work-item", "_meta": _meta(tasks=True)},
        ),
        context=_context(),
    )

    assert response["error"]["code"] == -32603


@pytest.mark.asyncio
async def test_tasks_require_current_request_capability_and_bearer() -> None:
    gateway = GraphOSV2Gateway(RecordingGraphOS())
    missing_capability = await gateway.dispatch(
        _request("tasks/get", {"taskId": "job:opaque", "_meta": _meta()}),
        context=_context(),
    )
    assert missing_capability["error"]["code"] == MISSING_REQUIRED_EXTENSION_CAPABILITY
    assert missing_capability["error"]["data"]["requiredCapabilities"] == {
        "extensions": {TASKS_EXTENSION: {}}
    }

    missing_bearer = await gateway.dispatch(
        _request("tools/list", {"_meta": _meta()}),
        context=_context(authorization=""),
    )
    assert missing_bearer["error"]["code"] == -32001


@pytest.mark.asyncio
async def test_non_task_tool_remains_a_complete_graphos_result() -> None:
    gateway = GraphOSV2Gateway(RecordingGraphOS())
    response = await gateway.dispatch(
        _request(
            "tools/call", {"name": "graph_query", "arguments": {}, "_meta": _meta()}
        ),
        context=_context(),
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


@pytest.mark.asyncio
async def test_tools_list_adds_required_private_cache_contract() -> None:
    gateway = GraphOSV2Gateway(RecordingGraphOS())

    response = await gateway.dispatch(
        _request("tools/list", {"_meta": _meta()}),
        context=_context(),
    )

    result = response["result"]
    assert result["resultType"] == "complete"
    assert result["ttlMs"] == 60_000
    assert result["cacheScope"] == "private"
    assert [tool["name"] for tool in result["tools"]] == [
        "graph_jobs",
        "graph_query",
    ]


@pytest.mark.asyncio
async def test_transport_enforces_origin_accept_version_method_and_name() -> None:
    transport = StreamableHTTPGateway(
        GraphOSV2Gateway(RecordingGraphOS()),
        allowed_origins=["https://agent.example"],
    )
    request = _request(
        "tools/call",
        {"name": "graph_query", "arguments": {}, "_meta": _meta()},
    )
    body = jsonlib.dumps(request).encode()

    ok = await transport.handle(
        path="/mcp",
        headers=_http_headers(
            "tools/call", name="graph_query", origin="https://agent.example"
        ),
        body=body,
    )
    assert ok.status_code == 200

    bad_origin = await transport.handle(
        path="/mcp",
        headers=_http_headers(
            "tools/call", name="graph_query", origin="https://attacker.example"
        ),
        body=body,
    )
    assert bad_origin.status_code == 403

    bad_method_headers = _http_headers("tools/list", name="graph_query")
    mismatch = await transport.handle(
        path="/mcp", headers=bad_method_headers, body=body
    )
    assert mismatch.status_code == 400
    assert mismatch.body is not None
    assert mismatch.body["error"]["code"] == -32020

    missing_accept = [
        item
        for item in _http_headers("tools/call", name="graph_query")
        if item[0] != "Accept"
    ]
    rejected_accept = await transport.handle(
        path="/mcp", headers=missing_accept, body=body
    )
    assert rejected_accept.status_code == 400


@pytest.mark.asyncio
async def test_transport_uses_required_status_codes_and_rejects_notifications() -> None:
    transport = StreamableHTTPGateway(GraphOSV2Gateway(RecordingGraphOS()))
    unknown = _request("unknown/method", {"_meta": _meta()})
    response = await transport.handle(
        path="/mcp",
        headers=_http_headers("unknown/method"),
        body=jsonlib.dumps(unknown).encode(),
    )
    assert response.status_code == 404
    assert response.body is not None
    assert response.body["error"]["code"] == -32601

    unsupported = _request(
        "tools/list",
        {
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": "2099-01-01",
                "io.modelcontextprotocol/clientCapabilities": {},
            }
        },
    )
    unsupported_headers = [
        (name, "2099-01-01" if name == "MCP-Protocol-Version" else value)
        for name, value in _http_headers("tools/list")
    ]
    version_response = await transport.handle(
        path="/mcp",
        headers=unsupported_headers,
        body=jsonlib.dumps(unsupported).encode(),
    )
    assert version_response.status_code == 400
    assert version_response.body is not None
    assert version_response.body["error"]["code"] == -32022
    assert version_response.body["error"]["data"] == {
        "supported": [MCP_V2_PROTOCOL_VERSION],
        "requested": "2099-01-01",
    }

    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/cancelled",
        "params": {},
    }
    notification_response = await transport.handle(
        path="/mcp",
        headers=[("Content-Type", "application/json")],
        body=jsonlib.dumps(notification).encode(),
    )
    assert notification_response.status_code == 400
    assert notification_response.body is not None
    assert notification_response.body["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_x_mcp_header_validation_and_trace_propagation() -> None:
    downstream = RecordingGraphOS()
    downstream.tools.append(
        {
            "name": "tenant_lookup",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tenant": {
                        "type": "string",
                        "x-mcp-header": "Tenant",
                    }
                },
            },
        }
    )
    transport = StreamableHTTPGateway(GraphOSV2Gateway(downstream))
    request = _request(
        "tools/call",
        {
            "name": "tenant_lookup",
            "arguments": {"tenant": "tenant-a"},
            "_meta": _meta(
                traceparent="00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
                tracestate="vendor=value",
                baggage="tenant=tenant-a",
            ),
        },
    )
    traceparent = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    headers = _http_headers(
        "tools/call",
        name="tenant_lookup",
        extra={
            "Mcp-Param-Tenant": "tenant-a",
            "traceparent": traceparent,
            "tracestate": "vendor=value",
            "baggage": "tenant=tenant-a",
        },
    )
    response = await transport.handle(
        path="/mcp",
        headers=headers,
        body=jsonlib.dumps(request).encode(),
    )
    assert response.status_code == 200
    call_context = downstream.calls[-1][2]
    assert call_context.traceparent == traceparent
    assert call_context.tracestate == "vendor=value"
    assert call_context.baggage == "tenant=tenant-a"
    assert call_context.downstream_headers()["mcp-param-tenant"] == "tenant-a"

    missing_parameter_header = [
        item for item in headers if item[0] != "Mcp-Param-Tenant"
    ]
    rejected = await transport.handle(
        path="/mcp",
        headers=missing_parameter_header,
        body=jsonlib.dumps(request).encode(),
    )
    assert rejected.status_code == 400
    assert rejected.body is not None
    assert rejected.body["error"]["code"] == -32020


@pytest.mark.asyncio
async def test_transport_rejects_invalid_or_divergent_trace_context() -> None:
    transport = StreamableHTTPGateway(GraphOSV2Gateway(RecordingGraphOS()))
    request = _request(
        "tools/list",
        {"_meta": _meta(traceparent="not-a-trace")},
    )
    invalid = await transport.handle(
        path="/mcp",
        headers=_http_headers("tools/list"),
        body=jsonlib.dumps(request).encode(),
    )
    assert invalid.status_code == 400
    assert invalid.body is not None
    assert invalid.body["error"]["code"] == -32602

    request["params"] = {"_meta": _meta(tracestate="vendor=value")}
    invalid_state = await transport.handle(
        path="/mcp",
        headers=_http_headers("tools/list"),
        body=jsonlib.dumps(request).encode(),
    )
    assert invalid_state.status_code == 400
    assert invalid_state.body is not None
    assert invalid_state.body["error"]["code"] == -32602

    valid_trace = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    params = request["params"]
    assert isinstance(params, dict)
    params["_meta"] = _meta(traceparent=valid_trace)
    divergent = await transport.handle(
        path="/mcp",
        headers=_http_headers(
            "tools/list",
            extra={
                "traceparent": "00-abcdefabcdefabcdefabcdefabcdefab-0123456789abcdef-01"
            },
        ),
        body=jsonlib.dumps(request).encode(),
    )
    assert divergent.status_code == 400
    assert divergent.body is not None
    assert divergent.body["error"]["code"] == -32020


@pytest.mark.asyncio
async def test_transport_trims_ows_in_mirrored_headers_and_rejects_disabled_accept() -> (
    None
):
    transport = StreamableHTTPGateway(GraphOSV2Gateway(RecordingGraphOS()))
    request = _request(
        "tools/call",
        {"name": "graph_query", "arguments": {}, "_meta": _meta()},
    )
    headers = [
        (name, f" {value} " if name == "Mcp-Name" else value)
        for name, value in _http_headers("tools/call", name="graph_query")
    ]
    accepted = await transport.handle(
        path="/mcp", headers=headers, body=jsonlib.dumps(request).encode()
    )
    assert accepted.status_code == 200

    disabled = [
        ("Accept", "application/json;q=0, text/event-stream"),
        *[
            item
            for item in _http_headers("tools/call", name="graph_query")
            if item[0] != "Accept"
        ],
    ]
    rejected = await transport.handle(
        path="/mcp", headers=disabled, body=jsonlib.dumps(request).encode()
    )
    assert rejected.status_code == 400


@pytest.mark.asyncio
async def test_invalid_x_mcp_header_tool_is_filtered() -> None:
    downstream = RecordingGraphOS()
    downstream.tools.append(
        {
            "name": "invalid_header_tool",
            "inputSchema": {
                "type": "object",
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "string",
                                "x-mcp-header": "Value",
                            }
                        },
                    }
                ],
            },
        }
    )
    gateway = GraphOSV2Gateway(downstream)

    response = await gateway.dispatch(
        _request("tools/list", {"_meta": _meta()}),
        context=_context(),
    )

    assert "invalid_header_tool" not in {
        tool["name"] for tool in response["result"]["tools"]
    }


@pytest.mark.asyncio
async def test_downstream_legacy_handshake_matches_sse_response_and_closes_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = StatefulLegacyHTTPClient()
    monkeypatch.setattr(
        "mcp_v2_gateway.gateway.httpx.AsyncClient",
        lambda **_kwargs: fake,
    )
    client = StreamableHTTPGraphOSClient("http://127.0.0.1:8000/mcp")
    traceparent = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    context = GatewayRequestContext(
        authorization="Bearer tenant-token",
        headers={"mcp-param-tenant": "tenant-a"},
        traceparent=traceparent,
        tracestate="vendor=value",
        baggage="tenant=tenant-a",
    )

    result = await client.list_tools(context)
    called = await client.call_tool("graph_jobs", {"action": "status"}, context)

    assert [tool["name"] for tool in result["tools"]] == [
        "graph_query",
        "graph_jobs",
    ]
    assert jsonlib.loads(called["content"][0]["text"]) == {
        "status": "queued",
        "created_at": "2026-07-30T00:00:00Z",
    }
    assert [post[1]["method"] for post in fake.posts] == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
        "tools/list",
        "tools/call",
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
        "tools/list",
        "tools/call",
        "tools/call",
    ]
    assert fake.posts[3][1]["params"] == {
        "name": "load_tools",
        "arguments": {"tools": ["graph_jobs"], "auto_unload": True},
    }
    assert fake.posts[9][1]["params"] == fake.posts[3][1]["params"]
    assert fake.posts[5][1]["params"] == {
        "name": "unload_tools",
        "arguments": {"tools": ["graph_jobs"]},
    }
    assert fake.posts[-1][1]["params"] == fake.posts[5][1]["params"]
    assert len(fake.deletes) == 2
    assert fake.graph_jobs_loaded == set()
    assert fake.auto_unload == set()
    _assert_every_session_step_preserves_headers(fake)


@pytest.mark.asyncio
async def test_task_dispatch_uses_clean_sessions_for_catalog_call_and_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = StatefulLegacyHTTPClient()
    monkeypatch.setattr(
        "mcp_v2_gateway.gateway.httpx.AsyncClient",
        lambda **_kwargs: fake,
    )
    context = GatewayRequestContext(
        authorization="Bearer tenant-token",
        headers={"mcp-param-tenant": "tenant-a"},
        traceparent="00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
        tracestate="vendor=value",
        baggage="tenant=tenant-a",
    )

    response = await GraphOSV2Gateway(
        StreamableHTTPGraphOSClient("http://127.0.0.1:8000/mcp")
    ).dispatch(
        _request(
            "tools/call",
            {
                "name": "graph_jobs",
                "arguments": {"action": "dispatch", "task": "write report"},
                "_meta": _meta(tasks=True),
            },
        ),
        context=context,
    )

    assert response["result"]["resultType"] == "task"
    assert response["result"]["taskId"] == "job:opaque-work-item"
    assert fake.session_count == 3
    assert len(fake.deletes) == 3
    graph_job_actions = [
        params["arguments"]["action"]
        for _, request in fake.posts
        if request["method"] == "tools/call"
        and isinstance((params := request["params"]), dict)
        and params.get("name") == "graph_jobs"
        and isinstance(params.get("arguments"), dict)
    ]
    assert graph_job_actions == ["dispatch", "status"]
    assert (
        sum(
            request.get("params")
            == {
                "name": "load_tools",
                "arguments": {"tools": ["graph_jobs"], "auto_unload": True},
            }
            for _, request in fake.posts
        )
        == 3
    )
    assert (
        sum(
            request.get("params")
            == {
                "name": "unload_tools",
                "arguments": {"tools": ["graph_jobs"]},
            }
            for _, request in fake.posts
        )
        == 3
    )
    assert fake.graph_jobs_loaded == set()
    assert fake.auto_unload == set()
    _assert_every_session_step_preserves_headers(fake)


@pytest.mark.asyncio
async def test_concurrent_task_sessions_do_not_share_visibility_or_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = StatefulLegacyHTTPClient(yield_between_requests=True)
    monkeypatch.setattr(
        "mcp_v2_gateway.gateway.httpx.AsyncClient",
        lambda **_kwargs: fake,
    )
    gateway = GraphOSV2Gateway(StreamableHTTPGraphOSClient("http://127.0.0.1:8000/mcp"))
    contexts = [
        GatewayRequestContext(
            authorization=f"Bearer tenant-{label}-token",
            headers={"mcp-param-tenant": f"tenant-{label}"},
            traceparent=(f"00-{label * 32}-0123456789abcdef-01"),
            tracestate=f"vendor={label}",
            baggage=f"tenant=tenant-{label}",
        )
        for label in ("a", "b")
    ]

    responses = await asyncio.gather(
        *(
            gateway.dispatch(
                _request(
                    "tasks/get",
                    {"taskId": f"job:{label}", "_meta": _meta(tasks=True)},
                    request_id=index,
                ),
                context=context,
            )
            for index, (label, context) in enumerate(
                zip(("a", "b"), contexts, strict=True),
                start=1,
            )
        )
    )

    assert all(response["result"]["status"] == "working" for response in responses)
    assert fake.session_count == 2
    assert len(fake.session_headers) == 2
    assert len(fake.deletes) == 2
    assert {headers["Authorization"] for headers in fake.session_headers.values()} == {
        "Bearer tenant-a-token",
        "Bearer tenant-b-token",
    }
    assert {
        headers["mcp-param-tenant"] for headers in fake.session_headers.values()
    } == {"tenant-a", "tenant-b"}
    assert fake.graph_jobs_loaded == set()
    assert fake.auto_unload == set()
    _assert_every_session_step_preserves_headers(fake)


@pytest.mark.asyncio
async def test_downstream_graph_jobs_activation_fails_closed_when_not_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = StatefulLegacyHTTPClient(expose_graph_jobs=False)
    monkeypatch.setattr(
        "mcp_v2_gateway.gateway.httpx.AsyncClient",
        lambda **_kwargs: fake,
    )

    response = await GraphOSV2Gateway(
        StreamableHTTPGraphOSClient("http://127.0.0.1:8000/mcp")
    ).dispatch(_request("server/discover", {"_meta": _meta()}), context=_context())

    assert response["error"]["code"] == -32603
    assert "result" not in response
    assert len(fake.deletes) == 1
    assert fake.graph_jobs_loaded == set()
    assert fake.auto_unload == set()
    assert not any(
        isinstance(params := request.get("params"), dict)
        and params.get("name") == "graph_jobs"
        for _, request in fake.posts
    )


@pytest.mark.asyncio
async def test_anyio_cancel_scope_cleans_one_shot_session_and_propagates_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = CancellingLegacyHTTPClient()
    monkeypatch.setattr(
        "mcp_v2_gateway.gateway.httpx.AsyncClient",
        lambda **_kwargs: fake,
    )
    client = StreamableHTTPGraphOSClient("http://127.0.0.1:8000/mcp")
    cancelled = anyio.Event()

    async def call_graph_jobs() -> None:
        try:
            await client.call_tool("graph_jobs", {"action": "status"}, _context())
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(call_graph_jobs)
        await fake.graph_jobs_call_started.wait()
        assert fake.graph_jobs_loaded
        task_group.cancel_scope.cancel()

    assert cancelled.is_set()
    assert fake.unload_attempts == 1
    assert len(fake.deletes) == 1
    assert fake.graph_jobs_loaded == set()
    assert fake.auto_unload == set()


@pytest.mark.asyncio
async def test_cancel_during_normal_cleanup_finishes_delete_and_propagates_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = CancellingLegacyHTTPClient(block_unload=True)
    monkeypatch.setattr(
        "mcp_v2_gateway.gateway.httpx.AsyncClient",
        lambda **_kwargs: fake,
    )
    client = StreamableHTTPGraphOSClient("http://127.0.0.1:8000/mcp")
    task = asyncio.create_task(client.list_tools(_context()))

    await fake.unload_started.wait()
    task.cancel()
    fake.release_unload.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert fake.unload_attempts == 1
    assert len(fake.deletes) == 1
    assert fake.graph_jobs_loaded == set()
    assert fake.auto_unload == set()


@pytest.mark.asyncio
async def test_second_task_cancel_during_unload_still_deletes_and_propagates_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = CancellingLegacyHTTPClient(block_unload=True)
    monkeypatch.setattr(
        "mcp_v2_gateway.gateway.httpx.AsyncClient",
        lambda **_kwargs: fake,
    )
    client = StreamableHTTPGraphOSClient("http://127.0.0.1:8000/mcp")
    task = asyncio.create_task(
        client.call_tool("graph_jobs", {"action": "status"}, _context())
    )

    await fake.graph_jobs_call_started.wait()
    task.cancel()
    await fake.unload_started.wait()
    task.cancel()
    fake.release_unload.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert fake.unload_attempts == 1
    assert len(fake.deletes) == 1
    assert fake.graph_jobs_loaded == set()
    assert fake.auto_unload == set()


@pytest.mark.asyncio
async def test_repeated_cancel_during_delete_finishes_cleanup_and_propagates_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = CancellingLegacyHTTPClient(block_delete=True)
    monkeypatch.setattr(
        "mcp_v2_gateway.gateway.httpx.AsyncClient",
        lambda **_kwargs: fake,
    )
    client = StreamableHTTPGraphOSClient("http://127.0.0.1:8000/mcp")
    task = asyncio.create_task(
        client.call_tool("graph_jobs", {"action": "status"}, _context())
    )

    await fake.graph_jobs_call_started.wait()
    task.cancel()
    await fake.delete_started.wait()
    task.cancel()
    fake.release_delete.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert fake.unload_attempts == 1
    assert len(fake.deletes) == 1
    assert fake.graph_jobs_loaded == set()
    assert fake.auto_unload == set()


@pytest.mark.asyncio
async def test_cleanup_timeout_still_attempts_delete_and_propagates_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = CancellingLegacyHTTPClient(block_unload=True)
    monkeypatch.setattr(
        "mcp_v2_gateway.gateway.httpx.AsyncClient",
        lambda **_kwargs: fake,
    )
    client = StreamableHTTPGraphOSClient(
        "http://127.0.0.1:8000/mcp",
        timeout_seconds=0.01,
    )
    task = asyncio.create_task(
        client.call_tool("graph_jobs", {"action": "status"}, _context())
    )

    await fake.graph_jobs_call_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert task.cancelled()
    assert fake.unload_attempts == 1
    assert len(fake.deletes) == 1
    assert fake.graph_jobs_loaded == set()
    assert fake.auto_unload == set()


@pytest.mark.asyncio
async def test_downstream_error_text_is_not_exposed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = httpx.Response(
        200,
        headers={"Content-Type": "application/json"},
        json={
            "jsonrpc": "2.0",
            "id": "expected",
            "error": {"code": -32603, "message": "sensitive downstream detail"},
        },
        request=httpx.Request("POST", "http://127.0.0.1/mcp"),
    )

    with pytest.raises(
        GatewayProtocolError, match="Downstream MCP request failed"
    ) as caught:
        from mcp_v2_gateway.gateway import _parse_downstream_payload

        _parse_downstream_payload(response, expected_id="expected")

    assert "sensitive" not in str(caught.value)
