"""Conformance-style JSON-RPC checks for the isolated MCP v2 gateway."""

from __future__ import annotations

import json as jsonlib
from collections.abc import Mapping

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
    class FakeAsyncClient:
        def __init__(self, **_kwargs: object) -> None:
            self.posts: list[tuple[dict[str, str], dict[str, object]]] = []
            self.deletes: list[dict[str, str]] = []
            self.session_count = 0
            self.graph_jobs_loaded: set[str] = set()

        async def __aenter__(self) -> FakeAsyncClient:
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
            self.posts.append((dict(headers), json))
            method = json["method"]
            if method == "notifications/initialized":
                return httpx.Response(
                    202,
                    request=httpx.Request("POST", url),
                )
            request_id = json["id"]
            if method == "initialize":
                self.session_count += 1
                session_id = f"synthetic-session-{self.session_count}"
                result = {"protocolVersion": "2025-11-25", "capabilities": {}}
            else:
                session_id = headers["Mcp-Session-Id"]
                if method == "tools/list":
                    tools = [{"name": "graph_query"}]
                    if session_id in self.graph_jobs_loaded:
                        tools.append({"name": "graph_jobs"})
                    result = {"tools": tools}
                elif json["params"] == {
                    "name": "load_tools",
                    "arguments": {"tools": ["graph_jobs"]},
                }:
                    self.graph_jobs_loaded.add(session_id)
                    result = {"content": [{"type": "text", "text": "{}"}]}
                else:
                    result = {
                        "content": [{"type": "text", "text": '{"status":"queued"}'}]
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
            self.deletes.append(dict(headers))
            return httpx.Response(200, request=httpx.Request("DELETE", url))

    fake = FakeAsyncClient()
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
    )

    result = await client.list_tools(context)
    called = await client.call_tool("graph_jobs", {"action": "status"}, context)

    assert result == {"tools": [{"name": "graph_query"}, {"name": "graph_jobs"}]}
    assert called == {"content": [{"type": "text", "text": '{"status":"queued"}'}]}
    assert [post[1]["method"] for post in fake.posts] == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
        "tools/list",
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
        "tools/list",
        "tools/call",
    ]
    assert fake.posts[3][1]["params"] == {
        "name": "load_tools",
        "arguments": {"tools": ["graph_jobs"]},
    }
    assert fake.posts[8][1]["params"] == fake.posts[3][1]["params"]
    request_headers = fake.posts[-1][0]
    assert request_headers["MCP-Protocol-Version"] == "2025-11-25"
    assert request_headers["Mcp-Session-Id"] == "synthetic-session-2"
    assert request_headers["traceparent"] == traceparent
    assert request_headers["tracestate"] == "vendor=value"
    assert request_headers["mcp-param-tenant"] == "tenant-a"
    assert len(fake.deletes) == 2


@pytest.mark.asyncio
async def test_downstream_graph_jobs_activation_fails_closed_when_not_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAsyncClient:
        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def post(
            self, url: str, *, headers: dict[str, str], json: dict[str, object]
        ) -> httpx.Response:
            method = json["method"]
            request_id = json.get("id")
            if method == "notifications/initialized":
                return httpx.Response(202, request=httpx.Request("POST", url))
            result = (
                {"protocolVersion": "2025-11-25", "capabilities": {}}
                if method == "initialize"
                else {"tools": [{"name": "graph_query"}]}
            )
            response_headers = {"Content-Type": "application/json"}
            if method == "initialize":
                response_headers["Mcp-Session-Id"] = "synthetic-session"
            return httpx.Response(
                200,
                headers=response_headers,
                json={"jsonrpc": "2.0", "id": request_id, "result": result},
                request=httpx.Request("POST", url),
            )

        async def delete(self, url: str, *, headers: dict[str, str]) -> httpx.Response:
            return httpx.Response(200, request=httpx.Request("DELETE", url))

    monkeypatch.setattr(
        "mcp_v2_gateway.gateway.httpx.AsyncClient", lambda **_kwargs: FakeAsyncClient()
    )

    response = await GraphOSV2Gateway(
        StreamableHTTPGraphOSClient("http://127.0.0.1:8000/mcp")
    ).dispatch(_request("server/discover", {"_meta": _meta()}), context=_context())

    assert response["error"]["code"] == -32603
    assert "result" not in response


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
