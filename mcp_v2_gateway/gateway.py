"""The isolated MCP 2026-07-28 protocol boundary.

This module deliberately has no dependency on ``agent_utilities`` or FastMCP.
It is copied into a separate Python environment where the official ``mcp`` 2.x
SDK is installed.  GraphOS remains the policy, identity, tenant, consent, and
native WorkItem authority; this gateway only translates its authenticated MCP
tool surface into the stateless 2026 JSON-RPC envelope.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import re
import secrets
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from . import tracing

MCP_V2_PROTOCOL_VERSION = "2026-07-28"
LEGACY_PROTOCOL_VERSION = "2025-11-25"
TASKS_EXTENSION = "io.modelcontextprotocol/tasks"
TASKS_EXTENSION_REVISION = "2c1425d9a288b9b1f489430fe1e00bb392b47e48"
MISSING_REQUIRED_EXTENSION_CAPABILITY = -32003
_TASK_CAPABILITY: dict[str, dict[str, dict[str, object]]] = {
    "extensions": {TASKS_EXTENSION: {}}
}
_HEADER_TOKEN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_TRACEPARENT = re.compile(
    r"^(?P<version>[0-9a-f]{2})-(?P<trace>[0-9a-f]{32})-"
    r"(?P<parent>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})$"
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GatewayProtocolError(Exception):
    """A public JSON-RPC error; details never include credentials or endpoints."""

    code: int
    message: str
    data: dict[str, Any] | None = None


@dataclass(frozen=True)
class GatewayRequestContext:
    """Per-request security and propagation data; never retained by the gateway."""

    authorization: str
    headers: Mapping[str, str] = field(default_factory=dict)
    traceparent: str | None = None
    tracestate: str | None = None
    baggage: str | None = None

    def downstream_headers(self) -> dict[str, str]:
        """Return only explicitly safe, request-scoped forwarding headers."""
        forwarded = {
            name: value
            for name, value in self.headers.items()
            if name.lower().startswith("mcp-param-")
        }
        if self.traceparent is not None:
            forwarded["traceparent"] = self.traceparent
        if self.tracestate is not None:
            forwarded["tracestate"] = self.tracestate
        if self.baggage is not None:
            forwarded["baggage"] = self.baggage
        return forwarded


@dataclass(frozen=True)
class HTTPGatewayResponse:
    """Pure Streamable HTTP adapter result used by the server and tests."""

    status_code: int
    body: dict[str, Any] | None = None
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _HeaderBinding:
    name: str
    path: tuple[str, ...]
    value_type: str


class GraphOSClient(Protocol):
    """Minimal downstream GraphOS contract used by the gateway.

    It intentionally works through GraphOS's ordinary MCP tool surface.  No
    WorkItem record, task table, session, or authorization decision is kept in
    this sidecar.
    """

    async def list_tools(self, context: GatewayRequestContext) -> dict[str, Any]:
        raise NotImplementedError  # ABSTRACT-OK: downstream transport contract

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        raise NotImplementedError  # ABSTRACT-OK: downstream transport contract


class StreamableHTTPGraphOSClient(GraphOSClient):
    """One-shot client for GraphOS's legacy FastMCP streamable HTTP endpoint."""

    def __init__(self, url: str, *, timeout_seconds: float = 30.0) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("GRAPHOS_MCP_URL must be an absolute HTTP(S) URL")
        if parsed.scheme == "http" and parsed.hostname not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            raise ValueError("GRAPHOS_MCP_URL must use HTTPS outside loopback")
        self._url = url
        self._timeout = timeout_seconds

    async def list_tools(self, context: GatewayRequestContext) -> dict[str, Any]:
        return await self._request("tools/list", {}, context, activate_graph_jobs=True)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        return await self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
            context,
            activate_graph_jobs=name == "graph_jobs",
        )

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        context: GatewayRequestContext,
        *,
        activate_graph_jobs: bool = False,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": context.authorization,
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": LEGACY_PROTOCOL_VERSION,
        }
        headers.update(context.downstream_headers())
        async with httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=False
        ) as client:
            initialize_id = secrets.token_hex(12)
            initialized = await client.post(
                self._url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": initialize_id,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": LEGACY_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {
                            "name": "graphos-mcp-v2-gateway",
                            "version": "1",
                        },
                    },
                },
            )
            if initialized.status_code != 200:
                raise GatewayProtocolError(-32001, "Downstream authorization failed")
            session_id = initialized.headers.get("mcp-session-id")
            if not session_id:
                raise GatewayProtocolError(-32603, "Downstream MCP session unavailable")
            headers["Mcp-Session-Id"] = session_id
            try:
                _parse_downstream_payload(initialized, expected_id=initialize_id)
                acknowledged = await client.post(
                    self._url,
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized",
                        "params": {},
                    },
                )
                if acknowledged.status_code not in {200, 202}:
                    raise GatewayProtocolError(
                        -32603, "Downstream MCP initialization failed"
                    )
                if activate_graph_jobs:
                    # The condensed GraphOS surface scopes load_tools exposure to
                    # this legacy MCP session.  Each public gateway request opens a
                    # new session, so discovery and Tasks must activate the one
                    # owned capability again before advertising or calling it.
                    await self._post_request(client, headers, "tools/list", {})
                    await self._post_request(
                        client,
                        headers,
                        "tools/call",
                        {
                            "name": "load_tools",
                            "arguments": {
                                "tools": ["graph_jobs"],
                                "auto_unload": True,
                            },
                        },
                    )
                    activated_tools = await self._post_request(
                        client, headers, "tools/list", {}
                    )
                    if not _has_tool(activated_tools, "graph_jobs"):
                        raise GatewayProtocolError(
                            -32603, "Downstream graph_jobs tool unavailable"
                        )
                    if method == "tools/list":
                        return activated_tools
                return await self._post_request(client, headers, method, params)
            finally:
                await self._finish_session_cleanup(
                    client,
                    headers,
                    activate_graph_jobs=activate_graph_jobs,
                )

    async def _finish_session_cleanup(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        *,
        activate_graph_jobs: bool,
    ) -> None:
        """Finish bounded cleanup before propagating caller cancellation."""
        cleanup_task = asyncio.create_task(
            self._cleanup_session(
                client,
                headers,
                activate_graph_jobs=activate_graph_jobs,
            )
        )
        cancelled: asyncio.CancelledError | None = None
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError as exc:
                if cleanup_task.done():
                    cleanup_task.result()
                cancelled = exc
        cleanup_task.result()
        if cancelled is not None:
            raise cancelled

    async def _cleanup_session(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        *,
        activate_graph_jobs: bool,
    ) -> None:
        """Retract visibility and terminate one legacy session within fixed bounds."""
        try:
            if activate_graph_jobs:
                try:
                    await asyncio.wait_for(
                        self._post_request(
                            client,
                            headers,
                            "tools/call",
                            {
                                "name": "unload_tools",
                                "arguments": {"tools": ["graph_jobs"]},
                            },
                        ),
                        timeout=self._timeout,
                    )
                except (GatewayProtocolError, httpx.HTTPError, TimeoutError):
                    # Retraction is idempotent and best-effort; DELETE still
                    # terminates the short-lived downstream session.
                    _LOGGER.warning("Downstream MCP session visibility cleanup failed")
        finally:
            try:
                terminated = await asyncio.wait_for(
                    client.delete(self._url, headers=headers),
                    timeout=self._timeout,
                )
                if not 200 <= terminated.status_code < 300:
                    _LOGGER.warning("Downstream MCP session cleanup failed")
            except (httpx.HTTPError, TimeoutError):
                # Cleanup is best-effort after a session has been established.
                _LOGGER.warning("Downstream MCP session cleanup failed")

    async def _post_request(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        request_id = secrets.token_hex(12)
        response = await client.post(
            self._url,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            },
        )
        if response.status_code != 200:
            raise GatewayProtocolError(-32603, "Downstream MCP request failed")
        return _parse_downstream_payload(response, expected_id=request_id)


def _has_tool(result: Mapping[str, Any], name: str) -> bool:
    """Whether a legacy tools/list result exposes an exact named tool."""
    tools = result.get("tools")
    return isinstance(tools, list) and any(
        isinstance(tool, dict) and tool.get("name") == name for tool in tools
    )


def _parse_downstream_payload(
    response: httpx.Response, *, expected_id: str
) -> dict[str, Any]:
    """Decode JSON or the matching final JSON-RPC response in a legacy SSE body."""
    try:
        if "text/event-stream" not in response.headers.get("content-type", ""):
            payload = response.json()
        else:
            payload = _matching_sse_response(response.text, expected_id)
    except (json.JSONDecodeError, StopIteration, ValueError) as exc:
        raise GatewayProtocolError(
            -32603, "Downstream MCP response was invalid"
        ) from exc
    if not isinstance(payload, dict) or payload.get("id") != expected_id:
        raise GatewayProtocolError(-32603, "Downstream MCP response was invalid")
    if "error" in payload:
        raise GatewayProtocolError(-32603, "Downstream MCP request failed")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise GatewayProtocolError(-32603, "Downstream MCP response was invalid")
    return result


def _matching_sse_response(text: str, expected_id: str) -> dict[str, Any]:
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        candidate = json.loads(line[6:])
        if isinstance(candidate, dict) and candidate.get("id") == expected_id:
            return candidate
    raise StopIteration


class GraphOSV2Gateway:
    """Stateless, per-request MCP 2026-07-28 dispatcher."""

    def __init__(
        self,
        downstream: GraphOSClient,
        *,
        clock: Callable[[], float] = time.time,
        version: str = "1",
    ) -> None:
        self._downstream = downstream
        self._clock = clock
        self._version = version

    async def dispatch(
        self,
        request: Mapping[str, Any],
        *,
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        request_id = request.get("id")
        raw_method = request.get("method")
        method = raw_method if isinstance(raw_method, str) else "invalid"
        task_id = self._trace_task_id(request)
        with tracing.traced_dispatch(
            method=method,
            protocol_version=MCP_V2_PROTOCOL_VERSION,
            task_id=task_id,
            traceparent=context.traceparent,
            tracestate=context.tracestate,
            baggage=context.baggage,
        ) as span:
            try:
                self._validate_request(request)
                method = request["method"]
                params = request.get("params") or {}
                if method == "server/discover":
                    self._require_modern(params)
                    self._require_authorization(context)
                    tools = self._list_tools_result(
                        await self._downstream.list_tools(context)
                    )
                    response = self._success(request_id, self._discovery_result(tools))
                elif method == "tools/list":
                    self._require_modern(params)
                    self._require_authorization(context)
                    response = self._success(
                        request_id,
                        self._list_tools_result(
                            await self._downstream.list_tools(context)
                        ),
                    )
                elif method == "tools/call":
                    self._require_modern(params)
                    self._require_authorization(context)
                    result = await self._call_tool(params, context)
                    response = self._success(request_id, result)
                elif method in {"tasks/get", "tasks/update", "tasks/cancel"}:
                    self._require_modern(params)
                    self._require_tasks_capability(params)
                    self._require_authorization(context)
                    result = await self._task_method(method, params, context)
                    response = self._success(request_id, result)
                else:
                    raise GatewayProtocolError(-32601, "Method not found")
            except GatewayProtocolError as exc:
                response = self._error(request_id, exc)
            except Exception:
                # Deliberately do not serialize exception text: it can carry a
                # bearer, endpoint, tenant, or downstream implementation detail.
                response = self._error(
                    request_id, GatewayProtocolError(-32603, "Internal error")
                )
            tracing.finish_dispatch(
                span,
                method=method,
                protocol_version=MCP_V2_PROTOCOL_VERSION,
                task_id=task_id,
                response=response,
            )
            return response

    @staticmethod
    def _trace_task_id(request: Mapping[str, Any]) -> str | None:
        params = request.get("params")
        if not isinstance(params, Mapping):
            return None
        task_id = params.get("taskId")
        return task_id if isinstance(task_id, str) and task_id else None

    def _validate_request(self, request: Mapping[str, Any]) -> None:
        if request.get("jsonrpc") != "2.0" or not isinstance(
            request.get("method"), str
        ):
            raise GatewayProtocolError(-32600, "Invalid Request")
        request_id = request.get("id")
        if (
            "id" not in request
            or isinstance(request_id, bool)
            or not isinstance(request_id, (str, int))
        ):
            raise GatewayProtocolError(-32600, "Invalid Request")
        params = request.get("params") or {}
        if not isinstance(params, dict):
            raise GatewayProtocolError(-32602, "Invalid params")

    def _require_modern(self, params: Mapping[str, Any]) -> None:
        meta = params.get("_meta")
        if not isinstance(meta, dict):
            raise GatewayProtocolError(
                -32022,
                "Unsupported protocol version",
                {
                    "supported": [MCP_V2_PROTOCOL_VERSION],
                    "requested": "",
                },
            )
        requested = meta.get("io.modelcontextprotocol/protocolVersion")
        if requested != MCP_V2_PROTOCOL_VERSION:
            raise GatewayProtocolError(
                -32022,
                "Unsupported protocol version",
                {
                    "supported": [MCP_V2_PROTOCOL_VERSION],
                    "requested": str(requested or ""),
                },
            )
        capabilities = meta.get("io.modelcontextprotocol/clientCapabilities")
        if not isinstance(capabilities, dict):
            raise GatewayProtocolError(-32602, "Invalid params")

    def _require_tasks_capability(self, params: Mapping[str, Any]) -> None:
        meta = params.get("_meta")
        capabilities = (
            meta.get("io.modelcontextprotocol/clientCapabilities", {})
            if isinstance(meta, dict)
            else {}
        )
        extensions = (
            capabilities.get("extensions", {}) if isinstance(capabilities, dict) else {}
        )
        if not isinstance(extensions, dict) or TASKS_EXTENSION not in extensions:
            raise GatewayProtocolError(
                MISSING_REQUIRED_EXTENSION_CAPABILITY,
                "Missing required client capability",
                {"requiredCapabilities": _TASK_CAPABILITY},
            )

    @staticmethod
    def _require_authorization(context: GatewayRequestContext) -> None:
        authorization = context.authorization
        if (
            not isinstance(authorization, str)
            or not authorization.startswith("Bearer ")
            or len(authorization) <= 7
        ):
            raise GatewayProtocolError(-32001, "Unauthorized")

    async def _call_tool(
        self,
        params: Mapping[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise GatewayProtocolError(-32602, "Invalid params")
        tools = self._list_tools_result(await self._downstream.list_tools(context))[
            "tools"
        ]
        tool = next(
            (
                candidate
                for candidate in tools
                if isinstance(candidate, dict) and candidate.get("name") == name
            ),
            None,
        )
        if tool is None:
            raise GatewayProtocolError(-32602, "Unknown tool")
        self._validate_parameter_headers(tool, arguments, context.headers)
        if name == "graph_jobs" and arguments.get("action") == "dispatch":
            meta = params.get("_meta")
            if self._has_tasks_capability(meta):
                dispatched = self._tool_object(
                    await self._downstream.call_tool(name, arguments, context)
                )
                job_id = self._job_id(dispatched)
                # A task result is only permitted after the WorkItem can be read.
                status = await self._job_status(job_id, context)
                return self._task_from_status(job_id, status, result_type="task")
        return self._complete(
            await self._downstream.call_tool(name, arguments, context)
        )

    @staticmethod
    def _has_tasks_capability(meta: Any) -> bool:
        if not isinstance(meta, dict):
            return False
        capabilities = meta.get("io.modelcontextprotocol/clientCapabilities", {})
        return isinstance(capabilities, dict) and TASKS_EXTENSION in (
            capabilities.get("extensions", {}) or {}
        )

    async def _task_method(
        self,
        method: str,
        params: Mapping[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        task_id = params.get("taskId")
        if not isinstance(task_id, str) or not task_id:
            raise GatewayProtocolError(-32602, "Invalid params")
        if method == "tasks/get":
            return self._task_from_status(
                task_id, await self._job_status(task_id, context)
            )
        # WorkItems have no server-to-client input requests.  Update is therefore
        # an idempotent ack after the same tenant-scoped access check as polling.
        if method == "tasks/update":
            if not isinstance(params.get("inputResponses"), dict):
                raise GatewayProtocolError(-32602, "Invalid params")
            await self._job_status(task_id, context)
            return self._complete({})
        cancelled = self._tool_object(
            await self._downstream.call_tool(
                "graph_jobs", {"action": "cancel", "job_id": task_id}, context
            )
        )
        if str(cancelled.get("status", "")) == "not_cancelled":
            raise GatewayProtocolError(-32602, "Failed to cancel task")
        return self._complete({})

    async def _job_status(
        self, task_id: str, context: GatewayRequestContext
    ) -> dict[str, Any]:
        status = self._tool_object(
            await self._downstream.call_tool(
                "graph_jobs", {"action": "status", "job_id": task_id}, context
            )
        )
        if (
            not status
            or status.get("error")
            or str(status.get("status", "")).lower() == "not_found"
        ):
            raise GatewayProtocolError(-32602, "Failed to retrieve task")
        return status

    @staticmethod
    def _tool_object(result: Mapping[str, Any]) -> dict[str, Any]:
        """Decode GraphOS's normal JSON text tool result without trusting it."""
        if result.get("isError"):
            raise GatewayProtocolError(-32603, "Downstream GraphOS tool failed")
        if "content" not in result:
            return dict(result)
        content = result.get("content")
        if not isinstance(content, list) or not content:
            raise GatewayProtocolError(-32603, "Downstream GraphOS result was invalid")
        text = content[0].get("text") if isinstance(content[0], dict) else None
        if not isinstance(text, str):
            raise GatewayProtocolError(-32603, "Downstream GraphOS result was invalid")
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GatewayProtocolError(
                -32603, "Downstream GraphOS result was invalid"
            ) from exc
        if not isinstance(decoded, dict):
            raise GatewayProtocolError(-32603, "Downstream GraphOS result was invalid")
        return decoded

    @staticmethod
    def _job_id(dispatched: Mapping[str, Any]) -> str:
        job_id = dispatched.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise GatewayProtocolError(
                -32603, "Downstream dispatch did not return a task handle"
            )
        return job_id

    def _list_tools_result(self, downstream: Mapping[str, Any]) -> dict[str, Any]:
        raw_tools = downstream.get("tools")
        if not isinstance(raw_tools, list):
            raise GatewayProtocolError(-32603, "Downstream tool catalog was invalid")
        tools: list[dict[str, Any]] = []
        for candidate in raw_tools:
            if not isinstance(candidate, dict):
                continue
            name = candidate.get("name")
            schema = candidate.get("inputSchema")
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(schema, dict)
                or schema.get("type") != "object"
            ):
                _LOGGER.warning("Filtered malformed downstream tool definition")
                continue
            try:
                _header_bindings(schema)
            except ValueError:
                # Tool names are non-secret catalog data, but avoid logging schema
                # content or exceptions because descriptions can contain source data.
                _LOGGER.warning(
                    "Filtered tool with invalid x-mcp-header annotations: %s", name
                )
                continue
            tools.append(dict(candidate))
        result: dict[str, Any] = {
            "resultType": "complete",
            "tools": tools,
            "ttlMs": _bounded_ttl(downstream.get("ttlMs")),
            # The catalog is bearer/tenant filtered, so it is never public-cacheable.
            "cacheScope": "private",
        }
        next_cursor = downstream.get("nextCursor")
        if isinstance(next_cursor, str) and next_cursor:
            result["nextCursor"] = next_cursor
        return result

    @staticmethod
    def _validate_parameter_headers(
        tool: Mapping[str, Any],
        arguments: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> None:
        schema = tool.get("inputSchema")
        if not isinstance(schema, dict):
            raise GatewayProtocolError(-32603, "Tool definition was invalid")
        try:
            bindings = _header_bindings(schema)
        except ValueError as exc:
            raise GatewayProtocolError(-32602, "Tool definition was invalid") from exc
        normalized = {name.lower(): value for name, value in headers.items()}
        for binding in bindings:
            present, value = _value_at(arguments, binding.path)
            header_name = f"mcp-param-{binding.name}".lower()
            supplied = normalized.get(header_name)
            if not present or value is None:
                if supplied is not None:
                    raise GatewayProtocolError(-32020, "Header mismatch")
                continue
            try:
                expected = _parameter_string(value, binding.value_type)
                decoded = (
                    _decode_header_value(supplied) if supplied is not None else None
                )
            except ValueError as exc:
                raise GatewayProtocolError(-32020, "Header mismatch") from exc
            if decoded != expected:
                raise GatewayProtocolError(-32020, "Header mismatch")

    def _task_from_status(
        self, task_id: str, status: Mapping[str, Any], *, result_type: str = "complete"
    ) -> dict[str, Any]:
        raw_status = str(status.get("status", "")).lower()
        projected = {
            "queued": "working",
            "pending": "working",
            "ready": "working",
            "leased": "working",
            "running": "working",
            "executing": "working",
            "succeeded": "completed",
            "success": "completed",
            "completed": "completed",
            "failed": "failed",
            "dead_letter": "failed",
            "cancelled": "cancelled",
        }.get(raw_status)
        if projected is None:
            raise GatewayProtocolError(-32603, "Downstream task status was invalid")
        now = self._timestamp()
        created_value = status.get("created_at")
        created_at = _iso_timestamp(created_value, fallback=now)
        updated_at = _iso_timestamp(status.get("updated_at"), fallback=created_at)
        if _parse_iso_timestamp(updated_at) < _parse_iso_timestamp(created_at):
            raise GatewayProtocolError(
                -32603, "Downstream task timestamps were invalid"
            )
        if status.get("retention_ttl_ms") is not None:
            # WorkItems currently have no automatic record-retention deadline.
            # A future bounded policy cannot be safely projected as unlimited.
            raise GatewayProtocolError(-32603, "Downstream task retention was invalid")
        result: dict[str, Any] = {
            "resultType": result_type,
            "taskId": task_id,
            "status": projected,
            "createdAt": created_at,
            "lastUpdatedAt": updated_at,
            # Native WorkItems have no automatic record-expiration policy: they
            # are durable graph records, so the extension's unlimited value is exact.
            "ttlMs": None,
            "pollIntervalMs": 1_000,
        }
        if projected == "completed" and result_type == "complete":
            result["result"] = self._completed_task_result(task_id, status)
        elif projected == "failed" and result_type == "complete":
            result["error"] = {"code": -32603, "message": "GraphOS WorkItem failed"}
        return result

    @staticmethod
    def _completed_task_result(
        task_id: str, status: Mapping[str, Any]
    ) -> dict[str, Any]:
        value = status.get("result", status.get("output"))
        if isinstance(value, (dict, list)):
            text = json.dumps(value, separators=(",", ":"), default=str)
            structured: Any = value
        elif value is not None:
            text = str(value)
            structured = {"taskId": task_id, "output": value}
        else:
            text = "Task completed."
            structured = {"taskId": task_id, "status": "completed"}
        return {
            "resultType": "complete",
            "content": [{"type": "text", "text": text}],
            "structuredContent": structured,
            "isError": False,
        }

    def _timestamp(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._clock()))

    def _discovery_result(self, tools: Mapping[str, Any]) -> dict[str, Any]:
        visible_tools = tools.get("tools")
        capabilities: dict[str, Any] = {}
        if isinstance(visible_tools, list):
            capabilities["tools"] = {}
            if any(
                isinstance(tool, dict) and tool.get("name") == "graph_jobs"
                for tool in visible_tools
            ):
                capabilities["extensions"] = {TASKS_EXTENSION: {}}
        return {
            "resultType": "complete",
            "supportedVersions": [MCP_V2_PROTOCOL_VERSION],
            "capabilities": capabilities,
            "_meta": {
                "io.modelcontextprotocol/serverInfo": {
                    "name": "graphos-mcp-v2-gateway",
                    "version": self._version,
                }
            },
            "instructions": "Stateless GraphOS MCP gateway. Use graph_jobs dispatch for durable task handles.",
            "ttlMs": 60_000,
            "cacheScope": "private",
        }

    @staticmethod
    def _complete(value: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(value)
        result.setdefault("resultType", "complete")
        return result

    @staticmethod
    def _success(request_id: Any, result: Mapping[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": dict(result)}

    @staticmethod
    def _error(request_id: Any, error: GatewayProtocolError) -> dict[str, Any]:
        body: dict[str, Any] = {"code": error.code, "message": error.message}
        if error.data is not None:
            body["data"] = error.data
        return {"jsonrpc": "2.0", "id": request_id, "error": body}


class StreamableHTTPGateway:
    """Thin, stateless HTTP binding for the MCP 2026-07-28 dispatcher."""

    def __init__(
        self,
        gateway: GraphOSV2Gateway,
        *,
        allowed_origins: Sequence[str] = (),
        max_request_bytes: int = 1_048_576,
    ) -> None:
        if max_request_bytes <= 0:
            raise ValueError("max_request_bytes must be positive")
        self._gateway = gateway
        self._allowed_origins = frozenset(allowed_origins)
        self._max_request_bytes = max_request_bytes

    async def handle(
        self,
        *,
        path: str,
        headers: Iterable[tuple[str, str]],
        body: bytes,
    ) -> HTTPGatewayResponse:
        try:
            normalized = _normalize_headers(headers)
        except GatewayProtocolError as exc:
            return self._response(400, GraphOSV2Gateway._error(None, exc))
        origin = normalized.get("origin")
        if origin is not None and origin not in self._allowed_origins:
            return self._response(
                403,
                GraphOSV2Gateway._error(
                    None, GatewayProtocolError(-32001, "Forbidden origin")
                ),
            )
        if path != "/mcp":
            return self._response(404)
        if len(body) > self._max_request_bytes:
            return self._response(413)
        content_type = normalized.get("content-type", "").split(";", 1)[0].strip()
        if content_type != "application/json":
            return self._response(415)
        try:
            decoded = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._response(
                400,
                GraphOSV2Gateway._error(
                    None, GatewayProtocolError(-32700, "Parse error")
                ),
            )
        if not isinstance(decoded, dict):
            return self._response(
                400,
                GraphOSV2Gateway._error(
                    None, GatewayProtocolError(-32600, "Invalid Request")
                ),
            )
        request_id = decoded.get("id")
        if "id" not in decoded:
            # The 2026 core defines no client-to-server HTTP notifications and
            # this gateway advertises no extension notification methods.
            return self._response(
                400,
                GraphOSV2Gateway._error(
                    None, GatewayProtocolError(-32601, "Method not found")
                ),
            )
        try:
            self._validate_accept(normalized)
            self._validate_mirrored_headers(decoded, normalized)
        except GatewayProtocolError as exc:
            return self._response(400, GraphOSV2Gateway._error(request_id, exc))
        try:
            trace_context = _trace_context(decoded, normalized)
        except GatewayProtocolError as exc:
            return self._response(400, GraphOSV2Gateway._error(request_id, exc))
        context = GatewayRequestContext(
            authorization=normalized.get("authorization", ""),
            headers=normalized,
            **trace_context,
        )
        response = await self._gateway.dispatch(decoded, context=context)
        status = self._status_for(response)
        headers_out: dict[str, str] = {}
        if status == 401:
            headers_out["WWW-Authenticate"] = "Bearer"
        return self._response(status, response, extra_headers=headers_out)

    @staticmethod
    def _validate_accept(headers: Mapping[str, str]) -> None:
        accepted: set[str] = set()
        for item in headers.get("accept", "").split(","):
            value, *parameters = item.split(";")
            if any(parameter.strip().lower() == "q=0" for parameter in parameters):
                continue
            accepted.add(value.strip().lower())
        if not {"application/json", "text/event-stream"}.issubset(accepted):
            raise GatewayProtocolError(-32600, "Invalid Request")

    @staticmethod
    def _validate_mirrored_headers(
        request: Mapping[str, Any], headers: Mapping[str, str]
    ) -> None:
        params = request.get("params")
        meta = params.get("_meta") if isinstance(params, dict) else None
        body_version = (
            meta.get("io.modelcontextprotocol/protocolVersion")
            if isinstance(meta, dict)
            else None
        )
        header_version = _trim_header_value(headers.get("mcp-protocol-version"))
        if header_version is None or header_version != body_version:
            raise GatewayProtocolError(-32020, "Header mismatch")
        if header_version != MCP_V2_PROTOCOL_VERSION:
            raise GatewayProtocolError(
                -32022,
                "Unsupported protocol version",
                {
                    "supported": [MCP_V2_PROTOCOL_VERSION],
                    "requested": header_version,
                },
            )
        method = request.get("method")
        header_method = _trim_header_value(headers.get("mcp-method"))
        if not isinstance(method, str) or header_method is None:
            raise GatewayProtocolError(-32020, "Header mismatch")
        try:
            _validate_plain_header_value(header_method)
        except ValueError as exc:
            raise GatewayProtocolError(-32020, "Header mismatch") from exc
        if header_method != method:
            raise GatewayProtocolError(-32020, "Header mismatch")
        if method in {"tools/call", "resources/read", "prompts/get"}:
            source_key = "uri" if method == "resources/read" else "name"
            body_name = params.get(source_key) if isinstance(params, dict) else None
            header_name = headers.get("mcp-name")
            if not isinstance(body_name, str) or header_name is None:
                raise GatewayProtocolError(-32020, "Header mismatch")
            try:
                decoded_name = _decode_header_value(header_name)
            except ValueError as exc:
                raise GatewayProtocolError(-32020, "Header mismatch") from exc
            if decoded_name != body_name:
                raise GatewayProtocolError(-32020, "Header mismatch")

    @staticmethod
    def _status_for(response: Mapping[str, Any]) -> int:
        error = response.get("error")
        if not isinstance(error, dict):
            return 200
        code = error.get("code")
        if code == -32601:
            return 404
        if code == -32001:
            return 401
        if code in {
            -32700,
            -32600,
            -32602,
            -32020,
            MISSING_REQUIRED_EXTENSION_CAPABILITY,
            -32022,
        }:
            return 400
        if code == -32603:
            return 500
        return 200

    @staticmethod
    def _response(
        status: int,
        body: dict[str, Any] | None = None,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> HTTPGatewayResponse:
        headers = {"Cache-Control": "no-store"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if extra_headers:
            headers.update(extra_headers)
        return HTTPGatewayResponse(status_code=status, body=body, headers=headers)


def _normalize_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_name, raw_value in headers:
        name = raw_name.lower()
        if not _HEADER_TOKEN.fullmatch(raw_name):
            raise GatewayProtocolError(-32020, "Header mismatch")
        if "\r" in raw_value or "\n" in raw_value:
            raise GatewayProtocolError(-32020, "Header mismatch")
        if name in normalized:
            if name == "accept":
                normalized[name] = f"{normalized[name]},{raw_value}"
                continue
            raise GatewayProtocolError(-32020, "Header mismatch")
        normalized[name] = raw_value
    return normalized


def _bounded_ttl(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 300_000:
        return value
    return 60_000


def _iso_timestamp(value: Any, *, fallback: str) -> str:
    candidate = fallback if value is None else value
    if isinstance(candidate, bool):
        raise GatewayProtocolError(-32603, "Downstream task timestamp was invalid")
    if isinstance(candidate, (int, float)):
        try:
            parsed = datetime.fromtimestamp(candidate, tz=UTC)
        except (OverflowError, OSError, ValueError) as exc:
            raise GatewayProtocolError(
                -32603, "Downstream task timestamp was invalid"
            ) from exc
        return parsed.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if not isinstance(candidate, str):
        raise GatewayProtocolError(-32603, "Downstream task timestamp was invalid")
    parsed = _parse_iso_timestamp(candidate)
    return (
        parsed.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )


def _parse_iso_timestamp(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise GatewayProtocolError(
            -32603, "Downstream task timestamp was invalid"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise GatewayProtocolError(-32603, "Downstream task timestamp was invalid")
    return parsed


def _header_bindings(schema: Mapping[str, Any]) -> list[_HeaderBinding]:
    bindings: list[_HeaderBinding] = []
    seen: set[str] = set()

    def visit(node: Any, path: tuple[str, ...], *, reachable: bool) -> None:
        if not isinstance(node, dict):
            if _contains_header_annotation(node):
                raise ValueError("x-mcp-header is not on a schema property")
            return
        annotation = node.get("x-mcp-header")
        if annotation is not None:
            value_type = node.get("type")
            if (
                not reachable
                or not path
                or not isinstance(annotation, str)
                or not _HEADER_TOKEN.fullmatch(annotation)
                or value_type not in {"string", "integer", "boolean"}
            ):
                raise ValueError("invalid x-mcp-header annotation")
            key = annotation.lower()
            if key in seen:
                raise ValueError("duplicate x-mcp-header annotation")
            seen.add(key)
            bindings.append(
                _HeaderBinding(name=annotation, path=path, value_type=value_type)
            )
        for key, value in node.items():
            if key == "x-mcp-header":
                continue
            if key == "properties" and reachable:
                if not isinstance(value, dict):
                    if _contains_header_annotation(value):
                        raise ValueError("invalid properties schema")
                    continue
                for property_name, child in value.items():
                    if not isinstance(property_name, str):
                        raise ValueError("invalid property name")
                    visit(child, (*path, property_name), reachable=True)
                continue
            if _contains_header_annotation(value):
                raise ValueError("x-mcp-header is not statically reachable")

    visit(schema, (), reachable=True)
    return bindings


def _contains_header_annotation(value: Any) -> bool:
    if isinstance(value, dict):
        return "x-mcp-header" in value or any(
            _contains_header_annotation(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_header_annotation(child) for child in value)
    return False


def _value_at(arguments: Mapping[str, Any], path: Sequence[str]) -> tuple[bool, Any]:
    current: Any = arguments
    for segment in path:
        if not isinstance(current, Mapping) or segment not in current:
            return False, None
        current = current[segment]
    return True, current


def _parameter_string(value: Any, value_type: str) -> str:
    if value_type == "string" and isinstance(value, str):
        return value
    if value_type == "boolean" and isinstance(value, bool):
        return "true" if value else "false"
    if (
        value_type == "integer"
        and isinstance(value, int)
        and not isinstance(value, bool)
        and -(2**53) + 1 <= value <= 2**53 - 1
    ):
        return str(value)
    raise ValueError("parameter value does not match the annotated primitive type")


def _decode_header_value(value: str | None) -> str:
    if value is None:
        raise ValueError("missing header")
    trimmed = _trim_header_value(value)
    if trimmed is None:
        raise ValueError("missing header")
    if trimmed.startswith("=?base64?") and trimmed.endswith("?="):
        encoded = trimmed[len("=?base64?") : -2]
        try:
            return base64.b64decode(encoded, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise ValueError("invalid encoded header") from exc
    _validate_plain_header_value(trimmed)
    return trimmed


def _validate_plain_header_value(value: str) -> None:
    if any(char != "\t" and not 0x20 <= ord(char) <= 0x7E for char in value):
        raise ValueError("unsafe header character")


def _trim_header_value(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip(" \t")


def _validate_traceparent(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise GatewayProtocolError(-32602, "Invalid trace context")
    match = _TRACEPARENT.fullmatch(value)
    if (
        match is None
        or match.group("version") == "ff"
        or match.group("trace") == "0" * 32
        or match.group("parent") == "0" * 16
    ):
        raise GatewayProtocolError(-32602, "Invalid trace context")
    return value


def _validate_tracestate(value: Any, *, traceparent: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or traceparent is None or len(value) > 512:
        raise GatewayProtocolError(-32602, "Invalid trace context")
    members = value.split(",")
    if not members or len(members) > 32:
        raise GatewayProtocolError(-32602, "Invalid trace context")
    seen: set[str] = set()
    for member in members:
        if member != member.strip(" \t") or "=" not in member:
            raise GatewayProtocolError(-32602, "Invalid trace context")
        key, member_value = member.split("=", 1)
        if (
            not _valid_tracestate_key(key)
            or key in seen
            or not member_value
            or member_value.endswith((" ", "\t"))
            or any(
                not 0x20 <= ord(char) <= 0x7E or char in {",", "="}
                for char in member_value
            )
        ):
            raise GatewayProtocolError(-32602, "Invalid trace context")
        seen.add(key)
    return value


def _validate_baggage(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 8192:
        raise GatewayProtocolError(-32602, "Invalid trace context")
    for member in value.split(","):
        if "=" not in member:
            raise GatewayProtocolError(-32602, "Invalid trace context")
        key, member_value = member.split("=", 1)
        if (
            not key.strip()
            or not member_value.strip()
            or any(not 0x20 <= ord(char) <= 0x7E for char in member)
        ):
            raise GatewayProtocolError(-32602, "Invalid trace context")
    return value


def _trace_context(
    request: Mapping[str, Any], headers: Mapping[str, str]
) -> dict[str, str | None]:
    params = request.get("params")
    meta = params.get("_meta") if isinstance(params, Mapping) else None
    if not isinstance(meta, Mapping):
        return {"traceparent": None, "tracestate": None, "baggage": None}
    traceparent = _validate_traceparent(meta.get("traceparent"))
    tracestate = _validate_tracestate(meta.get("tracestate"), traceparent=traceparent)
    baggage = _validate_baggage(meta.get("baggage"))
    for key, body_value in (
        ("traceparent", traceparent),
        ("tracestate", tracestate),
        ("baggage", baggage),
    ):
        header_value = headers.get(key)
        if header_value is not None and _trim_header_value(header_value) != body_value:
            raise GatewayProtocolError(-32020, "Header mismatch")
    return {
        "traceparent": traceparent,
        "tracestate": tracestate,
        "baggage": baggage,
    }


def _valid_tracestate_key(value: str) -> bool:
    if "@" not in value:
        return bool(
            len(value) <= 256 and re.fullmatch(r"[a-z][a-z0-9_\-\*/]{0,255}", value)
        )
    tenant, system = value.split("@", 1)
    return bool(
        1 <= len(tenant) <= 241
        and re.fullmatch(r"[a-z0-9][a-z0-9_\-\*/]{0,240}", tenant)
        and 1 <= len(system) <= 14
        and re.fullmatch(r"[a-z][a-z0-9_\-\*/]{0,13}", system)
    )
