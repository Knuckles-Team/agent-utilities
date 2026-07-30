"""The isolated MCP 2026-07-28 protocol boundary.

This module deliberately has no dependency on ``agent_utilities`` or FastMCP.
It is copied into a separate Python environment where the official ``mcp`` 2.x
SDK is installed.  GraphOS remains the policy, identity, tenant, consent, and
native WorkItem authority; this gateway only translates its authenticated MCP
tool surface into the stateless 2026 JSON-RPC envelope.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

MCP_V2_PROTOCOL_VERSION = "2026-07-28"
TASKS_EXTENSION = "io.modelcontextprotocol/tasks"
_TASK_CAPABILITY: dict[str, dict[str, dict[str, object]]] = {
    "extensions": {TASKS_EXTENSION: {}}
}


@dataclass(frozen=True)
class GatewayProtocolError(Exception):
    """A public JSON-RPC error; details never include credentials or endpoints."""

    code: int
    message: str
    data: dict[str, Any] | None = None


class GraphOSClient(Protocol):
    """Minimal downstream GraphOS contract used by the gateway.

    It intentionally works through GraphOS's ordinary MCP tool surface.  No
    WorkItem record, task table, session, or authorization decision is kept in
    this sidecar.
    """

    async def list_tools(self, authorization: str) -> dict[str, Any]:
        raise NotImplementedError  # ABSTRACT-OK: downstream transport contract

    async def call_tool(
        self, name: str, arguments: dict[str, Any], authorization: str
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

    async def list_tools(self, authorization: str) -> dict[str, Any]:
        return await self._request("tools/list", {}, authorization)

    async def call_tool(
        self, name: str, arguments: dict[str, Any], authorization: str
    ) -> dict[str, Any]:
        return await self._request(
            "tools/call", {"name": name, "arguments": arguments}, authorization
        )

    async def _request(
        self, method: str, params: dict[str, Any], authorization: str
    ) -> dict[str, Any]:
        headers = {
            "Authorization": authorization,
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=False
        ) as client:
            initialized = await client.post(
                self._url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": secrets.token_hex(12),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
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
            response = await client.post(
                self._url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": secrets.token_hex(12),
                    "method": method,
                    "params": params,
                },
            )
        if response.status_code != 200:
            raise GatewayProtocolError(-32603, "Downstream MCP request failed")
        return _parse_downstream_payload(response)


def _parse_downstream_payload(response: httpx.Response) -> dict[str, Any]:
    """Decode JSON or the one JSON-RPC frame GraphOS returns over SSE."""
    try:
        if "text/event-stream" not in response.headers.get("content-type", ""):
            payload = response.json()
        else:
            payload = next(
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ")
            )
    except (json.JSONDecodeError, StopIteration, ValueError) as exc:
        raise GatewayProtocolError(
            -32603, "Downstream MCP response was invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise GatewayProtocolError(-32603, "Downstream MCP response was invalid")
    if "error" in payload:
        error = payload["error"]
        message = (
            error.get("message", "Downstream MCP request failed")
            if isinstance(error, dict)
            else "Downstream MCP request failed"
        )
        raise GatewayProtocolError(-32603, str(message)[:256])
    result = payload.get("result")
    if not isinstance(result, dict):
        raise GatewayProtocolError(-32603, "Downstream MCP response was invalid")
    return result


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
        self, request: Mapping[str, Any], *, authorization: str | None
    ) -> dict[str, Any]:
        request_id = request.get("id")
        try:
            self._validate_request(request)
            method = request["method"]
            params = request.get("params") or {}
            if method == "server/discover":
                self._require_modern(params)
                auth = self._require_authorization(authorization)
                tools = await self._downstream.list_tools(auth)
                return self._success(request_id, self._discovery_result(tools))
            if method == "tools/list":
                self._require_modern(params)
                auth = self._require_authorization(authorization)
                tools = await self._downstream.list_tools(auth)
                return self._success(request_id, self._complete(tools))
            if method == "tools/call":
                self._require_modern(params)
                auth = self._require_authorization(authorization)
                result = await self._call_tool(params, auth)
                return self._success(request_id, result)
            if method in {"tasks/get", "tasks/update", "tasks/cancel"}:
                self._require_modern(params)
                self._require_tasks_capability(params)
                auth = self._require_authorization(authorization)
                result = await self._task_method(method, params, auth)
                return self._success(request_id, result)
            raise GatewayProtocolError(-32601, "Method not found")
        except GatewayProtocolError as exc:
            return self._error(request_id, exc)
        except Exception:
            # Deliberately do not serialize exception text: it can carry a bearer,
            # endpoint, tenant, or downstream implementation detail.
            return self._error(
                request_id, GatewayProtocolError(-32603, "Internal error")
            )

    def _validate_request(self, request: Mapping[str, Any]) -> None:
        if request.get("jsonrpc") != "2.0" or not isinstance(
            request.get("method"), str
        ):
            raise GatewayProtocolError(-32600, "Invalid Request")
        params = request.get("params") or {}
        if not isinstance(params, dict):
            raise GatewayProtocolError(-32602, "Invalid params")

    def _require_modern(self, params: Mapping[str, Any]) -> None:
        meta = params.get("_meta")
        if (
            not isinstance(meta, dict)
            or meta.get("io.modelcontextprotocol/protocolVersion")
            != MCP_V2_PROTOCOL_VERSION
        ):
            raise GatewayProtocolError(-32022, "Unsupported protocol version")

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
                -32021,
                "Missing required client capability",
                {"requiredCapabilities": _TASK_CAPABILITY},
            )

    @staticmethod
    def _require_authorization(authorization: str | None) -> str:
        if (
            not isinstance(authorization, str)
            or not authorization.startswith("Bearer ")
            or len(authorization) <= 7
        ):
            raise GatewayProtocolError(-32001, "Unauthorized")
        return authorization

    async def _call_tool(
        self, params: Mapping[str, Any], authorization: str
    ) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise GatewayProtocolError(-32602, "Invalid params")
        if name == "graph_jobs" and arguments.get("action") == "dispatch":
            meta = params.get("_meta")
            if self._has_tasks_capability(meta):
                dispatched = self._tool_object(
                    await self._downstream.call_tool(name, arguments, authorization)
                )
                job_id = self._job_id(dispatched)
                # A task result is only permitted after the WorkItem can be read.
                status = await self._job_status(job_id, authorization)
                return self._task_from_status(job_id, status, result_type="task")
        return self._complete(
            await self._downstream.call_tool(name, arguments, authorization)
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
        self, method: str, params: Mapping[str, Any], authorization: str
    ) -> dict[str, Any]:
        task_id = params.get("taskId")
        if not isinstance(task_id, str) or not task_id:
            raise GatewayProtocolError(-32602, "Invalid params")
        if method == "tasks/get":
            return self._task_from_status(
                task_id, await self._job_status(task_id, authorization)
            )
        # WorkItems have no server-to-client input requests.  Update is therefore
        # an idempotent ack after the same tenant-scoped access check as polling.
        if method == "tasks/update":
            await self._job_status(task_id, authorization)
            return self._complete({})
        cancelled = self._tool_object(
            await self._downstream.call_tool(
                "graph_jobs", {"action": "cancel", "job_id": task_id}, authorization
            )
        )
        if str(cancelled.get("status", "")) == "not_cancelled":
            raise GatewayProtocolError(-32602, "Failed to cancel task")
        return self._complete({})

    async def _job_status(self, task_id: str, authorization: str) -> dict[str, Any]:
        status = self._tool_object(
            await self._downstream.call_tool(
                "graph_jobs", {"action": "status", "job_id": task_id}, authorization
            )
        )
        if not status or status.get("error"):
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
        result: dict[str, Any] = {
            "resultType": result_type,
            "taskId": task_id,
            "status": projected,
            "createdAt": str(status.get("created_at") or now),
            "lastUpdatedAt": str(status.get("updated_at") or now),
            "ttlMs": 86_400_000,
            "pollIntervalMs": 1_000,
        }
        if projected == "completed":
            result["result"] = self._complete(
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(dict(status), separators=(",", ":")),
                        }
                    ],
                    "isError": False,
                }
            )
        elif projected == "failed":
            result["error"] = {"code": -32603, "message": "GraphOS WorkItem failed"}
        return result

    def _timestamp(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._clock()))

    def _discovery_result(self, tools: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "resultType": "complete",
            "supportedVersions": [MCP_V2_PROTOCOL_VERSION],
            "capabilities": {
                "tools": {"listChanged": True},
                "extensions": {TASKS_EXTENSION: {}},
            },
            "_meta": {
                "io.modelcontextprotocol/serverInfo": {
                    "name": "graphos-mcp-v2-gateway",
                    "version": self._version,
                }
            },
            "instructions": "Stateless GraphOS MCP gateway. Use graph_jobs dispatch for durable task handles.",
            "ttlMs": 60_000,
            "cacheScope": "private",
            "tools": tools.get("tools", []),
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
