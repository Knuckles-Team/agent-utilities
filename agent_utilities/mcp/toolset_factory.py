#!/usr/bin/python
from __future__ import annotations

"""Build pydantic-ai v2 ``MCPToolset`` clients from connection specs.

CONCEPT:ECO-4.0

pydantic-ai v2 removed ``MCPServerSSE`` / ``MCPServerStreamableHTTP`` /
``MCPServerStdio`` / ``FastMCPToolset``; the unified MCP client is a single
``MCPToolset`` wrapping a transport (``StreamableHttpTransport`` /
``SSETransport`` / ``StdioTransport``). This module is the ONE place that knows
how to turn a connection spec (a URL or a stdio command) into a toolset, so
callers (the agent factory, the orchestration runner, the graph builder, the
coordinated-KG path) never repeat transport construction.

SSL verification and request timeout must be configured through the transport's
``httpx_client_factory`` (when both ``verify`` and ``httpx_client_factory`` are
given, pydantic-ai ignores ``verify``), so this is where that contract lives.
"""

from typing import Any

DEFAULT_MCP_TIMEOUT = 60.0


def _httpx_client_factory(verify: bool | str, default_timeout: float) -> Any:
    """Return a ``McpHttpClientFactory`` closing over SSL ``verify`` + timeout.

    CONCEPT:ORCH-1.101 — the transport invokes this factory with a transport-version
    dependent kwarg set. fastmcp's streamable-HTTP transport calls it as
    ``factory(headers=, auth=, follow_redirects=, timeout=)`` (the ``follow_redirects``
    kwarg was added in fastmcp ≥3.x); the older shape was ``factory(headers=, timeout=,
    auth=)``. Binding a remote MCP toolset (e.g. an AgentTemplate's ``graph-os``) failed
    hard with ``factory() got an unexpected keyword argument 'follow_redirects'``, so the
    toolset could never connect and the agent ran tool-less. Accept ``follow_redirects``
    explicitly (forwarded to httpx, which strips ``Authorization`` on cross-origin
    redirects) and swallow any further forward-compat kwargs so a transport bump can
    never break the connect path. We honor the transport-supplied timeout when present
    and fall back to our default.
    """
    import httpx

    def factory(
        headers: dict[str, str] | None = None,
        timeout: Any | None = None,
        auth: Any | None = None,
        follow_redirects: bool = True,
        **_forward_compat: Any,
    ) -> Any:
        return httpx.AsyncClient(
            headers=headers,
            auth=auth,
            timeout=timeout if timeout is not None else httpx.Timeout(default_timeout),
            verify=verify,
            follow_redirects=follow_redirects,
        )

    return factory


def build_http_toolset(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    verify: bool | str = True,
    timeout: float = DEFAULT_MCP_TIMEOUT,
    toolset_id: str | None = None,
) -> Any:
    """Build an ``MCPToolset`` for an HTTP/SSE MCP server URL.

    Transport is inferred from the URL: a ``/sse`` suffix selects
    ``SSETransport``, otherwise streamable HTTP. ``verify`` and ``timeout`` are
    threaded through an httpx client factory.
    """
    from pydantic_ai.mcp import MCPToolset, SSETransport, StreamableHttpTransport

    transport_cls = (
        SSETransport
        if str(url).rstrip("/").lower().endswith("/sse")
        else StreamableHttpTransport
    )
    transport = transport_cls(
        url,
        headers=headers or None,
        httpx_client_factory=_httpx_client_factory(verify, timeout),
    )
    toolset = (
        MCPToolset(transport, id=toolset_id) if toolset_id else MCPToolset(transport)
    )
    return toolset


def build_stdio_toolset(
    command: str,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    toolset_id: str | None = None,
) -> Any:
    """Build an ``MCPToolset`` for a stdio (subprocess) MCP server."""
    from pydantic_ai.mcp import MCPToolset, StdioTransport

    transport = StdioTransport(command=command, args=args, env=env)
    return MCPToolset(transport, id=toolset_id) if toolset_id else MCPToolset(transport)
