#!/usr/bin/python
from __future__ import annotations

"""Build pydantic-ai v2 ``MCPToolset`` clients from connection specs.

CONCEPT:AU-ECO.messaging.native-backend-abstraction

pydantic-ai v2 removed ``MCPServerSSE`` / ``MCPServerStreamableHTTP`` /
``MCPServerStdio`` / ``FastMCPToolset``; the unified MCP client is a single
``MCPToolset`` wrapping a transport (``StreamableHttpTransport`` /
``SSETransport`` / ``StdioTransport``). This module is the ONE place that knows
how to turn a connection spec (a URL or a stdio command) into a toolset, so
callers (the agent factory, the orchestration runner, the graph builder, the
coordinated-KG path) never repeat transport construction.

TLS policy and request timeout are configured through the transport's
``httpx_client_factory``. This is the single AgentConfig-backed construction
path for remote MCP clients; boolean verification controls are not supported.
"""

from typing import Any

DEFAULT_MCP_TIMEOUT = 60.0


def _httpx_client_factory(tls_profile: Any, default_timeout: float) -> Any:
    """Return an MCP client factory closing over resolved TLS policy + timeout.

    CONCEPT:AU-ORCH.adapter.transport-toolset-factory — the transport invokes this factory with a transport-version
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

    from agent_utilities.core.http_client import create_async_http_client

    def factory(
        headers: dict[str, str] | None = None,
        timeout: Any | None = None,
        auth: Any | None = None,
        follow_redirects: bool = True,
        **_forward_compat: Any,
    ) -> Any:
        return create_async_http_client(
            headers=headers,
            auth=auth,
            timeout=timeout if timeout is not None else httpx.Timeout(default_timeout),
            follow_redirects=follow_redirects,
            **tls_profile.httpx_kwargs(),
        )

    return factory


def build_http_toolset(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    auth: Any | None = None,
    timeout: float = DEFAULT_MCP_TIMEOUT,
    toolset_id: str | None = None,
    tls_service: str = "mcp",
    tls_profile: str | None = None,
    tls_profile_ref: str | None = None,
) -> Any:
    """Build an ``MCPToolset`` for an HTTP/SSE MCP server URL.

    Transport is inferred from the URL: a ``/sse`` suffix selects
    ``SSETransport``, otherwise streamable HTTP. TLS is resolved from the
    active AgentConfig plus the optional neutral profile selectors.
    """
    from pydantic_ai.mcp import MCPToolset, SSETransport, StreamableHttpTransport

    from agent_utilities.core.transport_security import (
        resolve_configured_tls_profile,
    )

    trust = resolve_configured_tls_profile(
        tls_service,
        profile_name=tls_profile,
        profile_ref=tls_profile_ref,
    )

    try:
        transport_cls = (
            SSETransport
            if str(url).rstrip("/").lower().endswith("/sse")
            else StreamableHttpTransport
        )
        transport = transport_cls(
            url,
            headers=headers or None,
            auth=auth,
            httpx_client_factory=_httpx_client_factory(trust, timeout),
        )
        return (
            MCPToolset(transport, id=toolset_id)
            if toolset_id
            else MCPToolset(transport)
        )
    finally:
        # The SSLContext has already loaded CA/mTLS material; the client factory
        # closes over that context, so runtime files need not outlive construction.
        trust.cleanup()


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
