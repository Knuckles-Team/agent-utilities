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

from agent_utilities.mcp.httpx_boundary import coerce_httpx2_auth
from agent_utilities.mcp.protocol_compat import (
    force_legacy_protocol_mode,
    install_mcp_v2_bridge,
)

install_mcp_v2_bridge()

DEFAULT_MCP_TIMEOUT = 60.0


def _coerce_httpx_timeout(value: Any, default_timeout: float) -> Any:
    """Normalize any timeout-like ``value`` into a real :class:`httpx.Timeout`
    built from *this process's* ``httpx`` package (D-SNI-3).

    fastmcp's SDK v2 client (``fastmcp.client.transports.http``) is built
    against a separate, differently-versioned vendored copy of httpx —
    imported there as ``httpx2`` — and constructs its OWN ``httpx2.Timeout``
    (e.g. ``httpx2.Timeout(30.0, read=read_timeout_seconds)``) before calling
    the ``httpx_client_factory`` callback this module hands it. That foreign
    ``httpx2.Timeout`` instance is structurally identical to ``httpx.Timeout``
    (same ``connect``/``read``/``write``/``pool`` attributes) but is NOT an
    instance of it.

    ``httpx.Timeout.__init__`` only special-cases
    ``isinstance(timeout, httpx.Timeout)``; a same-shaped-but-foreign
    ``Timeout`` fails that check and falls through to the "explicit values"
    branch, which then assigns the *whole foreign object* — not a float — to
    ``self.connect``/``self.read``/``self.write``/``self.pool``. httpx's own
    connection-establishment code later computes a wall-clock deadline as
    ``time.monotonic() + self.connect`` (a float + that corrupted attribute)
    and raises ``TypeError: unsupported operand type(s) for +: 'float' and
    'Timeout'`` deep inside the connect path, surfaced by fastmcp as
    ``RuntimeError: Client failed to connect: ...``.

    Fix this at the boundary where a foreign type crosses into our own
    httpx-based client construction: duck-type any timeout-shaped object by
    its ``.connect``/``.read``/``.write``/``.pool`` attributes rather than
    trusting ``isinstance``, and rebuild a genuine local ``httpx.Timeout``
    from those (float | None) values.
    """
    import httpx

    if value is None:
        return httpx.Timeout(default_timeout)
    if isinstance(value, httpx.Timeout):
        return value
    if isinstance(value, int | float):
        return httpx.Timeout(float(value))
    if isinstance(value, tuple):
        return httpx.Timeout(value)
    timeout_attrs = ("connect", "read", "write", "pool")
    if all(hasattr(value, attr) for attr in timeout_attrs):
        return httpx.Timeout(**{attr: getattr(value, attr) for attr in timeout_attrs})
    raise TypeError(
        f"unsupported MCP client timeout type: {type(value)!r} (expected None, "
        "a number, a tuple, an httpx.Timeout, or a Timeout-shaped object "
        "exposing .connect/.read/.write/.pool)"
    )


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
            timeout=_coerce_httpx_timeout(timeout, default_timeout),
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
        # D-MTT-1: `auth` may be a local `httpx.Auth` built by
        # `client_credentials.child_auth()` (3 real call sites route through
        # here: skill_validation_assets.py, runtime_validation.py, and
        # agent_runner.py's `_spawn_auth()`, all reused per that function's
        # docstring). `SSETransport.__init__` stores it as `self.auth:
        # httpx2.Auth | None` and hands it straight to `mcp.client.sse.
        # sse_client(auth=self.auth, ...)` — the exact same crossing fixed at
        # multiplexer.py:1475 (see httpx_boundary.py's docstring), so it must
        # be coerced here. `StreamableHttpTransport` must NOT be coerced the
        # same way: it re-derives its own client via `_httpx_client_factory`
        # below, which builds a LOCAL `httpx.AsyncClient(auth=auth, ...)` —
        # `httpx.AsyncClient._build_auth` requires `isinstance(auth,
        # httpx.Auth)`, so handing it the httpx2-wrapped adapter instead
        # would swap a working case for a new, self-inflicted crossing
        # failure in the *other* direction. Branch on the transport actually
        # selected, not a blanket coercion.
        transport = transport_cls(
            url,
            headers=headers or None,
            auth=coerce_httpx2_auth(auth) if transport_cls is SSETransport else auth,
            httpx_client_factory=_httpx_client_factory(trust, timeout),
        )
        return force_legacy_protocol_mode(
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
    return force_legacy_protocol_mode(
        MCPToolset(transport, id=toolset_id) if toolset_id else MCPToolset(transport)
    )
