#!/usr/bin/python
from __future__ import annotations

"""Cross-boundary coercion for the httpx / httpx2 duality (D-MTT-1).

CONCEPT:AU-ECO.mcp.protocol-compat-bridge

This environment installs **two structurally-identical but distinct** HTTP
client packages: ``httpx`` 0.28.1 (this repo's own base dependency, imported
everywhere as ``import httpx``) and ``httpx2`` 2.7.0 (a real, separately
published PyPI package — ``github.com/pydantic/httpx2``, "the next generation
HTTP client", by the same author/maintainer as ``httpx`` — pulled in
transitively by ``fastmcp-slim[client]==4.0.0b1`` and by ``mcp==2.0.0``
itself, both required by this repo's deliberate early adoption of
``fastmcp>=4.0.0b1``, see the ``[mcp]`` extra and the
``[tool.uv] override-dependencies`` comment in ``pyproject.toml``). fastmcp's
client transports (``fastmcp.client.transports.http.StreamableHttpTransport``,
``fastmcp.client.transports.sse.SSETransport``) and the ``mcp`` SDK v2 client
functions they call (``mcp.client.sse.sse_client``,
``mcp.client.streamable_http.streamable_http_client``) are typed against
``httpx2``, not ``httpx`` — so any value THIS package builds with its own
``httpx`` and hands across that boundary is a foreign object from the
receiving side's point of view, even though the two classes are
field-for-field identical.

``httpx.Timeout`` crossing that boundary already caused a production outage
(D-SNI-3, fixed by :func:`agent_utilities.mcp.toolset_factory._coerce_httpx_timeout`
— read that function's docstring for the full incident). This module is the
SECOND, and now the ONE canonical, place that owns this class of coercion —
every new crossing point should add a function here rather than re-inventing
the duck-typed-rebuild idiom inline. ``toolset_factory._coerce_httpx_timeout``
is left in place (it is covered by its own regression tests importing it by
name) rather than relocated, but follows the exact same shape as this module.

**Do not over-coerce.** Coercion here only ever recognizes two shapes: "already
the target (``httpx2``) type" (pass through) and "a genuine local ``httpx``
instance of the equivalent base class" (wrap it). Anything else — ``None``, a
bare string, a fastmcp-native auth provider (``OAuth``,
``ClientCredentialsOAuthProvider``, ``PrivateKeyJWTOAuthProvider``) — is passed
through completely unchanged: those are shapes fastmcp's OWN
``_set_auth``/``McpHttpClientFactory`` machinery already understands and
handles correctly, and guessing at them here would risk masking a real type
error instead of fixing a packaging artifact.
"""

from typing import Any

__all__ = ["coerce_httpx2_auth"]


def _local_auth_adapter(wrapped: Any, httpx2_module: Any) -> Any:
    """Build an ``httpx2.Auth`` instance that fully delegates to ``wrapped``
    (a local :class:`httpx.Auth`).

    ``httpx.Auth`` and ``httpx2.Auth`` share an identical override contract:
    a sync generator ``auth_flow(request)`` and, for schemes that need to do
    I/O without blocking the event loop (e.g. this package's
    ``ClientCredentialsAuth``, which offloads its token mint to a thread),
    an independently-overridable async generator ``async_auth_flow(request)``.
    Both must be forwarded — delegating only ``auth_flow`` and relying on
    ``httpx2.Auth``'s own default ``async_auth_flow`` (which just re-drives
    ``self.auth_flow`` synchronously inside the async path) would silently
    turn a non-blocking token refresh back into one that blocks the event
    loop, trading the isinstance ``TypeError`` this fixes for a subtler
    latency regression. A sync generator supports ``yield from`` for
    bidirectional (`.send()`) delegation directly; an async generator does
    not (no ``yield from`` for async generators in Python), so
    ``async_auth_flow`` below forwards manually via ``__anext__``/``asend``.
    """

    class _ForeignAuthAdapter(httpx2_module.Auth):
        def auth_flow(self, request: Any) -> Any:
            yield from wrapped.auth_flow(request)

        async def async_auth_flow(self, request: Any) -> Any:
            flow = wrapped.async_auth_flow(request)
            next_request = await flow.__anext__()
            try:
                while True:
                    response = yield next_request
                    try:
                        next_request = await flow.asend(response)
                    except StopAsyncIteration:
                        return
            finally:
                aclose = getattr(flow, "aclose", None)
                if callable(aclose):
                    await aclose()

    return _ForeignAuthAdapter()


def coerce_httpx2_auth(value: Any) -> Any:
    """Normalize an ``auth`` value crossing into fastmcp/mcp SDK v2 client code.

    ``fastmcp.client.transports.http.StreamableHttpTransport``,
    ``fastmcp.client.transports.sse.SSETransport``, and the underlying
    ``mcp.client.sse.sse_client`` / ``mcp.client.streamable_http.streamable_http_client``
    all type their ``auth`` parameter as ``httpx2.Auth | None`` (plus a couple
    of fastmcp-native provider shapes handled entirely inside fastmcp's own
    ``_set_auth``). This package builds its outbound child/service auth with
    the LOCAL ``httpx`` package instead —
    :func:`agent_utilities.mcp.client_credentials.child_auth` returns an
    ``httpx.Auth`` (either a plain ``httpx.BasicAuth`` or the self-refreshing
    ``ClientCredentialsAuth``) — because that object is *also* used to
    authenticate this package's own ``httpx.AsyncClient``-based transports
    (:mod:`agent_utilities.core.http_client`), where the local type is
    correct and required.

    Mirrors the ``mcp.client.sse.sse_client`` code path exactly (the
    concrete failure surfaced by mypy at ``multiplexer.py:1475``):
    ``httpx2.AsyncClient._build_auth`` special-cases
    ``isinstance(auth, httpx2.Auth)``; a same-shaped-but-foreign
    ``httpx.Auth`` instance fails that check, falls through to
    ``elif callable(auth)`` (an ``httpx.Auth`` *instance* is not callable —
    only the class is, which is irrelevant here), and hits
    ``else: raise TypeError(f'Invalid "auth" argument: {auth!r}')`` — an
    immediate hard failure for every SSE-transport child/toolset connection
    that carries service auth (``MCP_CLIENT_AUTH=oidc-client-credentials`` or
    ``basic``), verified against this environment's actual installed
    ``httpx2`` 2.7.0 source.

    Returns ``value`` unchanged for ``None``, a bare string (fastmcp resolves
    ``"oauth"``/a bearer-token string itself), or anything that is already an
    ``httpx2.Auth`` (including fastmcp's own ``OAuth`` /
    ``ClientCredentialsOAuthProvider`` / ``PrivateKeyJWTOAuthProvider``, all of
    which subclass it) — coercion only ever activates for a genuine local
    ``httpx.Auth`` instance, so a real type error in a value that is neither
    local nor foreign-but-compatible still surfaces as-is instead of being
    silently absorbed.
    """
    if value is None or isinstance(value, str):
        return value

    try:
        import httpx2
    except ImportError:
        # The `[mcp]` extra (fastmcp/mcp, and therefore httpx2) isn't
        # installed, so nothing on the other side of this boundary can exist
        # to receive `value` either — this call path is unreachable without
        # it. Hand the value through rather than raising here.
        return value

    if isinstance(value, httpx2.Auth):
        return value

    import httpx

    if not isinstance(value, httpx.Auth):
        # Not our local httpx.Auth, and not already an httpx2.Auth: a real
        # type error (or a shape this function doesn't yet know about), not
        # the httpx/httpx2 duality. Pass through unchanged rather than
        # guessing — fastmcp's own `_set_auth` will raise/reject it with a
        # clearer message than a masked coercion here ever could.
        return value

    return _local_auth_adapter(value, httpx2)
