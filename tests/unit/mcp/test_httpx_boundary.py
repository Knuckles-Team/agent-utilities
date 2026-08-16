"""Regression tests for the httpx / httpx2 auth boundary adapter (U-51, D-MTT-1).

The MCP SDK's client transports (``mcp.client.auth.oauth2.OAuthClientProvider``,
fastmcp's ``StreamableHttpTransport``/``SSETransport``) type their ``auth``
parameter as ``httpx2.Auth``. This package's own outbound child/service auth
(:func:`agent_utilities.mcp.client_credentials.child_auth`) builds a plain
``httpx.Auth`` instance instead, because that same object also authenticates
this package's own ``httpx.AsyncClient``-based transports where the local type
is required. ``coerce_httpx2_auth`` is the one canonical adapter that lets a
local ``httpx.Auth`` instance cross into httpx2-typed SDK code without either
package being monkeypatched or aliased.

First test class proves the ORIGINAL bug is real (a raw local ``httpx.Auth``
handed to httpx2 fails closed with ``TypeError`` before any request is sent —
the "failing-without" baseline). The rest exercise the adapter itself: identity
passthrough for shapes it must never touch, and full bidirectional delegation
(sync and async) for the one shape it does wrap.
"""

from __future__ import annotations

import sys

import httpx
import httpx2
import pytest

from agent_utilities.mcp.httpx_boundary import coerce_httpx2_auth


class _LocalSyncAuth(httpx.Auth):
    """A local httpx.Auth using only the sync auth_flow override."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def auth_flow(self, request):  # noqa: ANN001 - httpx.Auth's own signature
        self.calls.append("sync-start")
        request.headers["Authorization"] = "Bearer sync-token"
        response = yield request
        self.calls.append(f"sync-saw-status:{response.status_code}")


class _LocalAsyncAuth(httpx.Auth):
    """A local httpx.Auth overriding async_auth_flow, mirroring the shape of
    this package's ClientCredentialsAuth (offloads its token mint off-thread
    and drives a bidirectional async generator rather than the sync default).
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def async_auth_flow(self, request):  # noqa: ANN001
        self.calls.append("async-start")
        # Simulate an awaited, off-thread token mint before the first yield.
        request.headers["Authorization"] = "Bearer async-token"
        response = yield request
        self.calls.append(f"async-saw-status:{response.status_code}")
        if response.status_code == 401:
            # Prove a second round-trip (e.g. a retry-after-refresh) is
            # forwarded too, not just the first request/response pair.
            request.headers["Authorization"] = "Bearer async-token-refreshed"
            response2 = yield request
            self.calls.append(f"async-saw-status-2:{response2.status_code}")


def _httpx2_request() -> httpx2.Request:
    return httpx2.Request("GET", "https://child.example.test/mcp")


def _httpx2_response(status_code: int, request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(status_code, request=request)


class TestOriginalBugIsReal:
    """Failing-without baseline: a raw local httpx.Auth crossing into httpx2
    client construction fails closed with TypeError, exactly as the D-MTT-1 /
    U-51 incident describes and as `multiplexer.py:1475` hit at runtime."""

    def test_raw_local_auth_rejected_by_httpx2_client(self) -> None:
        with pytest.raises(TypeError, match='Invalid "auth" argument'):
            httpx2.Client(auth=_LocalSyncAuth())

    def test_raw_local_auth_rejected_by_httpx2_async_client(self) -> None:
        with pytest.raises(TypeError, match='Invalid "auth" argument'):
            httpx2.AsyncClient(auth=_LocalAsyncAuth())

    def test_coerced_auth_is_accepted_by_httpx2_client(self) -> None:
        """The adapter is what turns the TypeError above into acceptance."""
        wrapped = coerce_httpx2_auth(_LocalSyncAuth())
        client = httpx2.Client(auth=wrapped)
        assert client.auth is wrapped


class TestPassthroughShapes:
    """Shapes the adapter must never touch — coercing them would risk masking
    a real type error instead of fixing the packaging artifact (see the
    module docstring's "Do not over-coerce" section)."""

    def test_none_passes_through(self) -> None:
        assert coerce_httpx2_auth(None) is None

    def test_bare_string_passes_through(self) -> None:
        assert coerce_httpx2_auth("oauth") == "oauth"

    def test_already_httpx2_auth_passes_through_unchanged(self) -> None:
        class _NativeAuth(httpx2.Auth):
            def auth_flow(self, request):  # noqa: ANN001
                yield request

        native = _NativeAuth()
        assert coerce_httpx2_auth(native) is native

    def test_unrelated_value_passes_through_unchanged(self) -> None:
        """Neither a local httpx.Auth nor an httpx2.Auth: a real type error
        (or an unrecognized shape) should surface as-is from fastmcp's own
        `_set_auth`, not be silently absorbed here."""
        sentinel = object()
        assert coerce_httpx2_auth(sentinel) is sentinel

    def test_missing_httpx2_returns_value_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the `[mcp]` extra (and therefore httpx2) isn't installed, nothing
        on the receiving side of the boundary can exist either; must not raise."""
        monkeypatch.setitem(sys.modules, "httpx2", None)
        auth = _LocalSyncAuth()
        assert coerce_httpx2_auth(auth) is auth


class TestBidirectionalDelegation:
    """The adapter must forward the COMPLETE flow — both directions, both the
    sync and the async override shape — not just the first yielded request."""

    def test_sync_auth_flow_delegates_via_yield_from(self) -> None:
        local = _LocalSyncAuth()
        wrapped = coerce_httpx2_auth(local)
        assert isinstance(wrapped, httpx2.Auth)

        request = _httpx2_request()
        gen = wrapped.auth_flow(request)
        first = next(gen)
        assert first.headers["Authorization"] == "Bearer sync-token"

        with pytest.raises(StopIteration):
            gen.send(_httpx2_response(200, request))

        assert local.calls == ["sync-start", "sync-saw-status:200"]

    async def test_async_auth_flow_forwards_bidirectionally(self) -> None:
        """Regression for the exact bug this module's docstring calls out:
        relying on httpx2.Auth's default async_auth_flow (which just re-drives
        auth_flow synchronously) would silently block the event loop for an
        auth scheme that only overrides async_auth_flow. This drives the
        wrapped generator through TWO round trips via asend(), proving both
        the first yield and the mid-flow resend are forwarded, not just the
        generator's start."""
        local = _LocalAsyncAuth()
        wrapped = coerce_httpx2_auth(local)
        assert isinstance(wrapped, httpx2.Auth)

        request = _httpx2_request()
        flow = wrapped.async_auth_flow(request)

        first = await flow.__anext__()
        assert first.headers["Authorization"] == "Bearer async-token"

        second = await flow.asend(_httpx2_response(401, request))
        assert second.headers["Authorization"] == "Bearer async-token-refreshed"

        with pytest.raises(StopAsyncIteration):
            await flow.asend(_httpx2_response(200, request))

        assert local.calls == [
            "async-start",
            "async-saw-status:401",
            "async-saw-status-2:200",
        ]

    async def test_async_auth_flow_closes_wrapped_generator(self) -> None:
        """`.aclose()` on the outer generator must close the inner one too, or
        an early-abandoned flow (e.g. a cancelled request) leaks the wrapped
        generator instead of running its cleanup."""
        closed = False

        class _TrackedAsyncAuth(httpx.Auth):
            async def async_auth_flow(self, request):  # noqa: ANN001
                nonlocal closed
                try:
                    yield request
                    yield request  # pragma: no cover - never reached
                finally:
                    closed = True

        wrapped = coerce_httpx2_auth(_TrackedAsyncAuth())
        request = _httpx2_request()
        flow = wrapped.async_auth_flow(request)
        await flow.__anext__()
        await flow.aclose()
        assert closed is True
