"""The multiplexer loads stdio AND remote (streamable-http / sse) children.

A child config with a ``command`` is a local stdio subprocess; one with a
``url`` (or http/sse ``transport``) is a remote server. Both must load from the
same ``mcp_config.json`` transparently.
"""

from __future__ import annotations

import contextlib
import importlib
from unittest.mock import MagicMock

import pytest

from agent_utilities.mcp import multiplexer as mod
from agent_utilities.mcp.multiplexer import MCPMultiplexer


@contextlib.asynccontextmanager
async def _streams_cm(streams):
    yield streams


def _fake_client(streams, recorder):
    def _client(*args, **kwargs):
        recorder.append({"args": args, "kwargs": kwargs})
        return _streams_cm(streams)

    return _client


class _FakeSession:
    async def initialize(self):  # noqa: D401
        return None

    async def list_tools(self):
        result = MagicMock()
        result.tools = []
        return result


class _FakeSessionCM:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return _FakeSession()

    async def __aexit__(self, *a):
        return False


@pytest.fixture
def transports(monkeypatch):
    rec = {"stdio": [], "http": [], "sse": []}
    # streamable-http yields a 3-tuple (read, write, get_session_id); the others 2.
    monkeypatch.setattr(mod, "stdio_client", _fake_client(("r", "w"), rec["stdio"]))
    monkeypatch.setattr(
        mod, "streamablehttp_client", _fake_client(("r", "w", "sid"), rec["http"])
    )
    monkeypatch.setattr(mod, "sse_client", _fake_client(("r", "w"), rec["sse"]))
    monkeypatch.setattr(mod, "ClientSession", _FakeSessionCM)
    return rec


@pytest.mark.asyncio
async def test_remote_child_uses_streamable_http(transports, tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    res = await mux._start_child("egeria-mcp", {"url": "http://egeria-mcp.arpa/mcp"})
    assert res is not None and res[0] == "egeria-mcp"
    assert len(transports["http"]) == 1 and not transports["stdio"]
    assert transports["http"][0]["args"][0] == "http://egeria-mcp.arpa/mcp"


@pytest.mark.asyncio
async def test_stdio_child_still_uses_stdio(transports, tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    res = await mux._start_child("graph-os", {"command": "graph-os", "args": []})
    assert res is not None
    assert len(transports["stdio"]) == 1 and not transports["http"]


@pytest.mark.asyncio
async def test_sse_url_uses_sse(transports, tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    res = await mux._start_child("foo", {"url": "http://foo.arpa/sse"})
    assert res is not None
    assert len(transports["sse"]) == 1 and not transports["http"]


@pytest.mark.asyncio
async def test_explicit_transport_without_url_is_remote(transports, tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    res = await mux._start_child(
        "bar", {"transport": "streamable-http", "url": "http://bar.arpa/mcp"}
    )
    assert res is not None and len(transports["http"]) == 1


@pytest.mark.asyncio
async def test_header_var_expansion(transports, tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret123")
    mux = MCPMultiplexer(tmp_path / "c.json")
    await mux._start_child(
        "auth-mcp",
        {
            "url": "http://auth.arpa/mcp",
            "headers": {"Authorization": "Bearer ${MY_TOKEN}"},
        },
    )
    assert (
        transports["http"][0]["kwargs"]["headers"]["Authorization"]
        == "Bearer secret123"
    )


@pytest.mark.asyncio
async def test_no_command_no_url_is_skipped(transports, tmp_path):
    mux = MCPMultiplexer(tmp_path / "c.json")
    res = await mux._start_child("bad", {})
    assert res is None
    assert not transports["stdio"] and not transports["http"] and not transports["sse"]


def _enable_service_auth(monkeypatch):
    monkeypatch.setenv("MCP_CLIENT_AUTH", "oidc-client-credentials")
    monkeypatch.setenv("OIDC_CLIENT_ID", "mcp-multiplexer")
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "s3cr3t")
    monkeypatch.setenv("OIDC_TOKEN_URL", "http://kc/token")
    import agent_utilities.mcp.client_credentials as cc

    importlib.reload(cc)  # reset provider cache under the new env
    return cc


@pytest.mark.asyncio
async def test_remote_child_gets_per_request_service_auth(
    transports, tmp_path, monkeypatch
):
    """A jwt child is authenticated via a per-request httpx.Auth, not a frozen
    Authorization header — so the pooled session survives token expiry."""
    cc = _enable_service_auth(monkeypatch)
    mux = MCPMultiplexer(tmp_path / "c.json")
    await mux._start_child("egeria-mcp", {"url": "http://egeria-mcp.arpa/mcp"})
    kwargs = transports["http"][0]["kwargs"]
    assert isinstance(kwargs["auth"], cc.ClientCredentialsAuth)
    # No baked-in (freezable) bearer in the session headers.
    assert "Authorization" not in (kwargs.get("headers") or {})


@pytest.mark.asyncio
async def test_child_own_authorization_not_overridden(
    transports, tmp_path, monkeypatch
):
    """A child that declares its own Authorization keeps it; the service auth
    flow is not attached (auth=None)."""
    _enable_service_auth(monkeypatch)
    mux = MCPMultiplexer(tmp_path / "c.json")
    await mux._start_child(
        "auth-mcp",
        {
            "url": "http://auth.arpa/mcp",
            "headers": {"Authorization": "Bearer child-own"},
        },
    )
    kwargs = transports["http"][0]["kwargs"]
    assert kwargs["auth"] is None
    assert kwargs["headers"]["Authorization"] == "Bearer child-own"
