"""The multiplexer speaks MCP SDK v2's remote-transport contract.

CONCEPT:AU-ECO.mcp.protocol-compat-bridge

Regression guard for the fastmcp-4 default. SDK v2 renamed
``streamablehttp_client`` -> ``streamable_http_client`` AND replaced its
``headers`` / ``auth`` / ``httpx_client_factory`` keywords with a single
pre-configured ``http_client=``. Before this guard existed the multiplexer
still imported the v1 name behind a ``try/except ImportError`` that bound it to
``None``, so every remote child raised ``RuntimeError: mcp SDK has no
streamablehttp_client`` — and since ``deploy/mcp-fleet.registry.yml`` defaults
every fleet service to ``transport: streamable-http``, that is the entire
deployed fleet.

The pre-existing coverage for this path lives in ``tests/test_multiplexer_transports.py``,
which ``pytest.ini``'s ``testpaths`` (``tests/unit tests/integration
tests/retrieval``) does not collect — it is dead coverage and caught none of
this. These live in ``tests/unit`` on purpose.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock

import pytest

from agent_utilities.mcp import multiplexer as mod
from agent_utilities.mcp.multiplexer import MCPMultiplexer


def test_multiplexer_imports_the_sdk_v2_transport_names() -> None:
    """Both remote transports are hard imports, not ``None`` fallbacks."""
    assert callable(mod.streamable_http_client)
    assert callable(mod.sse_client)
    assert not hasattr(mod, "streamablehttp_client")


class _FakeSession:
    async def initialize(self):
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
def recorded_http(monkeypatch):
    """Capture what the multiplexer hands to ``streamable_http_client``."""
    calls: list[dict] = []

    @contextlib.asynccontextmanager
    async def _fake(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        # SDK v2 yields (read, write) — the v1 third `get_session_id` is gone.
        yield ("r", "w")

    monkeypatch.setattr(mod, "streamable_http_client", _fake)
    monkeypatch.setattr(mod, "ClientSession", _FakeSessionCM)

    from agent_utilities.core.config import config

    # Plain-http fixture hostname, as a real TLS-terminated-at-ingress child.
    monkeypatch.setattr(config, "mcp_http_allowed_private_hosts", ["fleet.example"])
    return calls


@pytest.mark.asyncio
async def test_remote_child_uses_v2_http_client_keyword(recorded_http, tmp_path):
    """The URL is positional and the security-hardened client is passed as
    ``http_client=`` — the SDK v2 signature, not v1's headers/auth kwargs."""
    mux = MCPMultiplexer(tmp_path / "c.json")
    result = await mux._start_child("fleet", {"url": "http://fleet.example/mcp"})

    assert result is not None, "remote streamable-http child must start"
    assert len(recorded_http) == 1
    call = recorded_http[0]
    assert call["args"] == ("http://fleet.example/mcp",)
    assert set(call["kwargs"]) == {"http_client"}
    # v1-only keywords must not be resurrected — SDK v2 rejects them.
    assert "httpx_client_factory" not in call["kwargs"]


@pytest.mark.asyncio
async def test_remote_child_headers_ride_on_the_http_client(recorded_http, tmp_path):
    """Child-declared headers must reach the wire. Under SDK v2 they can only
    do so baked into the client, so assert them there rather than on a kwarg
    the transport no longer accepts."""
    mux = MCPMultiplexer(tmp_path / "c.json")
    await mux._start_child(
        "fleet",
        {"url": "http://fleet.example/mcp", "headers": {"X-Fleet-Tenant": "acme"}},
    )

    http_client = recorded_http[0]["kwargs"]["http_client"]
    assert http_client.headers["X-Fleet-Tenant"] == "acme"
    # The hardened factory's posture survives the handover.
    assert http_client.follow_redirects is False
