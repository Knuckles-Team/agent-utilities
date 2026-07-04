"""A spawned agent inherits service-account auth for remote MCP toolsets.

CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap / OS-5.32 — `execute_agent` resolves a fleet Server and binds
its remote (SSE/streamable-HTTP) toolset. Those toolsets must carry the same
service-account bearer the multiplexer attaches to its children, or a
jwt-protected `*.arpa` server rejects the call `401`. These tests pin that the
bearer (minted via `client_credentials.bearer_header`) reaches the toolset's
transport (pydantic-ai v2 carries auth headers on the MCP transport, which
threads them into the lazily-built httpx client at connect time), and that the
path is inert/safe when auth is disabled.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.orchestration import agent_runner


def _toolset_transport_headers(toolset: Any) -> dict[str, str]:
    """Extract the auth headers a built MCP toolset will present.

    pydantic-ai v2's ``MCPToolset`` wraps an ``fastmcp`` ``Client`` whose
    ``transport`` (``StreamableHttpTransport``/``SSETransport``) holds the
    headers; the httpx client is built lazily at connect time from those
    headers, so we assert against the transport — the eager pre-v2
    ``httpx.AsyncClient(headers=...)`` construction no longer happens at build.
    """
    transport = getattr(getattr(toolset, "client", None), "transport", None)
    return dict(getattr(transport, "headers", None) or {})


def test_spawn_auth_headers_returns_minted_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent_utilities.mcp.client_credentials.bearer_header",
        lambda _existing: {"Authorization": "Bearer TESTTOKEN"},
    )
    assert agent_runner._spawn_auth_headers() == {"Authorization": "Bearer TESTTOKEN"}


def test_spawn_auth_headers_inert_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent_utilities.mcp.client_credentials.bearer_header", lambda _existing: {}
    )
    assert agent_runner._spawn_auth_headers() == {}


def test_spawn_auth_headers_degrades_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_existing: Any) -> dict[str, str]:
        raise RuntimeError("mint failed")

    monkeypatch.setattr("agent_utilities.mcp.client_credentials.bearer_header", _boom)
    assert agent_runner._spawn_auth_headers() == {}


def _remote_meta() -> dict[str, Any]:
    return {
        "type": "server",
        "url": "http://repository-manager-mcp.arpa/mcp",  # streamable-http branch
        "tools": [],
        "capabilities": [],
    }


def test_remote_toolset_carries_bearer_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent_utilities.mcp.client_credentials.bearer_header",
        lambda _existing: {"Authorization": "Bearer TESTTOKEN"},
    )

    config = agent_runner._build_execution_config(
        object(), "code-enhancer", _remote_meta()
    )

    assert config["mcp_toolsets"], "a remote toolset should be bound"
    headers = _toolset_transport_headers(config["mcp_toolsets"][0])
    assert headers.get("Authorization") == "Bearer TESTTOKEN"


def test_remote_toolset_no_bearer_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent_utilities.mcp.client_credentials.bearer_header", lambda _existing: {}
    )

    config = agent_runner._build_execution_config(
        object(), "code-enhancer", _remote_meta()
    )

    assert config["mcp_toolsets"]
    headers = _toolset_transport_headers(config["mcp_toolsets"][0])
    assert "Authorization" not in headers
