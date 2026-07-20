"""A spawned agent inherits service-account auth for remote MCP toolsets.

CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap / OS-5.32 — `execute_agent` resolves a fleet Server and binds
its remote (SSE/streamable-HTTP) toolset. Those toolsets must carry the same
service-account bearer the multiplexer attaches to its children, or a
jwt-protected `*.arpa` server rejects the call `401`. These tests pin that the
bearer (minted via `client_credentials.child_auth_header`) reaches the toolset's
transport (pydantic-ai v2 carries auth headers on the MCP transport, which
threads them into the lazily-built httpx client at connect time), and that the
path is inert/safe when auth is disabled.
"""

from __future__ import annotations

from types import SimpleNamespace
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
        "agent_utilities.mcp.client_credentials.child_auth_header",
        lambda _existing: {"Authorization": "Bearer TESTTOKEN"},
    )
    assert agent_runner._spawn_auth_headers() == {"Authorization": "Bearer TESTTOKEN"}


def test_spawn_auth_headers_inert_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent_utilities.mcp.client_credentials.child_auth_header", lambda _existing: {}
    )
    assert agent_runner._spawn_auth_headers() == {}


def test_spawn_auth_headers_fails_closed_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_existing: Any) -> dict[str, str]:
        raise RuntimeError("mint failed")

    monkeypatch.setattr(
        "agent_utilities.mcp.client_credentials.child_auth_header", _boom
    )
    with pytest.raises(RuntimeError, match="mint failed"):
        agent_runner._spawn_auth_headers()


def _remote_meta() -> dict[str, Any]:
    return {
        "type": "server",
        "toolset_id": "repository-manager-mcp",
        "tools": [],
        "capabilities": [],
    }


def _configured_model() -> SimpleNamespace:
    return SimpleNamespace(
        id="synthetic-standard",
        provider="openai",
        base_url=None,
        api_key=None,
    )


def test_remote_toolset_carries_bearer_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent_utilities.mcp.client_credentials.child_auth_header",
        lambda _existing: {"Authorization": "Bearer TESTTOKEN"},
    )
    monkeypatch.setattr(
        agent_runner,
        "_fleet_server_url",
        lambda _server: "https://fleet.example.invalid/mcp",
    )
    monkeypatch.setattr(
        agent_runner, "_configured_model_for_class", lambda _class: _configured_model()
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
        "agent_utilities.mcp.client_credentials.child_auth_header", lambda _existing: {}
    )
    monkeypatch.setattr(
        agent_runner,
        "_fleet_server_url",
        lambda _server: "https://fleet.example.invalid/mcp",
    )
    monkeypatch.setattr(
        agent_runner, "_configured_model_for_class", lambda _class: _configured_model()
    )

    config = agent_runner._build_execution_config(
        object(), "code-enhancer", _remote_meta()
    )

    assert config["mcp_toolsets"]
    headers = _toolset_transport_headers(config["mcp_toolsets"][0])
    assert "Authorization" not in headers
