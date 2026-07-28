"""A spawned agent inherits service-account auth for remote MCP toolsets.

CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap / OS-5.32 — `execute_agent` resolves a fleet Server and binds
its remote (SSE/streamable-HTTP) toolset. Those toolsets must carry the same
service-account identity the multiplexer attaches to its children, or a
jwt-protected `*.arpa` server rejects the call `401`. These tests pin that the
refresh-capable auth object (built via `client_credentials.child_auth`) reaches
the toolset's transport, and that the path is inert/safe when auth is disabled.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent_utilities.orchestration import agent_runner


def _toolset_transport_auth(toolset: Any) -> Any:
    """Extract the auth provider a built MCP toolset will present.

    pydantic-ai v2's ``MCPToolset`` wraps an ``fastmcp`` ``Client`` whose
    ``transport`` (``StreamableHttpTransport``/``SSETransport``) holds the
    auth provider; the httpx client is built lazily at connect time, so we
    assert against the transport.
    """
    transport = getattr(getattr(toolset, "client", None), "transport", None)
    return getattr(transport, "auth", None)


def test_spawn_auth_returns_refresh_capable_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = object()
    monkeypatch.setattr(
        "agent_utilities.mcp.client_credentials.child_auth",
        lambda _existing: auth,
    )
    assert agent_runner._spawn_auth() is auth


def test_spawn_auth_inert_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent_utilities.mcp.client_credentials.child_auth", lambda _existing: None
    )
    assert agent_runner._spawn_auth() is None


def test_spawn_auth_fails_closed_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_existing: Any) -> Any:
        raise RuntimeError("auth configuration failed")

    monkeypatch.setattr("agent_utilities.mcp.client_credentials.child_auth", _boom)
    with pytest.raises(RuntimeError, match="auth configuration failed"):
        agent_runner._spawn_auth()


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
        api_key_ref=None,
    )


def test_remote_toolset_carries_refresh_capable_auth_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = object()
    monkeypatch.setattr(
        "agent_utilities.mcp.client_credentials.child_auth",
        lambda _existing: auth,
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
    assert _toolset_transport_auth(config["mcp_toolsets"][0]) is auth


def test_remote_toolset_has_no_auth_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent_utilities.mcp.client_credentials.child_auth", lambda _existing: None
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
    assert _toolset_transport_auth(config["mcp_toolsets"][0]) is None
