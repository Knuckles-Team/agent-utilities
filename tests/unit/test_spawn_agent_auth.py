"""A spawned agent inherits service-account auth for remote MCP toolsets.

CONCEPT:ORCH-1.21 / OS-5.32 — `execute_agent` resolves a fleet Server and binds
its remote (SSE/streamable-HTTP) toolset. Those toolsets must carry the same
service-account bearer the multiplexer attaches to its children, or a
jwt-protected `*.arpa` server rejects the call `401`. These tests pin that the
bearer (minted via `client_credentials.bearer_header`) reaches the toolset's
httpx client, and that the path is inert/safe when auth is disabled.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from agent_utilities.orchestration import agent_runner


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


class _RecordingClient(httpx.AsyncClient):
    """httpx.AsyncClient subclass that records the headers it was built with."""

    captured: dict[str, Any] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _RecordingClient.captured = dict(kwargs.get("headers") or {})
        super().__init__(*args, **kwargs)


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
    monkeypatch.setattr("httpx.AsyncClient", _RecordingClient)
    _RecordingClient.captured = {}

    config = agent_runner._build_execution_config(
        object(), "code-enhancer", _remote_meta()
    )

    assert config["mcp_toolsets"], "a remote toolset should be bound"
    assert _RecordingClient.captured.get("Authorization") == "Bearer TESTTOKEN"


def test_remote_toolset_no_bearer_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent_utilities.mcp.client_credentials.bearer_header", lambda _existing: {}
    )
    monkeypatch.setattr("httpx.AsyncClient", _RecordingClient)
    _RecordingClient.captured = {"sentinel": "unset"}

    config = agent_runner._build_execution_config(
        object(), "code-enhancer", _remote_meta()
    )

    assert config["mcp_toolsets"]
    assert "Authorization" not in _RecordingClient.captured
