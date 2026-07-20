"""TLS-contract tests for the one remote MCP toolset factory."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent_utilities.mcp.toolset_factory import (
    _httpx_client_factory,
    build_http_toolset,
)


def test_remote_mcp_factory_uses_resolved_tls_context(monkeypatch) -> None:
    context = object()
    trust = SimpleNamespace(
        httpx_kwargs=lambda: {"verify": context, "trust_env": False}
    )
    create = MagicMock(return_value=object())
    monkeypatch.setattr(
        "agent_utilities.core.http_client.create_async_http_client", create
    )

    _httpx_client_factory(trust, 12.0)(headers={"X-Test": "1"})

    kwargs = create.call_args.kwargs
    assert kwargs["verify"] is context
    assert kwargs["trust_env"] is False
    assert kwargs["headers"] == {"X-Test": "1"}


def test_remote_mcp_api_has_no_boolean_verify_control() -> None:
    parameters = inspect.signature(build_http_toolset).parameters
    assert "verify" not in parameters
    assert {"tls_service", "tls_profile", "tls_profile_ref"} <= set(parameters)


def test_remote_mcp_factory_accepts_refreshing_auth() -> None:
    parameters = inspect.signature(build_http_toolset).parameters
    assert "auth" in parameters
