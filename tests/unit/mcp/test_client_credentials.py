"""Tests for the multiplexer OIDC client-credentials token provider (A0)."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def cc(monkeypatch):
    """Fresh module with provider-state reset, fully configured + enabled."""
    monkeypatch.setenv("MCP_CLIENT_AUTH", "oidc-client-credentials")
    monkeypatch.setenv("OIDC_CLIENT_ID", "mcp-multiplexer")
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "s3cr3t")
    monkeypatch.setenv("OIDC_AUDIENCE", "agent-services")
    monkeypatch.setenv(
        "OIDC_TOKEN_URL", "http://keycloak.arpa/realms/master/protocol/openid-connect/token"
    )
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)  # reset module-level provider cache
    return module


def _resp(token: str, ttl: int = 300):
    r = MagicMock()
    r.json.return_value = {"access_token": token, "expires_in": ttl}
    r.raise_for_status.return_value = None
    return r


def test_disabled_returns_none(monkeypatch):
    monkeypatch.setenv("MCP_CLIENT_AUTH", "none")
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    assert module.get_provider() is None
    assert module.bearer_header(None) == {}


def test_mints_and_attaches_bearer(cc):
    with patch.object(cc.requests, "post", return_value=_resp("tok-abc")) as post:
        header = cc.bearer_header(None)
    assert header == {"Authorization": "Bearer tok-abc"}
    post.assert_called_once()
    # client_credentials grant + audience sent
    _, kwargs = post.call_args
    assert kwargs["data"]["grant_type"] == "client_credentials"
    assert kwargs["data"]["audience"] == "agent-services"
    assert kwargs["auth"] == ("mcp-multiplexer", "s3cr3t")


def test_token_is_cached(cc):
    with patch.object(cc.requests, "post", return_value=_resp("tok-1")) as post:
        cc.bearer_header(None)
        cc.bearer_header(None)
    post.assert_called_once()  # second call served from cache


def test_does_not_override_explicit_authorization(cc):
    with patch.object(cc.requests, "post", return_value=_resp("tok")) as post:
        assert cc.bearer_header({"Authorization": "Bearer child-own"}) == {}
    post.assert_not_called()


def test_mint_failure_degrades_to_no_header(cc):
    with patch.object(cc.requests, "post", side_effect=RuntimeError("keycloak down")):
        assert cc.bearer_header(None) == {}


def test_missing_secret_disables(monkeypatch):
    monkeypatch.setenv("MCP_CLIENT_AUTH", "oidc-client-credentials")
    monkeypatch.setenv("OIDC_CLIENT_ID", "mcp-multiplexer")
    monkeypatch.delenv("OIDC_CLIENT_SECRET", raising=False)
    monkeypatch.setenv("OIDC_TOKEN_URL", "http://kc/token")
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    assert module.get_provider() is None
