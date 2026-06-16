"""Tests for the multiplexer OIDC client-credentials token provider (A0)."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import httpx
import pytest


@pytest.fixture
def cc(monkeypatch):
    """Fresh module with provider-state reset, fully configured + enabled."""
    monkeypatch.setenv("MCP_CLIENT_AUTH", "oidc-client-credentials")
    monkeypatch.setenv("OIDC_CLIENT_ID", "mcp-multiplexer")
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "s3cr3t")
    monkeypatch.setenv("OIDC_AUDIENCE", "agent-services")
    monkeypatch.setenv(
        "OIDC_TOKEN_URL",
        "http://keycloak.arpa/realms/master/protocol/openid-connect/token",
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


# ── Per-request auth (the long-lived-session fix) ──────────────────────────


def test_get_token_force_bypasses_cache(cc):
    with patch.object(
        cc.requests, "post", side_effect=[_resp("tok-1"), _resp("tok-2")]
    ) as post:
        provider = cc.get_provider()
        assert provider.get_token() == "tok-1"
        assert provider.get_token() == "tok-1"  # cached
        assert provider.get_token(force=True) == "tok-2"  # cache bypassed
    assert post.call_count == 2


def test_bearer_auth_returns_auth_when_enabled(cc):
    with patch.object(cc.requests, "post", return_value=_resp("tok")):
        auth = cc.bearer_auth(None)
    assert isinstance(auth, cc.ClientCredentialsAuth)


def test_bearer_auth_none_when_disabled(monkeypatch):
    monkeypatch.setenv("MCP_CLIENT_AUTH", "none")
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    assert module.bearer_auth(None) is None


def test_bearer_auth_respects_explicit_authorization(cc):
    with patch.object(cc.requests, "post", return_value=_resp("tok")) as post:
        assert cc.bearer_auth({"Authorization": "Bearer child-own"}) is None
    post.assert_not_called()


def test_auth_flow_injects_bearer_and_remints_on_401(cc):
    """The per-request flow keeps a long-lived session authenticated: it mints
    on every request and re-mints once when the child answers 401 (expired/
    rotated token) — the exact wedge a frozen session header caused."""
    with patch.object(
        cc.requests, "post", side_effect=[_resp("tok-1"), _resp("tok-2")]
    ):
        auth = cc.bearer_auth(None)
        request = httpx.Request("POST", "http://child.arpa/mcp")
        flow = auth.auth_flow(request)
        first = next(flow)
        assert first.headers["Authorization"] == "Bearer tok-1"
        # Child rejects the (now-expired) token: flow re-mints and retries once.
        retried = flow.send(httpx.Response(401, request=first))
        assert retried.headers["Authorization"] == "Bearer tok-2"
        with pytest.raises(StopIteration):
            flow.send(httpx.Response(200, request=retried))


def test_auth_flow_no_retry_on_success(cc):
    with patch.object(cc.requests, "post", side_effect=[_resp("tok-1")]) as post:
        auth = cc.bearer_auth(None)
        request = httpx.Request("POST", "http://child.arpa/mcp")
        flow = auth.auth_flow(request)
        sent = next(flow)
        assert sent.headers["Authorization"] == "Bearer tok-1"
        with pytest.raises(StopIteration):
            flow.send(httpx.Response(200, request=sent))
    post.assert_called_once()  # no re-mint on a non-401 response


# ── Session-max-age (recycle-before-expiry) derivation ─────────────────────


def test_service_session_max_age_from_token_ttl(cc):
    with patch.object(cc.requests, "post", return_value=_resp("tok", ttl=60)):
        age = cc.service_session_max_age(None)
    # 60s TTL - 30s skew - 5s buffer = 25s
    assert age == 25.0


def test_service_session_max_age_floored(cc):
    with patch.object(cc.requests, "post", return_value=_resp("tok", ttl=10)):
        age = cc.service_session_max_age(None)
    assert age == cc._MIN_SESSION_MAX_AGE  # never thrash on a tiny TTL


def test_service_session_max_age_none_for_own_auth(cc):
    with patch.object(cc.requests, "post", return_value=_resp("tok")) as post:
        assert cc.service_session_max_age({"Authorization": "Bearer own"}) is None
    post.assert_not_called()


def test_service_session_max_age_none_when_disabled(monkeypatch):
    monkeypatch.setenv("MCP_CLIENT_AUTH", "none")
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    assert module.service_session_max_age(None) is None
