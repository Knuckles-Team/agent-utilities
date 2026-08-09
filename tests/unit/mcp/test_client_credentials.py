"""Tests for the multiplexer OIDC client-credentials token provider (A0)."""

from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest


@pytest.fixture
def cc(monkeypatch):
    """Fresh module with provider-state reset, fully configured + enabled."""
    monkeypatch.setenv("MCP_CLIENT_AUTH", "oidc-client-credentials")
    monkeypatch.setenv("OIDC_CLIENT_ID", "mcp-multiplexer")
    monkeypatch.setenv("OIDC_CLIENT_SECRET_REF", "env://TEST_OIDC_CLIENT_SECRET")
    monkeypatch.setenv("TEST_OIDC_CLIENT_SECRET", "s3cr3t")
    monkeypatch.setenv("OIDC_AUDIENCE", "graph-api")
    monkeypatch.setenv(
        "OIDC_TOKEN_URL",
        "https://identity.example.test/oauth2/token",
    )
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)  # reset module-level provider cache
    return module


def _resp(token: str, ttl: int = 300):
    r = MagicMock()
    r.__enter__.return_value = r
    r.__exit__.return_value = False
    r.iter_bytes.return_value = [
        json.dumps({"access_token": token, "expires_in": ttl}).encode()
    ]
    r.raise_for_status.return_value = None
    return r


def _oidc_http_client(*responses):
    client = MagicMock()
    client.stream.side_effect = list(responses)
    context = MagicMock()
    context.__enter__.return_value = client
    context.__exit__.return_value = False
    return context, client


def test_disabled_returns_none(monkeypatch):
    monkeypatch.setenv("MCP_CLIENT_AUTH", "none")
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    assert module.get_provider() is None
    assert module.child_auth_header(None) == {}


def test_mints_and_attaches_bearer(cc):
    context, client = _oidc_http_client(_resp("tok-abc"))
    with patch(
        "agent_utilities.security.oidc_discovery.oidc_http_client",
        return_value=context,
    ):
        header = cc.child_auth_header(None)
    assert header == {"Authorization": "Bearer tok-abc"}
    client.stream.assert_called_once()
    # client_credentials grant + audience sent
    _, kwargs = client.stream.call_args
    assert kwargs["data"]["grant_type"] == "client_credentials"
    assert kwargs["data"]["audience"] == "graph-api"
    assert kwargs["auth"] == ("mcp-multiplexer", "s3cr3t")


def test_token_is_cached(cc):
    context, client = _oidc_http_client(_resp("tok-1"))
    with patch(
        "agent_utilities.security.oidc_discovery.oidc_http_client",
        return_value=context,
    ):
        cc.child_auth_header(None)
        cc.child_auth_header(None)
    client.stream.assert_called_once()  # second call served from cache


def test_does_not_override_explicit_authorization(cc):
    context, client = _oidc_http_client(_resp("tok"))
    with patch(
        "agent_utilities.security.oidc_discovery.oidc_http_client",
        return_value=context,
    ):
        assert cc.child_auth_header({"Authorization": "Bearer child-own"}) == {}
    client.stream.assert_not_called()


def test_mint_failure_fails_closed(cc):
    with patch(
        "agent_utilities.security.oidc_discovery.oidc_http_client",
        side_effect=RuntimeError("identity provider unavailable"),
    ):
        with pytest.raises(
            RuntimeError, match="Could not mint outbound MCP credential"
        ):
            cc.child_auth_header(None)


def test_missing_secret_fails_closed(monkeypatch):
    monkeypatch.setenv("MCP_CLIENT_AUTH", "oidc-client-credentials")
    monkeypatch.setenv("OIDC_CLIENT_ID", "mcp-multiplexer")
    monkeypatch.delenv("OIDC_CLIENT_SECRET_REF", raising=False)
    monkeypatch.setenv("OIDC_AUDIENCE", "graph-api")
    monkeypatch.setenv("OIDC_TOKEN_URL", "https://identity.example.test/token")
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    with pytest.raises(RuntimeError, match="service identity is incomplete"):
        module.get_provider()


def test_env_secret_ref_resolves_without_secret_backend(monkeypatch):
    monkeypatch.setenv("OIDC_CLIENT_SECRET_REF", "env://TEST_OIDC_CLIENT_SECRET")
    monkeypatch.setenv("TEST_OIDC_CLIENT_SECRET", "process-local-secret")
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    with patch(
        "agent_utilities.security.secrets_client.create_secrets_client",
        side_effect=AssertionError("environment references must not open a backend"),
    ):
        assert (
            module._resolve_runtime_secret("OIDC_CLIENT_SECRET_REF")
            == "process-local-secret"
        )


def test_env_secret_ref_rejects_non_environment_target(monkeypatch):
    monkeypatch.setenv("OIDC_CLIENT_SECRET_REF", "env://NOT/A/VARIABLE")
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    with pytest.raises(RuntimeError, match="credential reference is invalid"):
        module._resolve_runtime_secret("OIDC_CLIENT_SECRET_REF")


def test_missing_audience_fails_closed(monkeypatch):
    monkeypatch.setenv("MCP_CLIENT_AUTH", "oidc-client-credentials")
    monkeypatch.setenv("OIDC_CLIENT_ID", "mcp-multiplexer")
    monkeypatch.setenv("OIDC_CLIENT_SECRET_REF", "env://TEST_OIDC_CLIENT_SECRET")
    monkeypatch.setenv("TEST_OIDC_CLIENT_SECRET", "runtime-secret")
    monkeypatch.setenv("OIDC_TOKEN_URL", "https://identity.example.test/token")
    monkeypatch.delenv("OIDC_AUDIENCE", raising=False)
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    with pytest.raises(RuntimeError, match="service identity is incomplete"):
        module.get_provider()


def test_raw_secret_has_no_legacy_fallback(monkeypatch):
    monkeypatch.setenv("MCP_CLIENT_AUTH", "oidc-client-credentials")
    monkeypatch.setenv("OIDC_CLIENT_ID", "mcp-multiplexer")
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "legacy-value")
    monkeypatch.delenv("OIDC_CLIENT_SECRET_REF", raising=False)
    monkeypatch.setenv("OIDC_AUDIENCE", "graph-api")
    monkeypatch.setenv("OIDC_TOKEN_URL", "https://identity.example.test/token")
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    with pytest.raises(RuntimeError, match="service identity is incomplete"):
        module.get_provider()


def test_token_url_discovery_uses_only_explicit_outbound_issuer(monkeypatch):
    monkeypatch.delenv("OIDC_TOKEN_URL", raising=False)
    monkeypatch.delenv("OIDC_ISSUER", raising=False)
    monkeypatch.setenv("FASTMCP_SERVER_AUTH_JWT_ISSUER", "https://inbound.example.test")
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    assert module._derive_token_url() is None

    monkeypatch.setenv("OIDC_ISSUER", "https://outbound.example.test")
    with patch(
        "agent_utilities.security.oidc_discovery.token_endpoint_for",
        return_value="https://outbound.example.test/token",
    ) as discover:
        assert module._derive_token_url() == "https://outbound.example.test/token"
    discover.assert_called_once_with("https://outbound.example.test")


# ── Per-request auth (the long-lived-session fix) ──────────────────────────


def test_get_token_force_bypasses_cache(cc):
    context, client = _oidc_http_client(_resp("tok-1"), _resp("tok-2"))
    with patch(
        "agent_utilities.security.oidc_discovery.oidc_http_client",
        return_value=context,
    ):
        provider = cc.get_provider()
        assert provider.get_token() == "tok-1"
        assert provider.get_token() == "tok-1"  # cached
        assert provider.get_token(force=True) == "tok-2"  # cache bypassed
    assert client.stream.call_count == 2


def test_bearer_auth_returns_auth_when_enabled(cc):
    auth = cc.child_auth(None)
    assert isinstance(auth, cc.ClientCredentialsAuth)


def test_bearer_auth_none_when_disabled(monkeypatch):
    monkeypatch.setenv("MCP_CLIENT_AUTH", "none")
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    assert module.child_auth(None) is None


def test_bearer_auth_respects_explicit_authorization(cc):
    assert cc.child_auth({"Authorization": "Bearer child-own"}) is None


def test_auth_flow_injects_bearer_and_remints_on_401(cc):
    """The per-request flow keeps a long-lived session authenticated: it mints
    on every request and re-mints once when the child answers 401 (expired/
    rotated token) — the exact wedge a frozen session header caused."""
    context, _client = _oidc_http_client(_resp("tok-1"), _resp("tok-2"))
    with patch(
        "agent_utilities.security.oidc_discovery.oidc_http_client",
        return_value=context,
    ):
        auth = cc.child_auth(None)
        request = httpx.Request("POST", "https://child.example.test/mcp")
        flow = auth.auth_flow(request)
        first = next(flow)
        assert first.headers["Authorization"] == "Bearer tok-1"
        # Child rejects the (now-expired) token: flow re-mints and retries once.
        retried = flow.send(httpx.Response(401, request=first))
        assert retried.headers["Authorization"] == "Bearer tok-2"
        with pytest.raises(StopIteration):
            flow.send(httpx.Response(200, request=retried))


def test_auth_flow_no_retry_on_success(cc):
    context, client = _oidc_http_client(_resp("tok-1"))
    with patch(
        "agent_utilities.security.oidc_discovery.oidc_http_client",
        return_value=context,
    ):
        auth = cc.child_auth(None)
        request = httpx.Request("POST", "https://child.example.test/mcp")
        flow = auth.auth_flow(request)
        sent = next(flow)
        assert sent.headers["Authorization"] == "Bearer tok-1"
        with pytest.raises(StopIteration):
            flow.send(httpx.Response(200, request=sent))
    client.stream.assert_called_once()  # no re-mint on a non-401 response


# ── Session-max-age (recycle-before-expiry) derivation ─────────────────────


def test_service_session_max_age_from_token_ttl(cc):
    context, _client = _oidc_http_client(_resp("tok", ttl=60))
    with patch(
        "agent_utilities.security.oidc_discovery.oidc_http_client",
        return_value=context,
    ):
        age = cc.service_session_max_age(None)
    # 60s TTL - 30s skew - 5s buffer = 25s
    assert age == 25.0


def test_service_session_max_age_floored(cc):
    context, _client = _oidc_http_client(_resp("tok", ttl=10))
    with patch(
        "agent_utilities.security.oidc_discovery.oidc_http_client",
        return_value=context,
    ):
        age = cc.service_session_max_age(None)
    assert age == cc._MIN_SESSION_MAX_AGE  # never thrash on a tiny TTL


def test_service_session_max_age_none_for_own_auth(cc):
    assert cc.service_session_max_age({"Authorization": "Bearer own"}) is None


def test_service_session_max_age_none_when_disabled(monkeypatch):
    monkeypatch.setenv("MCP_CLIENT_AUTH", "none")
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    assert module.service_session_max_age(None) is None


# ── HTTP Basic scheme (MCP_CLIENT_AUTH=basic) ──────────────────────────────


@pytest.fixture
def basic(monkeypatch):
    """Fresh module configured for the static HTTP Basic scheme."""
    monkeypatch.setenv("MCP_CLIENT_AUTH", "basic")
    monkeypatch.setenv("MCP_BASIC_AUTH_USERNAME", "svc")
    monkeypatch.setenv(
        "MCP_BASIC_AUTH_PASSWORD_REF", "env://TEST_MCP_BASIC_AUTH_PASSWORD"
    )
    monkeypatch.setenv("TEST_MCP_BASIC_AUTH_PASSWORD", "p@ss")
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    return module


def test_basic_header_is_base64_of_user_pass(basic):
    import base64

    header = basic.child_auth_header(None)
    expected = base64.b64encode(b"svc:p@ss").decode()
    assert header == {"Authorization": f"Basic {expected}"}


def test_basic_mints_no_oidc_token(basic):
    # Basic is static: no token endpoint is ever called, and no OIDC provider exists.
    with patch(
        "agent_utilities.security.oidc_discovery.oidc_http_client"
    ) as oidc_client:
        basic.child_auth_header(None)
    oidc_client.assert_not_called()
    assert basic.get_provider() is None


def test_basic_child_auth_is_httpx_basic_auth(basic):
    auth = basic.child_auth(None)
    assert isinstance(auth, httpx.BasicAuth)


def test_basic_does_not_override_explicit_authorization(basic):
    assert basic.child_auth_header({"Authorization": "Basic child-own"}) == {}
    assert basic.child_auth({"Authorization": "Basic child-own"}) is None


def test_basic_missing_credentials_fails_closed(monkeypatch):
    monkeypatch.setenv("MCP_CLIENT_AUTH", "basic")
    monkeypatch.setenv("MCP_BASIC_AUTH_USERNAME", "svc")
    monkeypatch.delenv("MCP_BASIC_AUTH_PASSWORD_REF", raising=False)
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    with pytest.raises(RuntimeError, match="basic identity is incomplete"):
        module.child_auth_header(None)
    with pytest.raises(RuntimeError, match="basic identity is incomplete"):
        module.child_auth(None)


def test_basic_session_max_age_is_none_static_credential(basic):
    # A static Basic credential never expires — no forced session recycle.
    assert basic.service_session_max_age(None) is None


def test_unknown_mode_fails_closed(monkeypatch):
    monkeypatch.setenv("MCP_CLIENT_AUTH", "totally-bogus")
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    with pytest.raises(RuntimeError, match="unsupported value"):
        module._auth_mode()
    with pytest.raises(RuntimeError, match="unsupported value"):
        module.child_auth_header(None)
    with pytest.raises(RuntimeError, match="unsupported value"):
        module.child_auth(None)


# ── Rotating file bearer (MCP_CLIENT_AUTH=rotating-file-bearer, BUG-051) ───
#
# The scheme exists because a token minted by an OUT-OF-PROCESS refresh daemon
# (e.g. services/graphos-token-refresh/refresh-graphos-token.sh, cron-driven,
# outside this process's control) rotates a FILE, not this process's memory.
# A header baked in once at connect time goes stale the moment that file
# rotates; RotatingFileBearerAuth re-reads it on every request instead.


def _write_token(path, token: str) -> None:
    path.write_text(token + "\n", encoding="utf-8")
    path.chmod(0o600)


@pytest.fixture
def rotating(monkeypatch, tmp_path):
    """Fresh module configured for the rotating-file-bearer scheme, with a
    valid mode-0600 token file already in place."""
    token_path = tmp_path / "access-token"
    _write_token(token_path, "tok-initial")
    monkeypatch.setenv("MCP_CLIENT_AUTH", "rotating-file-bearer")
    monkeypatch.setenv("MCP_BEARER_TOKEN_FILE", str(token_path))
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    return module, token_path


def test_rotating_header_reads_current_file_contents(rotating):
    module, token_path = rotating
    assert module.child_auth_header(None) == {"Authorization": "Bearer tok-initial"}


def test_rotating_header_picks_up_rotation_between_calls(rotating):
    """No memoization anywhere in the read path: two calls straddling a file
    rewrite must see two different tokens, with no module/provider reload and
    no reconnect of anything — this is the whole mechanism BUG-051 needs."""
    module, token_path = rotating
    assert module.child_auth_header(None)["Authorization"] == "Bearer tok-initial"
    _write_token(token_path, "tok-rotated")
    assert module.child_auth_header(None)["Authorization"] == "Bearer tok-rotated"


def test_rotating_mints_no_oidc_token(rotating):
    module, _token_path = rotating
    with patch(
        "agent_utilities.security.oidc_discovery.oidc_http_client"
    ) as oidc_client:
        module.child_auth_header(None)
    oidc_client.assert_not_called()
    assert module.get_provider() is None


def test_rotating_child_auth_is_rotating_file_bearer_auth(rotating):
    module, token_path = rotating
    auth = module.child_auth(None)
    assert isinstance(auth, module.RotatingFileBearerAuth)


def test_rotating_does_not_override_explicit_authorization(rotating):
    module, _token_path = rotating
    assert module.child_auth_header({"Authorization": "Bearer child-own"}) == {}
    assert module.child_auth({"Authorization": "Bearer child-own"}) is None


def test_rotating_missing_file_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_CLIENT_AUTH", "rotating-file-bearer")
    monkeypatch.setenv("MCP_BEARER_TOKEN_FILE", str(tmp_path / "does-not-exist"))
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    with pytest.raises(RuntimeError, match="unavailable"):
        module.child_auth_header(None)


def test_rotating_missing_path_config_fails_closed(monkeypatch):
    monkeypatch.setenv("MCP_CLIENT_AUTH", "rotating-file-bearer")
    monkeypatch.delenv("MCP_BEARER_TOKEN_FILE", raising=False)
    import agent_utilities.mcp.client_credentials as module

    importlib.reload(module)
    with pytest.raises(RuntimeError, match="incomplete"):
        module.child_auth_header(None)


def test_rotating_rejects_group_readable_file(rotating):
    """Mirrors ``graphos-codex-bridge.py``'s own ``_read_token`` mode check —
    a token file readable by anyone but its owner is refused outright rather
    than trusted."""
    module, token_path = rotating
    token_path.chmod(0o640)
    with pytest.raises(RuntimeError, match="permissive"):
        module.child_auth_header(None)


def test_rotating_rejects_empty_file(rotating):
    module, token_path = rotating
    token_path.write_text("", encoding="utf-8")
    token_path.chmod(0o600)
    with pytest.raises(RuntimeError, match="invalid"):
        module.child_auth_header(None)


def test_rotating_session_max_age_is_none_no_baked_in_expiry(rotating):
    # Nothing is baked in to age out — every request re-reads the file.
    module, _token_path = rotating
    assert module.service_session_max_age(None) is None


def test_rotating_auth_flow_reads_fresh_on_every_request_no_reconnect(rotating):
    """The direct analogue of ``test_auth_flow_injects_bearer_and_remints_on_401``
    for the OIDC scheme: proves the SAME ``httpx.Auth`` instance — i.e. the
    SAME long-lived child session, never reconstructed — presents a DIFFERENT
    bearer on its second request after the file rotates in between. This is
    in-band renewal: no reconnect, no new ``RotatingFileBearerAuth``, no new
    session, just the next request reading fresher disk state."""
    module, token_path = rotating
    auth = module.child_auth(None)  # ONE auth object for the whole "session"

    first_request = httpx.Request("POST", "https://graph-os.example/mcp")
    first_flow = auth.auth_flow(first_request)
    sent = next(first_flow)
    assert sent.headers["Authorization"] == "Bearer tok-initial"
    with pytest.raises(StopIteration):
        first_flow.send(httpx.Response(200, request=sent))

    # The out-of-process daemon rotates the file. Nothing here reconnects,
    # reloads the module, or touches ``auth`` at all.
    _write_token(token_path, "tok-rotated")

    second_request = httpx.Request("POST", "https://graph-os.example/mcp")
    second_flow = auth.auth_flow(second_request)
    sent_again = next(second_flow)
    assert sent_again.headers["Authorization"] == "Bearer tok-rotated"


def test_rotating_auth_flow_401_forces_a_fresh_read_and_retries(rotating):
    module, token_path = rotating
    auth = module.child_auth(None)
    request = httpx.Request("POST", "https://graph-os.example/mcp")
    flow = auth.auth_flow(request)
    first = next(flow)
    assert first.headers["Authorization"] == "Bearer tok-initial"
    # The child rejects it (expired). Between the read and the 401 landing,
    # the daemon rotates the file — the retry must see the NEW value, not
    # silently repeat the one that just failed.
    _write_token(token_path, "tok-rotated")
    retried = flow.send(httpx.Response(401, request=first))
    assert retried.headers["Authorization"] == "Bearer tok-rotated"
    with pytest.raises(StopIteration):
        flow.send(httpx.Response(200, request=retried))


def test_bug_051_expired_then_refreshed_token_succeeds_without_reconnect(tmp_path):
    """THE negative test for BUG-051, end to end over real HTTP request/response
    objects (``httpx.MockTransport`` standing in for the graph-os gateway),
    simulating expiry rather than waiting out a real ~10h TTL.

    BEFORE (the reported bug, reproduced): a client that bakes its bearer into
    a STATIC header at connect time — exactly what ``~/.claude.json``'s
    ``mcpServers.graph-os.headers.Authorization`` does today, and exactly what
    the incident report described ("file refreshed + probe 200, session still
    401") — keeps sending the stale token forever after the daemon rotates the
    file. The daemon's write is real and correct; nothing downstream ever
    reads it.

    AFTER (the fix): the SAME long-lived ``httpx.Client`` — never rebuilt,
    never reconnected — authenticated with ``RotatingFileBearerAuth`` instead
    succeeds on its very next request once the token file is rotated, because
    it re-reads the file per request rather than trusting a header set once at
    construction time.
    """
    from agent_utilities.mcp.client_credentials import RotatingFileBearerAuth

    token_path = tmp_path / "access-token"
    _write_token(token_path, "tok-old")
    valid_token = {"current": "tok-old"}

    def handler(request: httpx.Request) -> httpx.Response:
        presented = request.headers.get("authorization", "").removeprefix("Bearer ")
        if presented == valid_token["current"]:
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(
            401,
            json={"error": "token expired"},
            headers={"www-authenticate": "Bearer"},
        )

    # ── BEFORE: a static header, baked in once at "connect" time ───────────
    static_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": f"Bearer {valid_token['current']}"},
    )
    before_first = static_client.get("https://graph-os.example/mcp")
    assert before_first.status_code == 200  # session starts out authenticated

    # The daemon does its job: mints a new token, rotates the file. The
    # refresh itself is not in question — the SERVER-side half already works.
    valid_token["current"] = "tok-new"
    _write_token(token_path, "tok-new")

    # The static client was never told. No code path re-reads the file for it.
    before_second = static_client.get("https://graph-os.example/mcp")
    assert before_second.status_code == 401  # <- the reported outage, reproduced
    static_client.close()

    # ── AFTER: the same shape of long-lived client, rotating-file-bearer ───
    valid_token["current"] = "tok-old"
    _write_token(token_path, "tok-old")
    rotating_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        auth=RotatingFileBearerAuth(token_path),
    )
    after_first = rotating_client.get("https://graph-os.example/mcp")
    assert after_first.status_code == 200

    # Same rotation, same lack of any explicit reconnect call.
    valid_token["current"] = "tok-new"
    _write_token(token_path, "tok-new")

    after_second = rotating_client.get("https://graph-os.example/mcp")
    assert after_second.status_code == 200  # <- BUG-051 closed: no reconnect
    rotating_client.close()
