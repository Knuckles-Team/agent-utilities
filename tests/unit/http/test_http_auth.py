"""Tests for agent_utilities.http.auth strategies (CONCEPT:ECO-4.35)."""

from __future__ import annotations

import base64

import pytest

from agent_utilities.http.auth import (
    AuthHeaderInjector,
    BasicAuth,
    QueryApiKeyAuth,
    TokenAuth,
)


def test_base_injector_is_anonymous():
    auth = AuthHeaderInjector()
    assert auth.headers() == {}
    assert auth.params() == {}
    assert auth.secrets() == []


def test_token_auth_bearer_default():
    auth = TokenAuth("tok123")
    assert auth.headers() == {"Authorization": "Bearer tok123"}
    assert auth.secrets() == ["tok123"]


def test_token_auth_ssws_prefix():
    auth = TokenAuth("okta-token", prefix="SSWS")
    assert auth.headers() == {"Authorization": "SSWS okta-token"}


def test_token_auth_bare_header_no_prefix():
    auth = TokenAuth("apikey-xyz", header="X-API-Key", prefix=None)
    assert auth.headers() == {"X-API-Key": "apikey-xyz"}


def test_token_auth_provider_is_consulted_per_request():
    calls = {"n": 0}

    def provider() -> str:
        calls["n"] += 1
        return f"minted-{calls['n']}"

    auth = TokenAuth(token_provider=provider)
    assert auth.headers() == {"Authorization": "Bearer minted-1"}
    assert auth.headers() == {"Authorization": "Bearer minted-2"}
    # Provider-backed tokens are not minted just for redaction registration.
    assert auth.secrets() == []
    assert calls["n"] == 2


def test_token_auth_requires_exactly_one_credential_source():
    with pytest.raises(ValueError, match="exactly one"):
        TokenAuth()
    with pytest.raises(ValueError, match="exactly one"):
        TokenAuth("tok", token_provider=lambda: "tok")


def test_basic_auth_builds_rfc7617_header():
    auth = BasicAuth("alice", "wonder land")
    expected = base64.b64encode(b"alice:wonder land").decode("ascii")
    assert auth.headers() == {"Authorization": f"Basic {expected}"}
    assert auth.secrets() == ["wonder land"]


def test_query_api_key_auth_injects_param_only():
    auth = QueryApiKeyAuth("api_key", "k-123456")
    assert auth.params() == {"api_key": "k-123456"}
    assert auth.headers() == {}
    assert auth.secrets() == ["k-123456"]


def test_query_api_key_auth_validates_inputs():
    with pytest.raises(ValueError):
        QueryApiKeyAuth("", "key")
    with pytest.raises(ValueError):
        QueryApiKeyAuth("param", "")
