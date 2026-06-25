"""Tests for the universal CredentialProvider (OS-5.38) + typed credentials (OS-5.39)."""

from __future__ import annotations

import base64
import json
import time

import pytest

from agent_utilities.security.credential_provider import (
    CredentialProvider,
    get_credential_provider,
)
from agent_utilities.security.secrets_client import (
    InEpistemicGraphBackend,
    SecretsClient,
)
from agent_utilities.security.source_credentials import (
    ApiKeyCredential,
    AuthMaterial,
    BasicAuthCredential,
    CookieSessionCredential,
    NoCredential,
    OAuth2Credential,
    build_credential,
)


@pytest.fixture
def secrets() -> SecretsClient:
    """An engine-backed secrets client (unique throwaway graph) seeded with a few
    source secrets. The unique graph keeps each test isolated (CONCEPT:OS-5.66)."""
    import uuid

    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    graph = GraphComputeEngine(graph_name=f"__secrets_test_{uuid.uuid4().hex[:12]}__")
    client = SecretsClient(backend=InEpistemicGraphBackend(graph=graph))
    client.set("pulselink/x/session", json.dumps({"auth_token": "AT", "ct0": "C0"}))
    client.set("pulselink/reddit/cs", "reddit-client-secret")
    client.set("github/token", "ghp_abc123")
    return client


# --------------------------------------------------------------------------- #
# Typed credential shapes (OS-5.39)
# --------------------------------------------------------------------------- #


def test_no_credential_is_present_but_empty() -> None:
    cred = NoCredential()
    assert cred.is_present() is True  # keyless backends always qualify
    mat = cred.materialize()
    assert mat.headers == {} and mat.params == {} and mat.cookies == {}


def test_api_key_header_default_bearer(secrets: SecretsClient) -> None:
    cred = build_credential(
        {"type": "api_key", "secret": "vault://github/token"}, secrets
    )
    assert isinstance(cred, ApiKeyCredential)
    assert cred.is_present()
    assert cred.materialize().headers == {"Authorization": "Bearer ghp_abc123"}


def test_api_key_custom_prefix_and_query_placement(secrets: SecretsClient) -> None:
    header_cred = build_credential(
        {"type": "api_key", "secret": "vault://github/token", "prefix": "token "},
        secrets,
    )
    assert header_cred.materialize().headers == {"Authorization": "token ghp_abc123"}

    query_cred = build_credential(
        {
            "type": "api_key",
            "secret": "vault://github/token",
            "placement": "query",
            "name": "apikey",
            "prefix": "",
        },
        secrets,
    )
    mat = query_cred.materialize()
    assert mat.params == {"apikey": "ghp_abc123"}
    assert mat.headers == {}


def test_api_key_absent_is_not_present() -> None:
    cred = build_credential({"type": "api_key", "secret": "env://NOPE_MISSING"}, None)
    assert cred.is_present() is False
    assert cred.materialize().headers == {}


def test_cookie_session_from_json_blob(secrets: SecretsClient) -> None:
    cred = build_credential(
        {"type": "cookie_session", "secret": "vault://pulselink/x/session"}, secrets
    )
    assert isinstance(cred, CookieSessionCredential)
    assert cred.is_present()
    mat = cred.materialize()
    assert mat.cookies == {"auth_token": "AT", "ct0": "C0"}
    assert mat.headers["Cookie"] == "auth_token=AT; ct0=C0"


def test_cookie_session_from_cookie_string() -> None:
    import uuid

    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    graph = GraphComputeEngine(graph_name=f"__secrets_test_{uuid.uuid4().hex[:12]}__")
    client = SecretsClient(backend=InEpistemicGraphBackend(graph=graph))
    client.set("s", "reddit_session=abc; token=xyz")
    cred = build_credential({"type": "cookie_session", "secret": "vault://s"}, client)
    assert cred.materialize().cookies == {"reddit_session": "abc", "token": "xyz"}


def test_cookie_session_inline() -> None:
    cred = build_credential(
        {"type": "cookie_session", "cookies": {"a": "1"}}, None
    )
    assert cred.materialize().cookies == {"a": "1"}


def test_basic_auth(secrets: SecretsClient) -> None:
    cred = build_credential(
        {
            "type": "basic",
            "username": "user",
            "password_secret": "vault://github/token",
        },
        secrets,
    )
    assert isinstance(cred, BasicAuthCredential)
    expected = base64.b64encode(b"user:ghp_abc123").decode()
    assert cred.materialize().headers == {"Authorization": f"Basic {expected}"}


def test_oauth2_static_access_token() -> None:
    cred = OAuth2Credential(access_token="live-token")
    assert cred.is_present()
    assert cred.materialize().headers == {"Authorization": "Bearer live-token"}


def test_oauth2_refreshes_when_expired(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"access_token": "fresh", "expires_in": 3600}

    def _fake_post(url: str, **kw: object) -> _Resp:
        posted["url"] = url
        posted["data"] = kw.get("data")
        return _Resp()

    import requests

    monkeypatch.setattr(requests, "post", _fake_post)

    cred = OAuth2Credential(
        access_token=None,
        refresh_token="rt",
        token_url="https://idp/token",
        client_id="cid",
        client_secret="csecret",
        expires_at=time.monotonic() - 1,  # already expired
    )
    mat = cred.materialize()
    assert mat.headers == {"Authorization": "Bearer fresh"}
    assert posted["url"] == "https://idp/token"
    assert posted["data"]["grant_type"] == "refresh_token"  # type: ignore[index]


def test_oauth2_failed_refresh_degrades_to_no_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*a: object, **k: object) -> object:
        raise RuntimeError("idp down")

    import requests

    monkeypatch.setattr(requests, "post", _boom)
    cred = OAuth2Credential(
        access_token=None,
        refresh_token="rt",
        token_url="https://idp/token",
        client_id="cid",
        expires_at=time.monotonic() - 1,
    )
    assert cred.materialize().headers == {}  # never raises


def test_build_credential_unknown_type_raises() -> None:
    with pytest.raises(KeyError):
        build_credential({"type": "quantum_entanglement"}, None)


def test_auth_material_merged_into() -> None:
    mat = AuthMaterial(headers={"Authorization": "Bearer x"}, params={"k": "v"})
    h, p, c = mat.merged_into(headers={"Accept": "json"})
    assert h == {"Accept": "json", "Authorization": "Bearer x"}
    assert p == {"k": "v"}
    assert c == {}


# --------------------------------------------------------------------------- #
# Provider (OS-5.38)
# --------------------------------------------------------------------------- #


def test_provider_unknown_source_is_keyless(secrets: SecretsClient) -> None:
    provider = CredentialProvider(descriptors={}, secrets=secrets)
    cred = provider.get("youtube")  # keyless source, no descriptor
    assert isinstance(cred, NoCredential)
    # No real credential → cookie/official backends stay dark; keyless backend
    # (which declares NoCredential) is still eligible and applies nothing.
    assert provider.available("youtube") is False
    assert provider.get("youtube").materialize().headers == {}


def test_provider_available_gates_ladder(secrets: SecretsClient) -> None:
    provider = CredentialProvider(
        descriptors={
            "x": {"type": "cookie_session", "secret": "vault://pulselink/x/session"},
            "linkedin": {"type": "api_key", "secret": "env://ABSENT_LINKEDIN"},
        },
        secrets=secrets,
    )
    # X has a seeded cookie → its higher-fidelity backend lights up.
    assert provider.available("x") is True
    # LinkedIn's secret is absent → backend stays dark; ladder falls back to keyless.
    assert provider.available("linkedin") is False


def test_provider_caches_and_returns_live_object(secrets: SecretsClient) -> None:
    provider = CredentialProvider(
        descriptors={"github": {"type": "api_key", "secret": "vault://github/token"}},
        secrets=secrets,
    )
    first = provider.get("github")
    second = provider.get("github")
    assert first is second  # cached live object (OAuth refresh stays effective)


def test_provider_register_at_runtime(secrets: SecretsClient) -> None:
    provider = CredentialProvider(descriptors={}, secrets=secrets)
    assert provider.available("github") is False
    provider.register("github", {"type": "api_key", "secret": "vault://github/token"})
    assert provider.available("github") is True


def test_provider_bad_descriptor_falls_back_to_keyless(secrets: SecretsClient) -> None:
    provider = CredentialProvider(
        descriptors={"weird": {"type": "does_not_exist"}}, secrets=secrets
    )
    assert isinstance(provider.get("weird"), NoCredential)


def test_provider_status_never_leaks_secret_values(secrets: SecretsClient) -> None:
    provider = CredentialProvider(
        descriptors={
            "x": {"type": "cookie_session", "secret": "vault://pulselink/x/session"},
            "linkedin": {"type": "api_key", "secret": "env://ABSENT"},
        },
        secrets=secrets,
    )
    status = provider.status()
    assert status == {
        "x": {"type": "cookie_session", "available": True},
        "linkedin": {"type": "api_key", "available": False},
    }
    # No secret material anywhere in the serialized status.
    assert "AT" not in json.dumps(status)


def test_provider_loads_descriptors_from_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SOURCE_CREDENTIALS",
        json.dumps({"github": {"type": "api_key", "secret": "env://GH_TT"}}),
    )
    monkeypatch.setenv("GH_TT", "ghp_fromenv")
    provider = CredentialProvider()  # reads SOURCE_CREDENTIALS + default secrets client
    assert provider.available("github") is True
    assert provider.get("github").materialize().headers == {
        "Authorization": "Bearer ghp_fromenv"
    }


def test_get_credential_provider_singleton() -> None:
    a = get_credential_provider()
    b = get_credential_provider()
    assert a is b
