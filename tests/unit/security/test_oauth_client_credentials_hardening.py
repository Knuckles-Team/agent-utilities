from __future__ import annotations

from types import SimpleNamespace
from unittest import mock
from urllib.parse import parse_qs

import httpx
import pytest

from agent_utilities.core.config import AgentConfig
from agent_utilities.core.transport_security import tls_environment_from_config
from agent_utilities.security.oauth_client_credentials import (
    OAuth2ClientCredentialsAuth,
    OAuth2ClientCredentialsConfig,
    OAuthClientCredentialsProvider,
    get_client_credentials_provider,
    reset_client_credentials_cache,
)


@pytest.fixture(autouse=True)
def _clear_provider_cache():
    reset_client_credentials_cache()
    yield
    reset_client_credentials_cache()


def _provider_with_responses(
    *payloads: dict,
) -> tuple[OAuthClientCredentialsProvider, list]:
    pending = iter(payloads)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=next(pending))

    provider = OAuthClientCredentialsProvider(
        "https://identity.example/token", "client", "runtime-secret"
    )
    provider._http = httpx.Client(transport=httpx.MockTransport(handler))  # noqa: SLF001
    return provider, requests


def test_token_mint_is_bounded_form_post_and_cached():
    provider, requests = _provider_with_responses(
        {"access_token": "opaque-token", "expires_in": 300}
    )
    assert provider.get_token() == "opaque-token"
    assert provider.get_token() == "opaque-token"
    assert len(requests) == 1
    form = parse_qs(requests[0].content.decode("utf-8"))
    assert form == {
        "client_id": ["client"],
        "client_secret": ["runtime-secret"],
        "grant_type": ["client_credentials"],
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"token_url": "http://identity.example/token"},
        {"token_url": "https://user@identity.example/token"},
        {"verify": False},
        {"extra_params": {"grant_type": "password"}},
        {"client_secret": "plaintext-secret"},
    ],
)
def test_oauth_config_rejects_unsafe_transport_or_secret(overrides):
    values = {
        "token_url": "https://identity.example/token",
        "client_id": "client",
        "client_secret": "env://OAUTH_CLIENT_SECRET",
        **overrides,
    }
    with pytest.raises(ValueError):
        OAuth2ClientCredentialsConfig(**values)


def test_agent_config_carries_service_tls_profiles():
    cfg = AgentConfig(
        MODEL_TLS_PROFILE="model-runtime",
        MODEL_TLS_PROFILE_REF="env://MODEL_TLS_PROFILE",
        EMBEDDING_TLS_PROFILE="embedding-runtime",
        EMBEDDING_TLS_PROFILE_REF="secret://EMBEDDING_TLS_PROFILE",
        OAUTH2_TOKEN_TLS_PROFILE="token-runtime",
        OAUTH2_TOKEN_TLS_PROFILE_REF="vault://TOKEN_TLS_PROFILE",
    )
    environment = tls_environment_from_config(cfg)
    assert environment["MODEL_TLS_PROFILE"] == "model-runtime"
    assert environment["EMBEDDING_TLS_PROFILE"] == "embedding-runtime"
    assert environment["OAUTH2_TOKEN_TLS_PROFILE"] == "token-runtime"
    assert all("BEGIN CERTIFICATE" not in value for value in environment.values())


@pytest.mark.parametrize(
    (
        "provider_tls",
        "global_profile",
        "global_ref",
        "expected_profile",
        "expected_ref",
    ),
    [
        (
            {"tls_profile_ref": "env://IDENTITY_TLS_PROFILE"},
            "global-token-profile",
            None,
            None,
            "env://IDENTITY_TLS_PROFILE",
        ),
        (
            {"tls_profile": "identity-token-profile"},
            None,
            "env://GLOBAL_TOKEN_TLS_PROFILE",
            "identity-token-profile",
            None,
        ),
    ],
)
def test_provider_tls_selector_atomically_overrides_global_selector(
    provider_tls,
    global_profile,
    global_ref,
    expected_profile,
    expected_ref,
):
    runtime_config = SimpleNamespace(
        oauth2_token_tls_profile=global_profile,
        oauth2_token_tls_profile_ref=global_ref,
        model_http_allowed_private_hosts=[],
    )
    trust = SimpleNamespace(
        proxy_url=None,
        ssl_context=object(),
        cleanup=mock.Mock(),
    )
    client = mock.Mock()
    provider = OAuthClientCredentialsProvider(
        "https://identity.example/token",
        "client",
        "runtime-secret",
        **provider_tls,
    )

    with (
        mock.patch("agent_utilities.core.config.config", runtime_config),
        mock.patch(
            "agent_utilities.core.transport_security.resolve_configured_tls_profile",
            return_value=trust,
        ) as resolve_tls,
        mock.patch(
            "agent_utilities.core.http_client.create_http_client",
            return_value=client,
        ),
    ):
        assert provider._http_client() is client  # noqa: SLF001

    resolve_tls.assert_called_once_with(
        "oauth2-token",
        profile_name=expected_profile,
        profile_ref=expected_ref,
        config=runtime_config,
    )
    provider.close()


def test_auth_fails_closed_when_token_mint_fails():
    class BrokenProvider:
        def get_token(self, **_kwargs):
            raise RuntimeError("raw endpoint and secret-adjacent detail")

    auth = OAuth2ClientCredentialsAuth(BrokenProvider())
    flow = auth.auth_flow(httpx.Request("GET", "https://provider.example/v1"))
    with pytest.raises(httpx.ProtocolError, match="credential unavailable"):
        next(flow)


@pytest.mark.parametrize(
    "arguments",
    [
        ("https://identity.example/token", "client", "line\nbreak", {}),
        (
            "https://identity.example/token",
            "client",
            "runtime-secret",
            {"extra_params": {"grant_type": "password"}},
        ),
        (
            "https://identity.example/token",
            "client",
            "runtime-secret",
            {"tls_profile_ref": "configured/local/profile.json"},
        ),
    ],
)
def test_direct_provider_construction_revalidates_runtime_material(arguments):
    token_url, client_id, client_secret, keyword_arguments = arguments
    with pytest.raises(ValueError):
        OAuthClientCredentialsProvider(
            token_url,
            client_id,
            client_secret,
            **keyword_arguments,
        )


def test_provider_cache_isolated_by_audience_and_secret_rotation():
    first = get_client_credentials_provider(
        "https://identity.example/token",
        "client",
        "secret-one",
        audience="audience-one",
    )
    other_audience = get_client_credentials_provider(
        "https://identity.example/token",
        "client",
        "secret-one",
        audience="audience-two",
    )
    rotated_secret = get_client_credentials_provider(
        "https://identity.example/token",
        "client",
        "secret-two",
        audience="audience-one",
    )
    assert first is not other_audience
    assert first is not rotated_secret


def test_invalid_token_shape_and_expiry_fail_closed():
    provider, _ = _provider_with_responses(
        {"access_token": "contains\nnewline", "expires_in": 300}
    )
    with pytest.raises(ValueError, match="invalid access token"):
        provider.get_token()

    provider, _ = _provider_with_responses(
        {"access_token": "opaque-token", "expires_in": "not-a-number"}
    )
    with pytest.raises(ValueError, match="invalid expiry"):
        provider.get_token()
