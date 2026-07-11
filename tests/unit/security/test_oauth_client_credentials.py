"""Tests for the OAuth2 client-credentials token lifecycle (C7).

CONCEPT:AU-OS.identity.oauth2-client-credentials-lifecycle

Covers:
- Token mint parses access_token/expires_in from a mocked token endpoint (never a real network).
- The cache returns the same token within TTL and proactively refreshes before expiry, driven by
  an injected clock rather than real sleep.
- A 401 forces exactly one re-mint.
- Secret-reference resolution goes through the injected secrets resolver (asserted as CALLED),
  never a plaintext read.
- Config-level mutual exclusion between a static api_key/api_key_env and an oauth2 block.
- The bearer is actually attached to an outbound request (mocked httpx transport) — for the LLM
  client (model_factory.create_model), the embedding client (embedding_utilities), and the
  graph-os registry path (server.dependencies._build_model_from_registry).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent_utilities.core.embedding_utilities import (
    create_embedding_model as _real_create_embedding_model,
)
from agent_utilities.security.oauth_client_credentials import (
    OAuth2ClientCredentialsAuth,
    OAuth2ClientCredentialsConfig,
    OAuthClientCredentialsProvider,
    build_provider_from_config,
    get_client_credentials_provider,
    httpx_auth_from_config,
    reset_client_credentials_cache,
    resolve_effective_skew,
)

# The unit suite's ``tests/unit/conftest.py`` autouse fixture monkeypatches
# ``create_embedding_model`` to refuse any call (hermetic-network guard). The alias above was
# bound at import/collection time, BEFORE that per-test monkeypatch runs, so it keeps pointing at
# the real function — tests below call it directly to exercise the actual oauth2 wiring without
# fighting the guard or a live network (the token endpoint itself is still always mocked).


def _resp(token: str, ttl: int = 300, extra: dict | None = None) -> MagicMock:
    r = MagicMock()
    payload = {"access_token": token, "expires_in": ttl}
    if extra:
        payload.update(extra)
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_client_credentials_cache()
    yield
    reset_client_credentials_cache()


# ---------------------------------------------------------------------------
# Token mint — parses access_token/expires_in from a mocked endpoint
# ---------------------------------------------------------------------------


class TestMint:
    def test_mints_and_parses_access_token(self):
        provider = OAuthClientCredentialsProvider(
            "https://idp.example.com/oauth2/token",
            "client-a",
            "s3cr3t",
            scope="api://resource/.default",
        )
        with patch(
            "agent_utilities.security.oauth_client_credentials.requests.post",
            return_value=_resp("tok-abc", ttl=600),
        ) as post:
            token = provider.get_token()
        assert token == "tok-abc"
        assert provider.access_token_ttl == 600
        _, kwargs = post.call_args
        assert kwargs["data"]["grant_type"] == "client_credentials"
        assert kwargs["data"]["client_id"] == "client-a"
        assert kwargs["data"]["client_secret"] == "s3cr3t"
        assert kwargs["data"]["scope"] == "api://resource/.default"

    def test_audience_and_extra_params_sent(self):
        provider = OAuthClientCredentialsProvider(
            "https://idp.example.com/token",
            "client-a",
            "s3cr3t",
            audience="https://api.example.com",
            extra_params={"resource": "https://api.example.com"},
        )
        with patch(
            "agent_utilities.security.oauth_client_credentials.requests.post",
            return_value=_resp("tok"),
        ) as post:
            provider.get_token()
        _, kwargs = post.call_args
        assert kwargs["data"]["audience"] == "https://api.example.com"
        assert kwargs["data"]["resource"] == "https://api.example.com"

    def test_body_style_default_sends_no_basic_auth(self):
        """Default 'body' style keeps the historical behaviour: creds in the form body,
        no HTTP Basic ``auth`` on the token request."""
        provider = OAuthClientCredentialsProvider(
            "https://idp.example.com/token", "client-a", "s3cr3t"
        )
        with patch(
            "agent_utilities.security.oauth_client_credentials.requests.post",
            return_value=_resp("tok"),
        ) as post:
            provider.get_token()
        _, kwargs = post.call_args
        assert kwargs["auth"] is None
        assert kwargs["data"]["client_id"] == "client-a"
        assert kwargs["data"]["client_secret"] == "s3cr3t"

    def test_basic_style_uses_http_basic_and_omits_body_creds(self):
        """'basic' style (client_secret_basic) presents the credentials via HTTP Basic
        auth and keeps them OUT of the form body (RFC 6749 §2.3.1)."""
        provider = OAuthClientCredentialsProvider(
            "https://idp.example.com/token",
            "client-a",
            "s3cr3t",
            scope="api://resource/.default",
            token_auth_style="basic",
        )
        with patch(
            "agent_utilities.security.oauth_client_credentials.requests.post",
            return_value=_resp("tok-basic"),
        ) as post:
            token = provider.get_token()
        assert token == "tok-basic"
        _, kwargs = post.call_args
        assert kwargs["auth"] == ("client-a", "s3cr3t")
        assert "client_id" not in kwargs["data"]
        assert "client_secret" not in kwargs["data"]
        # non-credential params still ride in the body
        assert kwargs["data"]["grant_type"] == "client_credentials"
        assert kwargs["data"]["scope"] == "api://resource/.default"

    def test_missing_access_token_raises(self):
        provider = OAuthClientCredentialsProvider(
            "https://idp.example.com/token", "client-a", "s3cr3t"
        )
        bad = MagicMock()
        bad.json.return_value = {"token_type": "bearer"}  # no access_token
        bad.raise_for_status.return_value = None
        with (
            patch(
                "agent_utilities.security.oauth_client_credentials.requests.post",
                return_value=bad,
            ),
            pytest.raises(ValueError, match="access_token"),
        ):
            provider.get_token()


# ---------------------------------------------------------------------------
# Cache + proactive renewal — driven by an injected clock, never real sleep
# ---------------------------------------------------------------------------


class TestCacheAndRenewal:
    def test_returns_same_token_within_ttl(self):
        clock = {"t": 0.0}
        provider = OAuthClientCredentialsProvider(
            "https://idp/token",
            "client-a",
            "s3cr3t",
            clock=lambda: clock["t"],
        )
        with patch(
            "agent_utilities.security.oauth_client_credentials.requests.post",
            return_value=_resp("tok-1", ttl=300),
        ) as post:
            first = provider.get_token()
            clock["t"] += 50  # well within TTL - skew
            second = provider.get_token()
        assert first == second == "tok-1"
        post.assert_called_once()

    def test_refreshes_before_actual_expiry(self):
        """Skew defaults to max(60s, 20% of ttl); a 300s TTL renews at t=240 (60s early),
        strictly BEFORE the token's real 300s expiry — never waiting for a 401."""
        clock = {"t": 0.0}
        provider = OAuthClientCredentialsProvider(
            "https://idp/token",
            "client-a",
            "s3cr3t",
            clock=lambda: clock["t"],
        )
        with patch(
            "agent_utilities.security.oauth_client_credentials.requests.post",
            side_effect=[_resp("tok-1", ttl=300), _resp("tok-2", ttl=300)],
        ) as post:
            assert provider.get_token() == "tok-1"
            clock["t"] = 239.0  # inside the safe window still
            assert provider.get_token() == "tok-1"
            assert post.call_count == 1
            clock["t"] = 241.0  # past exp(300) - skew(60) = 240 -> must renew
            assert provider.get_token() == "tok-2"
        assert post.call_count == 2

    def test_explicit_skew_overrides_default(self):
        assert resolve_effective_skew(ttl_seconds=3600, explicit_skew=5.0) == 5.0
        # default: max(60, 20% of ttl)
        assert resolve_effective_skew(ttl_seconds=3600, explicit_skew=None) == 720.0
        assert resolve_effective_skew(ttl_seconds=120, explicit_skew=None) == 60.0

    def test_short_ttl_floors_skew_at_60s(self):
        clock = {"t": 0.0}
        provider = OAuthClientCredentialsProvider(
            "https://idp/token", "client-a", "s3cr3t", clock=lambda: clock["t"]
        )
        with patch(
            "agent_utilities.security.oauth_client_credentials.requests.post",
            side_effect=[_resp("tok-1", ttl=90), _resp("tok-2", ttl=90)],
        ) as post:
            provider.get_token()
            clock["t"] = 29.0  # ttl(90) - skew(60) = 30 -> still cached
            provider.get_token()
            assert post.call_count == 1
            clock["t"] = 31.0
            provider.get_token()
        assert post.call_count == 2


# ---------------------------------------------------------------------------
# 401 forces exactly one force-refresh
# ---------------------------------------------------------------------------


class TestForceRefreshOn401:
    def test_get_token_force_bypasses_cache(self):
        provider = OAuthClientCredentialsProvider("https://idp/token", "a", "s")
        with patch(
            "agent_utilities.security.oauth_client_credentials.requests.post",
            side_effect=[_resp("tok-1"), _resp("tok-2")],
        ) as post:
            assert provider.get_token() == "tok-1"
            assert provider.get_token() == "tok-1"  # cached, no second call
            assert provider.get_token(force=True) == "tok-2"
        assert post.call_count == 2

    def test_auth_flow_remints_once_on_401(self):
        provider = OAuthClientCredentialsProvider("https://idp/token", "a", "s")
        with patch(
            "agent_utilities.security.oauth_client_credentials.requests.post",
            side_effect=[_resp("tok-1"), _resp("tok-2")],
        ):
            auth = OAuth2ClientCredentialsAuth(provider)
            request = httpx.Request(
                "POST", "http://llm.example.com/v1/chat/completions"
            )
            flow = auth.auth_flow(request)
            first = next(flow)
            assert first.headers["Authorization"] == "Bearer tok-1"
            retried = flow.send(httpx.Response(401, request=first))
            assert retried.headers["Authorization"] == "Bearer tok-2"
            with pytest.raises(StopIteration):
                flow.send(httpx.Response(200, request=retried))

    def test_auth_flow_no_remint_on_success(self):
        provider = OAuthClientCredentialsProvider("https://idp/token", "a", "s")
        with patch(
            "agent_utilities.security.oauth_client_credentials.requests.post",
            return_value=_resp("tok-1"),
        ) as post:
            auth = OAuth2ClientCredentialsAuth(provider)
            request = httpx.Request(
                "POST", "http://llm.example.com/v1/chat/completions"
            )
            flow = auth.auth_flow(request)
            sent = next(flow)
            with pytest.raises(StopIteration):
                flow.send(httpx.Response(200, request=sent))
        post.assert_called_once()


# ---------------------------------------------------------------------------
# Secret-reference resolution — the injected resolver is called, never a plaintext read
# ---------------------------------------------------------------------------


class TestSecretResolution:
    def test_build_provider_uses_injected_resolver(self):
        fake_secrets = MagicMock()
        fake_secrets.resolve_ref.side_effect = lambda ref: {
            "vault://llm/azure/client_secret": "the-real-secret",
        }.get(ref)
        cfg = OAuth2ClientCredentialsConfig(
            token_url="https://login.microsoftonline.com/tenant/oauth2/v2.0/token",
            client_id="my-client-id",
            client_secret="vault://llm/azure/client_secret",
            scope="api://resource/.default",
        )
        provider = build_provider_from_config(cfg, secrets=fake_secrets)
        fake_secrets.resolve_ref.assert_called_once_with(
            "vault://llm/azure/client_secret"
        )
        # The provider carries the RESOLVED value, not the ref string.
        assert provider._client_secret == "the-real-secret"  # noqa: SLF001 - white-box check

    def test_sensitive_client_id_also_resolved_via_ref(self):
        fake_secrets = MagicMock()
        fake_secrets.resolve_ref.side_effect = lambda ref: {
            "env://LLM_CLIENT_ID": "resolved-client-id",
            "vault://llm/client_secret": "resolved-secret",
        }.get(ref)
        cfg = OAuth2ClientCredentialsConfig(
            token_url="https://idp/token",
            client_id="env://LLM_CLIENT_ID",
            client_secret="vault://llm/client_secret",
        )
        provider = build_provider_from_config(cfg, secrets=fake_secrets)
        assert fake_secrets.resolve_ref.call_count == 2
        assert provider.client_id == "resolved-client-id"

    def test_unresolved_secret_raises(self):
        fake_secrets = MagicMock()
        fake_secrets.resolve_ref.return_value = None
        cfg = OAuth2ClientCredentialsConfig(
            token_url="https://idp/token",
            client_id="a",
            client_secret="vault://missing/secret",
        )
        with pytest.raises(ValueError, match="did not resolve"):
            build_provider_from_config(cfg, secrets=fake_secrets)

    def test_plaintext_client_secret_rejected_at_validation(self):
        with pytest.raises(ValueError, match="secret reference"):
            OAuth2ClientCredentialsConfig(
                token_url="https://idp/token",
                client_id="a",
                client_secret="example-plaintext-secret",  # not a ref -> must reject
            )

    def test_literal_client_id_not_resolved(self):
        """A non-sensitive client_id (no '://') is used as-is — no resolver call for it."""
        fake_secrets = MagicMock()
        fake_secrets.resolve_ref.side_effect = lambda ref: {
            "vault://llm/client_secret": "resolved-secret",
        }.get(ref)
        cfg = OAuth2ClientCredentialsConfig(
            token_url="https://idp/token",
            client_id="literal-client-id",
            client_secret="vault://llm/client_secret",
        )
        provider = build_provider_from_config(cfg, secrets=fake_secrets)
        assert provider.client_id == "literal-client-id"
        fake_secrets.resolve_ref.assert_called_once_with("vault://llm/client_secret")


# ---------------------------------------------------------------------------
# Config-level mutual exclusion: api_key vs oauth2
# ---------------------------------------------------------------------------


class TestMutualExclusion:
    def test_chat_model_config_rejects_both(self):
        from agent_utilities.core.config import ChatModelConfig

        with pytest.raises(ValueError, match="mutually exclusive"):
            ChatModelConfig(
                id="m1",
                provider="openai",
                api_key="sk-plain",
                oauth2={
                    "token_url": "https://idp/token",
                    "client_id": "a",
                    "client_secret": "vault://x/y",
                },
            )

    def test_embedding_model_config_rejects_both(self):
        from agent_utilities.core.config import EmbeddingModelConfig

        with pytest.raises(ValueError, match="mutually exclusive"):
            EmbeddingModelConfig(
                id="e1",
                provider="openai",
                api_key="sk-plain",
                oauth2={
                    "token_url": "https://idp/token",
                    "client_id": "a",
                    "client_secret": "vault://x/y",
                },
            )

    def test_chat_model_config_accepts_oauth2_alone(self):
        from agent_utilities.core.config import ChatModelConfig

        cfg = ChatModelConfig(
            id="m1",
            provider="openai",
            oauth2={
                "token_url": "https://idp/token",
                "client_id": "a",
                "client_secret": "vault://x/y",
            },
        )
        assert cfg.oauth2["token_url"] == "https://idp/token"

    def test_chat_model_config_rejects_plaintext_secret_in_oauth2(self):
        from agent_utilities.core.config import ChatModelConfig

        with pytest.raises(ValueError):
            ChatModelConfig(
                id="m1",
                provider="openai",
                oauth2={
                    "token_url": "https://idp/token",
                    "client_id": "a",
                    "client_secret": "plaintext-oops",
                },
            )

    def test_model_definition_rejects_both(self):
        from agent_utilities.models.model_registry import ModelDefinition

        with pytest.raises(ValueError, match="mutually exclusive"):
            ModelDefinition(
                id="m1",
                name="M1",
                provider="openai",
                model_id="gpt-4o-mini",
                api_key_env="SOME_KEY",
                oauth2={
                    "token_url": "https://idp/token",
                    "client_id": "a",
                    "client_secret": "vault://x/y",
                },
            )


# ---------------------------------------------------------------------------
# The bearer is actually attached to an outbound request — LLM, embedding, graph-os path
# ---------------------------------------------------------------------------


OAUTH2_BLOCK = {
    "token_url": "https://idp.example.com/oauth2/v2.0/token",
    "client_id": "client-a",
    "client_secret": "vault://llm/client_secret",
    "scope": "api://resource/.default",
}


@pytest.fixture
def fake_secrets(monkeypatch):
    """Patch ``create_secrets_client`` so oauth wiring never touches a live engine/Vault."""
    fake = MagicMock()
    fake.resolve_ref.side_effect = lambda ref: {
        "vault://llm/client_secret": "the-real-secret",
    }.get(ref)
    monkeypatch.setattr(
        "agent_utilities.security.secrets_client.create_secrets_client",
        lambda: fake,
    )
    return fake


class TestBearerAttachedToLLMRequest:
    def test_httpx_auth_from_config_attaches_bearer(self, fake_secrets):
        auth = httpx_auth_from_config(OAUTH2_BLOCK)
        assert auth is not None
        fake_secrets.resolve_ref.assert_called_once_with("vault://llm/client_secret")

        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth_header"] = request.headers.get("Authorization")
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        with (
            patch(
                "agent_utilities.security.oauth_client_credentials.requests.post",
                return_value=_resp("llm-bearer-tok"),
            ),
            httpx.Client(transport=transport, auth=auth) as client,
        ):
            resp = client.get("https://llm.example.com/v1/chat/completions")
        assert resp.status_code == 200
        assert captured["auth_header"] == "Bearer llm-bearer-tok"

    def test_llm_client_via_model_factory(self, fake_secrets, monkeypatch):
        """model_factory.create_model wires the oauth2 bearer onto the AsyncOpenAI client's
        underlying http_client — verified by inspecting the auth object it builds."""
        monkeypatch.delenv("AGENT_UTILITIES_TESTING", raising=False)
        from agent_utilities.core import model_factory

        with patch(
            "agent_utilities.security.oauth_client_credentials.requests.post",
            return_value=_resp("llm-bearer-tok"),
        ):
            model = model_factory.create_model(
                provider="openai",
                model_id="gpt-4o-mini",
                base_url="https://llm.example.com/v1",
                oauth2=OAUTH2_BLOCK,
            )
        # A model was built (not a TestModel passthrough) and the underlying provider's
        # http client carries our OAuth2 auth object.
        http_client = model.client._client  # AsyncOpenAI -> httpx.AsyncClient
        assert isinstance(http_client.auth, OAuth2ClientCredentialsAuth)

    def test_llm_client_rejects_api_key_and_oauth2_together(
        self, fake_secrets, monkeypatch
    ):
        monkeypatch.delenv("AGENT_UTILITIES_TESTING", raising=False)
        from agent_utilities.core import model_factory

        with pytest.raises(ValueError, match="mutually exclusive"):
            model_factory.create_model(
                provider="openai",
                model_id="gpt-4o-mini",
                api_key="sk-plain",
                oauth2=OAUTH2_BLOCK,
            )


class TestBearerAttachedToEmbeddingRequest:
    def test_embedding_client_carries_oauth2_auth(self, fake_secrets):
        from agent_utilities.core import embedding_utilities

        embedding_utilities.clear_embedding_model_cache()
        with patch(
            "agent_utilities.security.oauth_client_credentials.requests.post",
            return_value=_resp("embed-bearer-tok"),
        ):
            mdl = _real_create_embedding_model(
                provider="openai",
                model="text-embedding-3-small",
                base_url="https://embed.example.com/v1",
                oauth2=OAUTH2_BLOCK,
            )
        assert mdl._http_client is not None  # noqa: SLF001 - white-box check
        assert isinstance(mdl._http_client.auth, OAuth2ClientCredentialsAuth)
        assert mdl._async_http_client is not None  # noqa: SLF001
        assert isinstance(mdl._async_http_client.auth, OAuth2ClientCredentialsAuth)
        embedding_utilities.clear_embedding_model_cache()

    def test_embedding_client_rejects_api_key_and_oauth2_together(self, fake_secrets):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _real_create_embedding_model(
                provider="openai",
                model="text-embedding-3-small",
                api_key="sk-plain",
                oauth2=OAUTH2_BLOCK,
            )


class TestGraphOSRegistryPath:
    def test_build_model_from_registry_threads_oauth2(self, fake_secrets, monkeypatch):
        """server.dependencies._build_model_from_registry (the graph-os path that historically
        only carried api_key_env) threads a ModelDefinition's oauth2 block into create_model."""
        monkeypatch.delenv("AGENT_UTILITIES_TESTING", raising=False)
        from agent_utilities.models import ModelDefinition, ModelRegistry
        from agent_utilities.server.dependencies import _build_model_from_registry

        registry = ModelRegistry(
            models=[
                ModelDefinition(
                    id="azure-gpt4",
                    name="Azure GPT-4",
                    provider="openai",
                    model_id="gpt-4o",
                    base_url="https://azure.example.com/v1",
                    oauth2=OAUTH2_BLOCK,
                    is_default=True,
                )
            ]
        )
        with patch(
            "agent_utilities.security.oauth_client_credentials.requests.post",
            return_value=_resp("azure-bearer-tok"),
        ):
            model = _build_model_from_registry(registry, "azure-gpt4")
        assert model is not None
        http_client = model.client._client
        assert isinstance(http_client.auth, OAuth2ClientCredentialsAuth)


# ---------------------------------------------------------------------------
# Never log the token or the secret
# ---------------------------------------------------------------------------


class TestNoSecretLogging:
    def test_mint_log_line_excludes_token_and_secret(self, caplog):
        provider = OAuthClientCredentialsProvider(
            "https://idp/token", "client-a", "top-secret-value"
        )
        with (
            patch(
                "agent_utilities.security.oauth_client_credentials.requests.post",
                return_value=_resp("super-secret-bearer-token"),
            ),
            caplog.at_level("DEBUG"),
        ):
            provider.get_token()
        combined = "\n".join(r.getMessage() for r in caplog.records)
        assert "super-secret-bearer-token" not in combined
        assert "top-secret-value" not in combined


# ---------------------------------------------------------------------------
# Keyed provider cache
# ---------------------------------------------------------------------------


class TestKeyedCache:
    def test_same_key_returns_same_provider(self):
        p1 = get_client_credentials_provider("https://idp/token", "a", "s1", scope="x")
        p2 = get_client_credentials_provider("https://idp/token", "a", "s2", scope="x")
        assert p1 is p2  # keyed on (token_url, client_id, scope) only

    def test_different_scope_returns_different_provider(self):
        p1 = get_client_credentials_provider("https://idp/token", "a", "s", scope="x")
        p2 = get_client_credentials_provider("https://idp/token", "a", "s", scope="y")
        assert p1 is not p2
