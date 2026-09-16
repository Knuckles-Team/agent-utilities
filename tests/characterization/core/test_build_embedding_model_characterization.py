"""Characterization test for ``_build_embedding_model`` (CX-AU-07).

Pins the OBSERVED behaviour of the current, native-HTTP function
(``agent_utilities/core/embedding_utilities.py``) — no llama-index import
anywhere (D2: GHSA-8mgp-746c-j5xp, nltk had no patched release and was
pulled in transitively via llama-index-core). Rewritten from the
llama-index-era version of this test, which characterized SDK classes
(``OpenAIEmbedding``/``HuggingFaceEmbedding``/``OllamaEmbedding``) that no
longer exist in this module.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.core import embedding_utilities as eu


def _config(**overrides):
    base = dict(
        embedding_tls_profile=None,
        embedding_tls_profile_ref=None,
        tls_system_trust=True,
        tls_trust_env=True,
        model_http_allowed_private_hosts=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch):
    monkeypatch.setattr(eu, "config", _config())


def test_openai_provider_builds_native_http_embedding_model():
    model = eu._build_embedding_model(
        provider_str="openai",
        model_str="text-embedding-3-small",
        base_url_str="https://embed.invalid/v1",
        api_key_str="k",
        timeout=30.0,
        provider="openai",
    )
    assert isinstance(model, eu._HttpEmbeddingModel)
    assert model.model_name == "text-embedding-3-small"
    assert str(model._client.base_url) == "https://embed.invalid/v1/"
    assert model._client.headers["authorization"] == "Bearer k"


def test_openai_provider_defaults_base_url_when_unconfigured():
    model = eu._build_embedding_model(
        provider_str="openai",
        model_str="m",
        base_url_str=None,
        api_key_str="k",
        timeout=30.0,
        provider="openai",
    )
    assert str(model._client.base_url) == "https://api.openai.com/v1/"


def test_huggingface_provider_redirects_to_data_science_mcp():
    with pytest.raises(ValueError, match="data-science-mcp"):
        eu._build_embedding_model(
            provider_str="huggingface",
            model_str="BAAI/bge-small-en",
            base_url_str=None,
            api_key_str=None,
            timeout=30.0,
            provider="huggingface",
        )


def test_local_provider_redirects_to_data_science_mcp():
    with pytest.raises(ValueError, match="data-science-mcp"):
        eu._build_embedding_model(
            provider_str="local",
            model_str="local-model",
            base_url_str=None,
            api_key_str=None,
            timeout=30.0,
            provider="local",
        )


def test_ollama_provider_requires_base_url():
    with pytest.raises(ValueError, match="Ollama embedding endpoint is not configured"):
        eu._build_embedding_model(
            provider_str="ollama",
            model_str="llama3",
            base_url_str=None,
            api_key_str=None,
            timeout=30.0,
            provider="ollama",
        )


def test_ollama_provider_builds_native_http_embedding_model():
    model = eu._build_embedding_model(
        provider_str="ollama",
        model_str="llama3",
        base_url_str="http://ollama.invalid:11434",
        api_key_str="tok",
        timeout=30.0,
        provider="ollama",
    )
    assert isinstance(model, eu._HttpEmbeddingModel)
    assert model.model_name == "llama3"
    assert str(model._client.base_url) == "http://ollama.invalid:11434"
    # Authorization header injected from api_key_str.
    assert model._client.headers["authorization"] == "Bearer tok"


def test_ollama_provider_does_not_overwrite_explicit_authorization_header():
    model = eu._build_embedding_model(
        provider_str="ollama",
        model_str="llama3",
        base_url_str="http://ollama.invalid:11434",
        api_key_str="tok",
        timeout=30.0,
        provider="ollama",
        headers={"Authorization": "Bearer explicit-header-wins"},
    )
    assert model._client.headers["authorization"] == "Bearer explicit-header-wins"


def test_unsupported_provider_raises_with_original_provider_name_in_message():
    # Observed: the error message uses the ORIGINAL `provider` arg, not the
    # lower-cased/resolved `provider_str`.
    with pytest.raises(ValueError, match="Unsupported embedding provider: WeirdCasing"):
        eu._build_embedding_model(
            provider_str="weirdcasing",
            model_str="m",
            base_url_str=None,
            api_key_str=None,
            timeout=30.0,
            provider="WeirdCasing",
        )


def test_oauth2_auth_is_resolved_and_attached_to_openai_http_client(monkeypatch):
    calls = {}

    def fake_httpx_auth_from_config(oauth2_cfg):
        calls["oauth2_cfg"] = oauth2_cfg
        return "SENTINEL_AUTH"

    monkeypatch.setattr(
        "agent_utilities.security.oauth_client_credentials.httpx_auth_from_config",
        fake_httpx_auth_from_config,
    )
    captured = {}
    real_create_http_client = eu.create_http_client

    def spy_create_http_client(**kwargs):
        captured["auth"] = kwargs.get("auth")
        # avoid httpx's strict auth-type validation by not forwarding the
        # sentinel further -- this test only needs to prove the VALUE reached
        # create_http_client, not build a real client with it.
        kwargs["auth"] = None
        return real_create_http_client(**kwargs)

    monkeypatch.setattr(eu, "create_http_client", spy_create_http_client)

    eu._build_embedding_model(
        provider_str="openai",
        model_str="m",
        base_url_str="https://embed.invalid/v1",
        api_key_str="k",
        timeout=30.0,
        provider="openai",
        oauth2_cfg={"token_url": "https://idp.invalid/token", "client_id": "c"},
    )
    assert calls["oauth2_cfg"]["client_id"] == "c"
    assert captured["auth"] == "SENTINEL_AUTH"


def test_proxy_incompatible_tls_profile_raises(monkeypatch):
    from agent_utilities.core.transport_security import TransportSecurityError

    fake_profile = SimpleNamespace(
        proxy_url="http://proxy.invalid:8080", ssl_context=None
    )
    monkeypatch.setattr(
        eu,
        "config",
        _config(embedding_tls_profile="proxied", embedding_tls_profile_ref=None),
    )
    monkeypatch.setattr(
        "agent_utilities.core.transport_security.resolve_configured_tls_profile",
        lambda *a, **kw: fake_profile,
    )
    with pytest.raises(TransportSecurityError):
        eu._build_embedding_model(
            provider_str="openai",
            model_str="m",
            base_url_str="https://embed.invalid/v1",
            api_key_str="k",
            timeout=30.0,
            provider="openai",
        )
