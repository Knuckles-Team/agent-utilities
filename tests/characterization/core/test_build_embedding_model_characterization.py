"""Characterization test for ``_build_embedding_model`` (CX-AU-07).

Pins the OBSERVED behaviour of the pre-refactor function (CCN 15,
``agent_utilities/core/embedding_utilities.py``) before it is decomposed
into per-concern helper functions (oauth2 auth resolution, TLS profile
resolution, the openai http-client pair, and one builder per provider).
This test must be green against the unmodified function; if it is not,
the test is wrong, not the code. It is not touched again in the refactor
commit.

Every SDK embedding class this function constructs is monkeypatched at
its IMPORT SOURCE (the deferred ``from llama_index.embeddings.X import Y``
inside the function grabs the patched attribute) or, for Ollama, at the
``embedding_utilities`` module level where it is imported eagerly. This
keeps the test hermetic -- in particular it never lets a real
``HuggingFaceEmbedding`` attempt a model download.
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


class _FakeOpenAIEmbedding:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeHuggingFaceEmbedding:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeInnerClient:
    def __init__(self, name):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True

    async def aclose(self):
        self.closed = True


class _FakeWrappedClient:
    def __init__(self, name):
        self._client = _FakeInnerClient(name)


class _FakeOllamaEmbedding:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._client = _FakeWrappedClient("sync")
        self._async_client = _FakeWrappedClient("async")


@pytest.fixture(autouse=True)
def _patch_sdk_classes(monkeypatch):
    monkeypatch.setattr(eu, "config", _config())
    monkeypatch.setattr(
        "llama_index.embeddings.openai.OpenAIEmbedding", _FakeOpenAIEmbedding
    )
    monkeypatch.setattr(
        "llama_index.embeddings.huggingface.HuggingFaceEmbedding",
        _FakeHuggingFaceEmbedding,
    )
    monkeypatch.setattr(eu, "OllamaEmbedding", _FakeOllamaEmbedding)


def test_openai_provider_builds_with_both_sync_and_async_http_clients():
    model = eu._build_embedding_model(
        provider_str="openai",
        model_str="text-embedding-3-small",
        base_url_str="https://embed.invalid/v1",
        api_key_str="k",
        timeout=30.0,
        provider="openai",
    )
    assert isinstance(model, _FakeOpenAIEmbedding)
    assert model.kwargs["model_name"] == "text-embedding-3-small"
    assert model.kwargs["api_key"] == "k"
    assert model.kwargs["http_client"] is not None
    assert model.kwargs["async_http_client"] is not None
    assert model.kwargs["max_retries"] == eu._EMBED_SDK_MAX_RETRIES


def test_huggingface_provider_builds_without_http_clients():
    model = eu._build_embedding_model(
        provider_str="huggingface",
        model_str="BAAI/bge-small-en",
        base_url_str=None,
        api_key_str=None,
        timeout=30.0,
        provider="huggingface",
    )
    assert isinstance(model, _FakeHuggingFaceEmbedding)
    assert model.kwargs["model_name"] == "BAAI/bge-small-en"
    assert model.kwargs["request_timeout"] == 30.0


def test_local_provider_builds_huggingface_embedding_with_only_model_name():
    model = eu._build_embedding_model(
        provider_str="local",
        model_str="local-model",
        base_url_str=None,
        api_key_str=None,
        timeout=30.0,
        provider="local",
    )
    assert isinstance(model, _FakeHuggingFaceEmbedding)
    assert model.kwargs == {"model_name": "local-model"}


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


def test_ollama_provider_builds_and_rewraps_transports():
    model = eu._build_embedding_model(
        provider_str="ollama",
        model_str="llama3",
        base_url_str="http://ollama.invalid:11434",
        api_key_str="tok",
        timeout=30.0,
        provider="ollama",
    )
    assert isinstance(model, _FakeOllamaEmbedding)
    # Authorization header injected from api_key_str.
    assert model.kwargs["client_kwargs"]["headers"]["Authorization"] == "Bearer tok"
    # Transports were replaced (not the original _FakeInnerClient instances,
    # which get closed) with real httpx clients.
    assert not isinstance(model._client._client, _FakeInnerClient)
    assert not isinstance(model._async_client._client, _FakeInnerClient)


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
    assert (
        model.kwargs["client_kwargs"]["headers"]["Authorization"]
        == "Bearer explicit-header-wins"
    )


def test_ollama_dependency_missing_raises_import_error(monkeypatch):
    monkeypatch.setattr(eu, "OllamaEmbedding", None)
    with pytest.raises(ImportError, match="llama-index-embeddings-ollama"):
        eu._build_embedding_model(
            provider_str="ollama",
            model_str="llama3",
            base_url_str="http://ollama.invalid:11434",
            api_key_str=None,
            timeout=30.0,
            provider="ollama",
        )


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
    real_create_async_http_client = eu.create_async_http_client

    def spy_create_http_client(**kwargs):
        captured["sync_auth"] = kwargs.get("auth")
        # avoid httpx's strict auth-type validation by not forwarding the
        # sentinel further -- this test only needs to prove the VALUE reached
        # create_http_client/create_async_http_client, not build a real
        # client with it.
        kwargs["auth"] = None
        return real_create_http_client(**kwargs)

    def spy_create_async_http_client(**kwargs):
        captured["async_auth"] = kwargs.get("auth")
        kwargs["auth"] = None
        return real_create_async_http_client(**kwargs)

    monkeypatch.setattr(eu, "create_http_client", spy_create_http_client)
    monkeypatch.setattr(eu, "create_async_http_client", spy_create_async_http_client)

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
    assert captured["sync_auth"] == "SENTINEL_AUTH"
    assert captured["async_auth"] == "SENTINEL_AUTH"


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
