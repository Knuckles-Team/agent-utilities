"""Characterization test for ``create_embedding_model`` (CX-AU-07).

Pins the OBSERVED behaviour of the pre-refactor function (CCN 52,
``agent_utilities/core/embedding_utilities.py``) before it is decomposed
into per-concern helper functions (timeout validation, active-endpoint
failover resolution, identity resolution, credential resolution,
provider-specific validation, and cache-key construction). This test must
be green against the unmodified function; if it is not, the test is wrong,
not the code. It is not touched again in the refactor commit.

Runs OUTSIDE ``tests/unit/`` on purpose: ``tests/unit/conftest.py``'s
autouse ``_hermetic_embeddings`` fixture monkeypatches
``create_embedding_model`` itself to always raise, which would defeat a
test of this exact function. ``_build_embedding_model`` (the one
network/SDK-touching construction site, itself a separate CCN-15 function
this lane does not touch) is stubbed directly instead, mirroring
``tests/unit/knowledge_graph/test_embedder_client_cache.py``'s own pattern.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.core import embedding_utilities as eu


@pytest.fixture(autouse=True)
def _isolated_cache_and_stub_build(monkeypatch):
    eu.clear_embedding_model_cache()

    def _fake_build(**kwargs):
        return SimpleNamespace(_build_kwargs=kwargs)

    monkeypatch.setattr(eu, "_build_embedding_model", _fake_build)
    yield
    eu.clear_embedding_model_cache()


def _config(**overrides):
    base = dict(
        default_embedding_model=None,
        default_chat_model=None,
        openai_api_key="config-openai-key",
        embedding_tls_profile=None,
        embedding_tls_profile_ref=None,
        model_http_allowed_private_hosts=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _embed_cfg(**overrides):
    base = dict(
        provider="openai",
        id="text-embedding-3-small",
        base_url=None,
        api_key_ref=None,
        oauth2=None,
        headers_ref=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _chat_cfg(**overrides):
    base = dict(
        provider="openai",
        base_url=None,
        api_key_ref=None,
        oauth2=None,
        headers_ref=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), 3_601, True])
def test_invalid_timeout_rejected(monkeypatch, timeout):
    monkeypatch.setattr(eu, "config", _config())
    with pytest.raises(ValueError, match="timeout"):
        eu.create_embedding_model(provider="openai", model="m", timeout=timeout)


def test_oauth2_and_api_key_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(eu, "config", _config())
    with pytest.raises(ValueError, match="mutually exclusive"):
        eu.create_embedding_model(
            provider="openai",
            model="m",
            api_key="k",
            oauth2={"token_url": "https://idp.invalid/token"},
        )


def test_no_embedding_model_configured_raises(monkeypatch):
    monkeypatch.setattr(eu, "config", _config())
    monkeypatch.setattr(
        "agent_utilities.core.embedding_failover.active_embedding_endpoint",
        lambda: (_ for _ in ()).throw(RuntimeError("no failover configured")),
    )
    with pytest.raises(ValueError, match="No embedding model is configured"):
        eu.create_embedding_model()


def test_explicit_args_used_verbatim_without_consulting_failover(monkeypatch):
    calls = {"failover": 0}

    def _fake_active_endpoint():
        calls["failover"] += 1
        raise AssertionError("failover must not be consulted when args are explicit")

    monkeypatch.setattr(eu, "config", _config())
    monkeypatch.setattr(
        "agent_utilities.core.embedding_failover.active_embedding_endpoint",
        _fake_active_endpoint,
    )
    model = eu.create_embedding_model(
        provider="openai",
        model="explicit-model",
        base_url="https://explicit.invalid",
        api_key="k",
    )
    assert model._build_kwargs["model_str"] == "explicit-model"
    assert calls["failover"] == 0


def test_default_embedding_model_used_when_nothing_explicit(monkeypatch):
    monkeypatch.setattr(
        eu, "config", _config(default_embedding_model=_embed_cfg(id="default-embed"))
    )
    monkeypatch.setattr(
        "agent_utilities.core.embedding_failover.active_embedding_endpoint",
        lambda: (_ for _ in ()).throw(RuntimeError("no failover configured")),
    )
    model = eu.create_embedding_model()
    assert model._build_kwargs["model_str"] == "default-embed"


def test_active_failover_endpoint_overrides_static_default(monkeypatch):
    monkeypatch.setattr(
        eu, "config", _config(default_embedding_model=_embed_cfg(id="static-default"))
    )
    fake_endpoint = SimpleNamespace(
        provider="openai",
        model_id="failover-endpoint-model",
        base_url="https://failover.invalid",
        api_key_ref=None,
        oauth2=None,
        headers_ref=None,
    )
    monkeypatch.setattr(
        "agent_utilities.core.embedding_failover.active_embedding_endpoint",
        lambda: fake_endpoint,
    )
    model = eu.create_embedding_model()
    assert model._build_kwargs["model_str"] == "failover-endpoint-model"
    assert model._build_kwargs["base_url_str"] == "https://failover.invalid"


def test_failover_exception_falls_back_to_static_default_silently(monkeypatch):
    monkeypatch.setattr(
        eu, "config", _config(default_embedding_model=_embed_cfg(id="static-default"))
    )
    monkeypatch.setattr(
        "agent_utilities.core.embedding_failover.active_embedding_endpoint",
        lambda: (_ for _ in ()).throw(RuntimeError("breaker open, no fallback either")),
    )
    model = eu.create_embedding_model()
    assert model._build_kwargs["model_str"] == "static-default"


def test_credentials_fall_back_to_chat_model_when_embedder_declares_neither(
    monkeypatch,
):
    monkeypatch.setattr(
        eu,
        "config",
        _config(
            default_embedding_model=_embed_cfg(id="e", api_key_ref=None, oauth2=None),
            default_chat_model=_chat_cfg(api_key_ref="env://CHAT_KEY"),
        ),
    )
    monkeypatch.setattr(
        "agent_utilities.core.embedding_failover.active_embedding_endpoint",
        lambda: (_ for _ in ()).throw(RuntimeError("no failover configured")),
    )
    monkeypatch.setenv("CHAT_KEY", "chat-fallback-secret")
    model = eu.create_embedding_model()
    assert model._build_kwargs["api_key_str"] == "chat-fallback-secret"


def test_embedder_own_api_key_ref_wins_over_chat_model_fallback(monkeypatch):
    monkeypatch.setattr(
        eu,
        "config",
        _config(
            default_embedding_model=_embed_cfg(id="e", api_key_ref="env://EMBED_KEY"),
            default_chat_model=_chat_cfg(api_key_ref="env://CHAT_KEY"),
        ),
    )
    # provider/model/base_url all None -> create_embedding_model consults the
    # live embedding-failover module; pin it explicitly so this test observes
    # the STATIC default_embedding_model path deterministically rather than
    # whatever failover state happens to be configured in this environment.
    monkeypatch.setattr(
        "agent_utilities.core.embedding_failover.active_embedding_endpoint",
        lambda: (_ for _ in ()).throw(RuntimeError("no failover configured")),
    )
    monkeypatch.setenv("EMBED_KEY", "embed-secret")
    monkeypatch.setenv("CHAT_KEY", "chat-secret")
    model = eu.create_embedding_model()
    assert model._build_kwargs["api_key_str"] == "embed-secret"


def test_ambiguous_auth_oauth2_and_api_key_both_resolved_raises(monkeypatch):
    # Observed: the EARLY explicit-arg exclusivity check (api_key= and oauth2=
    # both passed to the call) raises "mutually exclusive" first. To reach the
    # LATER "ambiguous" check instead, the ambiguity must come from RESOLVED
    # values via the embedder's own registry entry (which declares both an
    # api_key_ref and an oauth2 block) rather than from the caller's args.
    monkeypatch.setattr(
        eu,
        "config",
        _config(
            default_embedding_model=_embed_cfg(
                id="e",
                api_key_ref="env://EMBED_KEY",
                oauth2={"token_url": "https://idp.invalid/token"},
            )
        ),
    )
    monkeypatch.setattr(
        "agent_utilities.core.embedding_failover.active_embedding_endpoint",
        lambda: (_ for _ in ()).throw(RuntimeError("no failover configured")),
    )
    monkeypatch.setenv("EMBED_KEY", "embed-secret")
    with pytest.raises(ValueError, match="ambiguous"):
        eu.create_embedding_model()


def test_mock_provider_is_forbidden(monkeypatch):
    monkeypatch.setattr(eu, "config", _config())
    with pytest.raises(ValueError, match="Zero-Stub Compliance"):
        eu.create_embedding_model(provider="mock", model="m")


def test_openai_provider_falls_back_to_config_api_key_when_none_resolved(monkeypatch):
    monkeypatch.setattr(eu, "config", _config(openai_api_key="config-fallback-key"))
    model = eu.create_embedding_model(
        provider="openai", model="m", base_url="https://x.invalid"
    )
    assert model._build_kwargs["api_key_str"] == "config-fallback-key"


def test_openai_provider_with_no_key_anywhere_raises(monkeypatch):
    monkeypatch.setattr(eu, "config", _config(openai_api_key=None))
    with pytest.raises(ValueError, match="requires explicit"):
        eu.create_embedding_model(
            provider="openai", model="m", base_url="https://x.invalid"
        )


def test_openai_provider_oauth2_managed_sentinel_bypasses_missing_key(monkeypatch):
    monkeypatch.setattr(eu, "config", _config(openai_api_key=None))
    model = eu.create_embedding_model(
        provider="openai",
        model="m",
        base_url="https://x.invalid",
        oauth2={"token_url": "https://idp.invalid/token"},
    )
    assert model._build_kwargs["api_key_str"] == "oauth2-managed"


def test_ollama_provider_requires_base_url(monkeypatch):
    monkeypatch.setattr(eu, "config", _config())
    with pytest.raises(ValueError, match="Ollama embedding endpoint is not configured"):
        eu.create_embedding_model(provider="ollama", model="m", base_url=None)


def test_cache_returns_identical_object_for_identical_resolved_config(monkeypatch):
    monkeypatch.setattr(eu, "config", _config())
    m1 = eu.create_embedding_model(
        provider="openai", model="m", base_url="https://x.invalid", api_key="k"
    )
    m2 = eu.create_embedding_model(
        provider="openai", model="m", base_url="https://x.invalid", api_key="k"
    )
    assert m1 is m2


def test_cache_key_includes_model_so_different_models_get_different_clients(
    monkeypatch,
):
    monkeypatch.setattr(eu, "config", _config())
    m1 = eu.create_embedding_model(
        provider="openai", model="model-a", base_url="https://x.invalid", api_key="k"
    )
    m2 = eu.create_embedding_model(
        provider="openai", model="model-b", base_url="https://x.invalid", api_key="k"
    )
    assert m1 is not m2
