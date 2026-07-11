"""Per-model static headers + TLS for the embedding client (parity with model_factory).

A configured embedding model's ``headers`` (e.g. a gateway ``X-Client-Id``) and
``ssl_verify`` (bool / CA-bundle path / None-inherit) are honored natively by
``create_embedding_model`` for the openai-compatible embedder — an internal self-signed
embedder or a header-gated gateway works with no global change.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent_utilities.core import embedding_utilities
from agent_utilities.core.embedding_utilities import (
    create_embedding_model as _real_create_embedding_model,
)

# The unit suite's ``tests/unit/conftest.py`` autouse fixture patches
# ``create_embedding_model`` to refuse network. The alias above was bound at import time
# (before that fixture runs), so it still points at the real function — we call it directly
# to exercise the actual header/TLS resolution, while the token/HTTP client is patched.


def _stub_config(embed_cfg):
    return SimpleNamespace(
        default_embedding_model=embed_cfg,
        default_chat_model=None,
        openai_api_key="k",
    )


def test_embedding_per_model_headers_and_tls_reach_http_client(monkeypatch):
    captured: dict = {}
    real_create = embedding_utilities.create_http_client

    def spy(**kwargs):
        captured.update(kwargs)
        return real_create(**kwargs)

    embed_cfg = SimpleNamespace(
        provider="openai",
        id="internal-embed",
        base_url="https://embed.internal/v1",
        api_key="ek",
        oauth2=None,
        headers={"X-Client-Id": "svc-embed"},
        ssl_verify=False,
    )
    monkeypatch.setattr(embedding_utilities, "config", _stub_config(embed_cfg))
    monkeypatch.setattr(embedding_utilities, "create_http_client", spy)
    embedding_utilities.clear_embedding_model_cache()

    # Explicit args skip the failover resolution; headers/ssl_verify still come from cfg.
    _real_create_embedding_model(
        provider="openai", model="internal-embed", base_url="https://embed.internal/v1"
    )

    assert captured.get("verify") is False  # per-model TLS applied
    assert captured.get("headers") == {"X-Client-Id": "svc-embed"}


def test_embedding_ssl_verify_none_inherits_caller(monkeypatch):
    captured: dict = {}
    real_create = embedding_utilities.create_http_client

    def spy(**kwargs):
        captured.update(kwargs)
        return real_create(**kwargs)

    # ssl_verify None on the config ⇒ inherit the caller's ssl_verify (here False, so a
    # client is still built and we can observe the inherited verify value).
    embed_cfg = SimpleNamespace(
        provider="openai",
        id="e",
        base_url="https://e/v1",
        api_key="ek",
        oauth2=None,
        headers={},
        ssl_verify=None,
    )
    monkeypatch.setattr(embedding_utilities, "config", _stub_config(embed_cfg))
    monkeypatch.setattr(embedding_utilities, "create_http_client", spy)
    embedding_utilities.clear_embedding_model_cache()

    _real_create_embedding_model(
        provider="openai", model="e", base_url="https://e/v1", ssl_verify=False
    )

    assert captured.get("verify") is False  # inherited from the caller, not the config
