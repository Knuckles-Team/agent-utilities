"""Per-model static headers plus mandatory-profile TLS for embeddings.

A configured embedding model's ``headers`` (e.g. a gateway ``X-Client-Id``) and
runtime TLS profiles are honored natively by ``create_embedding_model`` for the
openai-compatible embedder.
"""

from __future__ import annotations

import ssl
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
        embedding_tls_profile=None,
        embedding_tls_profile_ref=None,
        tls_system_trust=True,
        tls_trust_env=True,
        model_http_allowed_private_hosts=[],
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
    )
    monkeypatch.setattr(embedding_utilities, "config", _stub_config(embed_cfg))
    monkeypatch.setattr(embedding_utilities, "create_http_client", spy)
    embedding_utilities.clear_embedding_model_cache()

    _real_create_embedding_model(
        provider="openai", model="internal-embed", base_url="https://embed.internal/v1"
    )

    assert isinstance(captured.get("verify"), ssl.SSLContext)
    assert captured["verify"].verify_mode == ssl.CERT_REQUIRED
    assert captured.get("headers") == {"X-Client-Id": "svc-embed"}
