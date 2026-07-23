"""CONCEPT:AU-KG.retrieval.embedding-fast-fail — bound the OpenAI SDK's own retry loop.

``llama_index.embeddings.openai.OpenAIEmbedding`` defaults to ``max_retries=10``
with exponential backoff (up to ~8s per retry) when the caller does not pass an
explicit value. agent-utilities already owns a separate, endpoint-aware
circuit-breaker/backoff layer, so a second unbounded SDK-internal retry loop
only adds latency — every embedder construction must pass an explicit, small
``max_retries`` instead of silently inheriting the SDK default.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent_utilities.core import embedding_utilities
from agent_utilities.core.embedding_utilities import (
    _EMBED_SDK_MAX_RETRIES,
    create_embedding_model as _real_create_embedding_model,
)


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


def test_max_retries_is_explicit_and_bounded(monkeypatch):
    """The constructed OpenAIEmbedding must NOT silently inherit the SDK's
    default of 10 retries — it must receive our small, explicit bound."""
    assert _EMBED_SDK_MAX_RETRIES < 10  # sanity: we are actually bounding it

    captured: dict = {}

    class _FakeOpenAIEmbedding:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "llama_index.embeddings.openai.OpenAIEmbedding", _FakeOpenAIEmbedding
    )

    embed_cfg = SimpleNamespace(
        provider="openai",
        id="bge-m3",
        base_url="https://embed.internal/v1",
        api_key="ek",
        api_key_ref=None,
        oauth2=None,
        headers=None,
        headers_ref=None,
    )
    monkeypatch.setattr(embedding_utilities, "config", _stub_config(embed_cfg))
    embedding_utilities.clear_embedding_model_cache()

    _real_create_embedding_model(
        provider="openai", model="bge-m3", base_url="https://embed.internal/v1"
    )

    assert captured.get("max_retries") == _EMBED_SDK_MAX_RETRIES
