"""The serving/host plane must ship the CANONICAL embedder (CONCEPT:AU-KG.memory.auto-similarity-memory-graph).

Regression for the e2e north-star ingestion profiler findings:

  * BUG 1 — every ``document`` ingest dead-lettered with
    ``ModuleNotFoundError: No module named 'llama_index.embeddings'`` (the document
    worker calls ``create_embedding_model()``).
  * BUG 2 — ``embed_calls: 0`` fleet-wide / ZERO embeddings (``make_embed_fn()`` /
    the backfill could not construct an embedder).

D2 (GHSA-8mgp-746c-j5xp, nltk had no patched release): removed llama-index
entirely. The canonical embedder is now a native OpenAI-compatible HTTP
client (``core/embedding_utilities._HttpEmbeddingModel``) talking to the
bge-m3 vLLM endpoint — it needs no optional package at all, so the failure
mode these tests originally guarded is now structurally impossible rather
than merely routed around by an extra. This test locks THAT invariant: the
openai embedding provider builds with base dependencies alone, and the
former ``embeddings``/``embeddings-openai`` extras (kept only as no-op
aliases for an external pin, see pyproject.toml) ship no packages.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


def _pyproject() -> dict:
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_embeddings_extras_are_no_op_aliases_not_llama_index():
    """Regression guard: these extras must never reintroduce llama-index."""
    extras = _pyproject()["project"]["optional-dependencies"]
    for name in ("embeddings", "embeddings-openai", "embeddings-ollama"):
        joined = " ".join(extras[name]).lower()
        assert "llama" not in joined, f"{name} must not depend on llama-index: {extras[name]}"


def test_openai_embedding_provider_builds_with_base_dependencies_only(monkeypatch):
    """The exact regression this file guards: constructing the canonical
    (openai/bge-m3) embedder must never raise ModuleNotFoundError for an
    optional package — it is native HTTP over `httpx`, already a base
    (transitive, via requests->urllib3 and http_client.py's own use) capability.
    """
    from types import SimpleNamespace

    from agent_utilities.core import embedding_utilities as eu

    monkeypatch.setattr(
        eu,
        "config",
        SimpleNamespace(
            embedding_tls_profile=None,
            embedding_tls_profile_ref=None,
            tls_system_trust=True,
            tls_trust_env=True,
            model_http_allowed_private_hosts=[],
        ),
    )
    model = eu._build_embedding_model(
        provider_str="openai",
        model_str="bge-m3",
        base_url_str="https://vllm.invalid/v1",
        api_key_str="k",
        timeout=30.0,
        provider="openai",
    )
    assert isinstance(model, eu._HttpEmbeddingModel)
    assert hasattr(model, "get_text_embedding_batch")


def test_huggingface_and_local_providers_no_longer_available_in_core():
    """Heavy local embedding inference was moved out of core (AGENTS.md
    'Dependency discipline') as part of removing llama-index-embeddings-huggingface
    (its own transitive path to llama-index-core -> nltk)."""
    from agent_utilities.core import embedding_utilities as eu

    with pytest.raises(ValueError, match="data-science-mcp"):
        eu._build_huggingface_embedding("BAAI/bge-small-en", timeout=30.0)
    with pytest.raises(ValueError, match="data-science-mcp"):
        eu._build_local_embedding("local-model")
