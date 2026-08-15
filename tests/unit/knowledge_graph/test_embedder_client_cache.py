"""Process-scoped embedder-client cache (CONCEPT:AU-KG.compute.config-keyed-embedder-client).

``create_embedding_model`` used to construct a brand-new LlamaIndex embedding
client (httpx client + TLS context + tokenizer) on EVERY call. On the ingest hot
path that is per-window / per-document / per-fact, so the live host log showed a
``Creating OpenAIEmbedding`` line on every embedding call. These tests lock the
fix: one client per distinct resolved config, reused across the run, thread-safe,
without weakening the fail-loud KG-2.3 contract.
"""

from __future__ import annotations

import threading

import pytest

from agent_utilities.core import embedding_utilities as eu
from agent_utilities.core.config import EmbeddingModelConfig, config

# Captured at collection time, before ``tests/unit/conftest.py``'s autouse
# ``_hermetic_embeddings`` fixture monkeypatches ``eu.create_embedding_model``
# to always raise (it blocks the live client so unit tests never touch the
# network). This module tests the caching behaviour of the real factory
# itself, so each test below restores it via ``_real_create_embedding_model``
# and only stubs the actual network-touching construction site
# (``_build_embedding_model``) instead.
_REAL_CREATE_EMBEDDING_MODEL = eu.create_embedding_model


@pytest.fixture(autouse=True)
def _real_create_embedding_model(monkeypatch):
    """Undo the hermetic block on ``create_embedding_model`` for this file.

    Per ``tests/unit/conftest.py``'s own docstring: "Tests that need a
    functioning embedder still @patch the factory themselves, and that
    per-test patch overrides this one." Restoring the real factory is safe
    here because every test also stubs ``_build_embedding_model`` (the one
    site that would touch the network), so the cache logic runs for real
    without ever constructing a live client.

    Also seeds ``config.embedding_models`` (empty by default in this
    sandbox, D-TC-4) with one entry so ``create_embedding_model()``'s
    default-resolution finds a ``model``/``provider`` to key its cache on —
    the "no embedding model is configured" ValueError this used to hit was a
    config-resolution gap, not a real construction, since construction stays
    stubbed via ``_build_embedding_model``.
    """
    monkeypatch.setattr(eu, "create_embedding_model", _REAL_CREATE_EMBEDDING_MODEL)
    monkeypatch.setattr(
        config,
        "embedding_models",
        [EmbeddingModelConfig(id="test-embed-model", provider="openai")],
    )


class _DummyModel:
    """Stand-in for a LlamaIndex embedder (no network / no llama-index needed)."""

    embed_batch_size = 0

    def get_text_embedding(self, _text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def counting_builder(monkeypatch):
    """Replace the un-cached construction site with a counter."""
    calls = {"n": 0}

    def _fake_build(**_kw):
        calls["n"] += 1
        return _DummyModel()

    monkeypatch.setattr(eu, "_build_embedding_model", _fake_build)
    eu.clear_embedding_model_cache()
    yield calls
    eu.clear_embedding_model_cache()


def test_built_once_for_many_calls(counting_builder):
    """N create_embedding_model() calls with the same config → ONE construction."""
    models = [eu.create_embedding_model() for _ in range(64)]
    assert counting_builder["n"] == 1
    # Every caller gets the SAME shared client (so an admission wrapper has one
    # stable instance to wrap).
    assert all(m is models[0] for m in models)


def test_distinct_config_builds_distinct_client(counting_builder):
    eu.create_embedding_model(provider="openai", model="model-a")
    eu.create_embedding_model(provider="openai", model="model-a")
    assert counting_builder["n"] == 1
    eu.create_embedding_model(provider="openai", model="model-b")
    assert counting_builder["n"] == 2


def test_make_embed_fn_reuses_one_client(counting_builder):
    """The batched enrichment embedder builds the client once across many fns."""
    from agent_utilities.knowledge_graph.enrichment.semantic import make_embed_fn

    fns = [make_embed_fn() for _ in range(8)]
    # Exercise each fn so any lazy construction would fire.
    for fn in fns:
        fn(["hello", "world"])
    assert counting_builder["n"] == 1


def test_cache_is_thread_safe(counting_builder):
    """Concurrent first-callers must not race into multiple constructions."""
    barrier = threading.Barrier(16)
    out: list[object] = []
    lock = threading.Lock()

    def _worker():
        barrier.wait()
        m = eu.create_embedding_model()
        with lock:
            out.append(m)

    threads = [threading.Thread(target=_worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counting_builder["n"] == 1
    assert all(m is out[0] for m in out)


def test_fail_loud_contract_preserved(monkeypatch):
    """A missing/unsupported provider still RAISES (KG-2.3), never caches a stub."""
    eu.clear_embedding_model_cache()

    def _boom(**_kw):
        raise ValueError("Unsupported embedding provider: bogus")

    monkeypatch.setattr(eu, "_build_embedding_model", _boom)
    with pytest.raises(ValueError):
        eu.create_embedding_model(provider="bogus")
    # Nothing was cached, so a later good config still builds cleanly.
    monkeypatch.setattr(eu, "_build_embedding_model", lambda **_k: _DummyModel())
    assert eu.create_embedding_model(provider="openai") is not None
    eu.clear_embedding_model_cache()
