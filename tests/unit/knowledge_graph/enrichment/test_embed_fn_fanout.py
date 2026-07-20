"""make_embed_fn fans batches out up to embedding capacity (CONCEPT:AU-KG.compute.concurrency-controller-sizing)."""

from __future__ import annotations

import threading
import time

import pytest

from agent_utilities.core.model_concurrency import reset_controllers


@pytest.fixture(autouse=True)
def _isolate_controllers():
    reset_controllers()
    yield
    reset_controllers()


class _FakeEmbedModel:
    """Records max concurrent in-flight ``get_text_embedding_batch`` calls."""

    def __init__(self, hold: float = 0.02):
        self.lock = threading.Lock()
        self.current = 0
        self.max_seen = 0
        self.hold = hold

    def get_text_embedding_batch(self, chunk):
        with self.lock:
            self.current += 1
            self.max_seen = max(self.max_seen, self.current)
        time.sleep(self.hold)
        with self.lock:
            self.current -= 1
        # Deterministic vector per text so we can assert order.
        return [[float(len(t)), float(hash(t) % 7)] for t in chunk]


def _install(monkeypatch, fake, capacity: int):
    """Wire a fake embed model + a real config-resolved capacity.

    ``make_embed_fn`` calls ``resolve_capacity('embedding')`` which reads the live
    config registry, so we set a real embedding model whose total_capacity equals
    the requested value (instances=1, max_parallel_calls=capacity).
    """
    import agent_utilities.core.embedding_utilities as eu
    from agent_utilities.core import config as config_mod
    from agent_utilities.core.config import EmbeddingModelConfig

    monkeypatch.setattr(eu, "create_embedding_model", lambda *a, **k: fake)
    monkeypatch.setattr(
        config_mod.config,
        "embedding_models",
        [
            EmbeddingModelConfig(
                id="bge-m3",
                provider="openai",
                parallel_instances=1,
                max_parallel_calls=capacity,
            )
        ],
    )


def test_capacity_one_sequential_and_order_preserved(monkeypatch):
    from agent_utilities.knowledge_graph.enrichment.semantic import make_embed_fn

    fake = _FakeEmbedModel()
    _install(monkeypatch, fake, capacity=1)

    texts = [f"text-{i}" for i in range(10)]
    fn = make_embed_fn(batch_size=2)  # 5 batches
    out = fn(texts)

    assert len(out) == len(texts)
    assert out == [[float(len(t)), float(hash(t) % 7)] for t in texts]  # order
    assert fake.max_seen == 1  # strictly sequential


def test_capacity_k_runs_up_to_k_batches_concurrently(monkeypatch):
    from agent_utilities.knowledge_graph.enrichment.semantic import make_embed_fn

    k = 4
    fake = _FakeEmbedModel()
    _install(monkeypatch, fake, capacity=k)

    texts = [f"text-{i}" for i in range(40)]
    fn = make_embed_fn(batch_size=2)  # 20 batches, capacity 4
    out = fn(texts)

    # Same vectors, same order as the sequential reference.
    assert out == [[float(len(t)), float(hash(t) % 7)] for t in texts]
    assert fake.max_seen == k  # saturates to exactly capacity


def test_empty_input(monkeypatch):
    from agent_utilities.knowledge_graph.enrichment.semantic import make_embed_fn

    fake = _FakeEmbedModel()
    _install(monkeypatch, fake, capacity=8)
    assert make_embed_fn(batch_size=4)([]) == []


def test_unavailable_embedder_fails_loud_not_stub(monkeypatch):
    """When the embedder can't be constructed, make_embed_fn RAISES (Zero-Stub).

    Regression for the e2e profiler finding "embed_calls: 0 / no embeddings":
    the serving plane shipped without the ``embeddings-openai`` extra, so
    ``create_embedding_model()`` raised ``ModuleNotFoundError: No module named
    'llama_index.embeddings'``. The old fallback silently returned 1-dim ``[[0.0]]``
    stub vectors, so enrichment "succeeded" while corrupting the vector store and the
    failure was invisible. It must fail loud instead. (CONCEPT:AU-KG.memory.auto-similarity-memory-graph)
    """
    import agent_utilities.core.embedding_utilities as eu
    from agent_utilities.knowledge_graph.enrichment.semantic import make_embed_fn

    def _boom(*a, **k):
        raise ModuleNotFoundError("No module named 'llama_index.embeddings'")

    monkeypatch.setattr(eu, "create_embedding_model", _boom)

    with pytest.raises(RuntimeError, match="embedding model unavailable"):
        make_embed_fn()


def test_available_embedder_returns_real_multidim_vectors(monkeypatch):
    """The document/enrichment live path gets REAL (>1-dim) vectors, never a stub.

    Asserts the constructed embed fn returns the model's true multi-dimensional
    vectors — proving the document-ingest embedding path produces usable embeddings
    rather than the degenerate 1-dim zero stub. (CONCEPT:AU-KG.memory.auto-similarity-memory-graph)
    """
    from agent_utilities.knowledge_graph.enrichment.semantic import make_embed_fn

    fake = _FakeEmbedModel()
    _install(monkeypatch, fake, capacity=1)
    out = make_embed_fn(batch_size=4)(["hello world"])
    assert len(out) == 1
    assert len(out[0]) > 1  # real embedding, NOT the 1-dim [0.0] degenerate fallback
    assert out[0] != [0.0]
