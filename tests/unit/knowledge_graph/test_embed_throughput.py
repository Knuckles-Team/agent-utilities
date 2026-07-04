"""Batched + concurrent embedding throughput (CONCEPT:AU-KG.ingest.applying-agents-md-batch).

The fresh north-star e2e run shifted its bottleneck to ENRICHMENT: embeds were
issued one HTTP round-trip at a time (the host log showed one
``POST /v1/embeddings`` every ~2-3s), dropping ``parallelism_factor`` to ~1.83.
``make_embed_fn`` now (a) sends a big LIST of inputs per request (one POST per
chunk, not per text) and (b) fans those chunk-requests out concurrently. These
tests prove both — N texts cost ``ceil(N/batch)`` POSTs (not N), and the batches
overlap — while preserving per-node vector correctness and input order.
"""

from __future__ import annotations

import threading
import time

import pytest

from agent_utilities.knowledge_graph.enrichment import semantic


class _FakeEmbedModel:
    """Stand-in for the llama-index OpenAIEmbedding pointed at bge-m3.

    ``get_text_embedding_batch`` is ONE POST: it records the call (the batch size),
    tracks max concurrency, and returns a deterministic order-encoding vector per
    text so the caller's flatten/order can be verified.
    """

    def __init__(self, sleep: float = 0.0) -> None:
        self.embed_batch_size = 10  # llama-index DEFAULT_EMBED_BATCH_SIZE
        self._sleep = sleep
        self.calls: list[int] = []  # batch size of each POST
        self._lock = threading.Lock()
        self._inflight = 0
        self.max_inflight = 0

    def get_text_embedding_batch(self, texts, **_kw):
        with self._lock:
            self._inflight += 1
            self.max_inflight = max(self.max_inflight, self._inflight)
            self.calls.append(len(texts))
        try:
            if self._sleep:
                time.sleep(self._sleep)
            # Encode each text's integer payload into a 1024-dim-ish vector so the
            # test can assert the right vector lands for the right text, in order.
            return [[float(int(t)), 1.0, 2.0] for t in texts]
        finally:
            with self._lock:
                self._inflight -= 1


@pytest.fixture(autouse=True)
def _reset_embed_cache(monkeypatch):
    # The cpu/load-derived concurrency is memoized at module scope — reset it so
    # each test controls it explicitly.
    monkeypatch.setattr(semantic, "_EMBED_CONCURRENCY", None)
    yield
    monkeypatch.setattr(semantic, "_EMBED_CONCURRENCY", None)


def _install_fake(monkeypatch, model):
    import agent_utilities.core.embedding_utilities as eu

    monkeypatch.setattr(eu, "create_embedding_model", lambda *a, **k: model)


def test_embeds_are_batched_not_per_text(monkeypatch):
    """100 texts at batch=50 → 2 POSTs (not 100), and the model's embed_batch_size
    is pinned big so it doesn't re-split a chunk into tiny sub-POSTs."""
    model = _FakeEmbedModel()
    _install_fake(monkeypatch, model)

    embed_fn = semantic.make_embed_fn(batch_size=50)
    texts = [str(i) for i in range(100)]
    vecs = embed_fn(texts)

    assert len(vecs) == 100
    # Two batched POSTs of 50 — NOT 100 per-text round-trips.
    assert model.calls == [50, 50]
    # Pinned so llama-index issues one POST per chunk (>= our max batch).
    assert model.embed_batch_size >= semantic._EMBED_MAX_BATCH
    # Order + correctness: vector i encodes text i.
    assert [int(v[0]) for v in vecs] == list(range(100))


def test_auto_batch_fills_concurrency_lanes_and_overlaps(monkeypatch):
    """With concurrency pinned to 4 and an auto-sized batch, the chunk-requests run
    CONCURRENTLY (max in-flight > 1) and finish far faster than serial."""
    model = _FakeEmbedModel(sleep=0.05)
    _install_fake(monkeypatch, model)
    monkeypatch.setattr(semantic, "_embed_concurrency", lambda: 4)

    embed_fn = semantic.make_embed_fn()  # auto batch
    n = 400
    texts = [str(i) for i in range(n)]

    t0 = time.monotonic()
    vecs = embed_fn(texts)
    wall = time.monotonic() - t0

    assert len(vecs) == n
    assert [int(v[0]) for v in vecs] == list(range(n))
    # Batched: far fewer POSTs than texts.
    assert len(model.calls) < n
    assert sum(model.calls) == n  # every text embedded exactly once
    # Concurrency actually happened.
    assert model.max_inflight >= 2
    # And it beats the serial wall time (serial would be len(calls) * 0.05).
    assert wall < len(model.calls) * 0.05


def test_auto_batch_sizes_big_posts(monkeypatch):
    """A large input with low concurrency still sends BIG batches (≤ the cap), so a
    single POST carries many texts rather than one each."""
    model = _FakeEmbedModel()
    _install_fake(monkeypatch, model)
    monkeypatch.setattr(semantic, "_embed_concurrency", lambda: 2)

    embed_fn = semantic.make_embed_fn()
    embed_fn([str(i) for i in range(2000)])
    # 2 lanes over 2000 → ~1000/chunk, clamped to the 256 cap → 8 POSTs of 256.
    assert max(model.calls) == semantic._EMBED_MAX_BATCH
    assert all(c >= 32 for c in model.calls)


def test_empty_input_makes_no_calls(monkeypatch):
    model = _FakeEmbedModel()
    _install_fake(monkeypatch, model)
    assert semantic.make_embed_fn()([]) == []
    assert model.calls == []


def test_fail_loud_when_embedder_unavailable(monkeypatch):
    """KG-2.3 fail-loud is preserved: a missing/unreachable embedder raises rather
    than returning a degenerate stub vector."""
    import agent_utilities.core.embedding_utilities as eu

    def _boom(*a, **k):
        raise RuntimeError("No module named 'llama_index.embeddings'")

    monkeypatch.setattr(eu, "create_embedding_model", _boom)
    with pytest.raises(RuntimeError, match="embedding model unavailable"):
        semantic.make_embed_fn()
