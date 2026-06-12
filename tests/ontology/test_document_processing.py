#!/usr/bin/python
"""Tests for the Document Processing → Ontology pipeline (CONCEPT:KG-2.48).

Self-contained against existing stable code: text is passed inline (no PDF/OCR
dependency), embeddings are injected (no network/model), and the live write path
is exercised against an in-memory fake writer so no daemon/backend is required.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.ontology.document_processing import (
    CHUNK_NODE_TYPE,
    CHUNK_OF_EDGE,
    DOCUMENT_NODE_TYPE,
    HAS_CHUNK_EDGE,
    ChunkingConfig,
    DocumentExtractionError,
    DocumentProcessor,
    chunk_text,
    process_document,
)

EMBED_DIM = 768

# A multi-paragraph document long enough to force several overlapping chunks.
_PARAGRAPH = (
    "The ontology document processing pipeline extracts text from a media set. "
    "It then splits that text into overlapping chunks using a separator priority. "
    "Each chunk is exploded into its own object and embedded for semantic search. "
)
DOC_TEXT = "\n\n".join(_PARAGRAPH * 3 for _ in range(6))


def _fake_embed(texts):
    """Deterministic 768-dim embeddings (no model/network)."""
    out = []
    for i, _t in enumerate(texts):
        vec = [float((i + 1) % 7) * 0.01] * EMBED_DIM
        out.append(vec)
    return out


class _FakeWriter:
    """In-memory add_node/add_edge sink mirroring the backend write contract."""

    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node_id, label="", **properties):
        self.nodes[node_id] = {"label": label, **properties}

    def add_edge(self, source, target, rel_type="", **properties):
        self.edges.append((source, target, rel_type, properties))


class _FakeFacade:
    """Stands in for KnowledgeGraph: exposes a `.store` with the writer."""

    def __init__(self, writer):
        self.store = writer


def test_chunking_overlap_and_monotonic_positions():
    cfg = ChunkingConfig(chunk_size=200, overlap=50)
    spans = chunk_text(DOC_TEXT, cfg)

    assert len(spans) >= 3, "multi-paragraph doc should split into several chunks"
    # Contiguous 0..N-1 indices.
    assert [s.index for s in spans] == list(range(len(spans)))
    # Monotonic non-decreasing char_start.
    starts = [s.char_start for s in spans]
    assert starts == sorted(starts)
    # No chunk exceeds the budget + overlap tail (sliding-window invariant).
    for s in spans:
        assert len(s.text) <= cfg.chunk_size + cfg.overlap + 1
    # Consecutive spans overlap (next starts at/before previous end).
    overlaps = [
        spans[i + 1].char_start <= spans[i].char_end for i in range(len(spans) - 1)
    ]
    assert all(overlaps), "successive chunks must overlap on the source text"


def test_overlap_must_be_less_than_chunk_size():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=100, overlap=100)


def test_process_materializes_document_and_linked_chunks():
    writer = _FakeWriter()
    proc = DocumentProcessor(
        _FakeFacade(writer),
        chunking=ChunkingConfig(chunk_size=200, overlap=50),
        embed_fn=_fake_embed,
        embedding_dim=EMBED_DIM,
    )
    result = proc.process(DOC_TEXT, title="Doc Pipeline", source="memory://doc1")

    assert result.persisted is True
    assert result.chunk_count >= 3
    assert len(result.chunk_nodes) == result.chunk_count

    doc_id = result.document_id
    assert result.document_node["type"] == DOCUMENT_NODE_TYPE
    assert result.document_node["chunk_count"] == result.chunk_count

    # Every chunk node is correctly typed, positioned monotonically, and carries
    # a 768-dim embedding.
    positions = [c["position"] for c in result.chunk_nodes]
    assert positions == list(range(len(result.chunk_nodes)))
    for c in result.chunk_nodes:
        assert c["type"] == CHUNK_NODE_TYPE
        assert c["document_id"] == doc_id
        assert len(c["embedding"]) == EMBED_DIM
        assert c["embedding_dim"] == EMBED_DIM

    # HAS_CHUNK (doc->chunk) and CHUNK_OF (chunk->doc) for every chunk.
    has_chunk = [e for e in result.edges if e["type"] == HAS_CHUNK_EDGE]
    chunk_of = [e for e in result.edges if e["type"] == CHUNK_OF_EDGE]
    assert len(has_chunk) == result.chunk_count
    assert len(chunk_of) == result.chunk_count
    for e in has_chunk:
        assert e["source"] == doc_id
        assert e["target"] in writer.nodes
    for e in chunk_of:
        assert e["target"] == doc_id
        assert e["source"] in writer.nodes


def test_live_write_path_persists_through_backend_integration():
    """The live write path actually calls add_node/add_edge on the backend."""
    writer = _FakeWriter()
    proc = DocumentProcessor(
        _FakeFacade(writer),
        chunking=ChunkingConfig(chunk_size=200, overlap=40),
        embed_fn=_fake_embed,
    )
    result = proc.process(DOC_TEXT, source="memory://doc2")

    # Document + one node per chunk written to the backend.
    assert result.document_id in writer.nodes
    assert writer.nodes[result.document_id]["type"] == DOCUMENT_NODE_TYPE
    chunk_node_ids = [c["id"] for c in result.chunk_nodes]
    for cid in chunk_node_ids:
        assert cid in writer.nodes
        assert writer.nodes[cid]["type"] == CHUNK_NODE_TYPE
    # Two edges (HAS_CHUNK + CHUNK_OF) per chunk hit the backend.
    assert len(writer.edges) == 2 * result.chunk_count
    rel_types = {rel for _s, _t, rel, _p in writer.edges}
    assert rel_types == {HAS_CHUNK_EDGE, CHUNK_OF_EDGE}


def test_idempotent_document_id():
    proc = DocumentProcessor(embed_fn=_fake_embed)
    r1 = proc.process(DOC_TEXT, source="memory://same")
    r2 = proc.process(DOC_TEXT, source="memory://same")
    assert r1.document_id == r2.document_id
    # Same chunk ids on re-process (idempotent materialization).
    assert [c["id"] for c in r1.chunk_nodes] == [c["id"] for c in r2.chunk_nodes]


def test_offline_returns_structure_without_persist():
    proc = DocumentProcessor(graph=None, embed_fn=_fake_embed)
    result = proc.process(DOC_TEXT, source="memory://offline")
    assert result.persisted is False
    assert result.chunk_count >= 1
    mapping = result.as_dict()
    assert set(mapping) == {"document_node", "chunk_nodes", "edges"}


def test_empty_text_raises_clear_error():
    proc = DocumentProcessor(embed_fn=_fake_embed)
    with pytest.raises(DocumentExtractionError):
        proc.process("   \n  ", source="memory://empty")


def test_embeddings_degrade_to_none_when_model_unavailable():
    """No embed_fn + unavailable model → chunks still materialize (embedding None)."""

    def _boom(_texts):
        raise RuntimeError("no model")

    proc = DocumentProcessor(embed_fn=_boom)
    result = proc.process(DOC_TEXT, source="memory://noembed")
    assert result.chunk_count >= 1
    for c in result.chunk_nodes:
        assert "embedding" not in c  # None embeddings are not written onto the node


def test_convenience_process_document_returns_mapping():
    writer = _FakeWriter()
    mapping = process_document(
        DOC_TEXT,
        _FakeFacade(writer),
        chunk_size=200,
        overlap=50,
        source="memory://conv",
    )
    assert "document_node" in mapping
    assert mapping["chunk_nodes"]
    assert mapping["edges"]
    # Persisted through the live path.
    assert mapping["document_node"]["id"] in writer.nodes


def test_bytes_input_is_decoded_and_processed():
    proc = DocumentProcessor(embed_fn=_fake_embed)
    result = proc.process(DOC_TEXT.encode("utf-8"), source="memory://bytes")
    assert result.chunk_count >= 1
    assert result.document_node["type"] == DOCUMENT_NODE_TYPE


def test_persist_uses_engine_bulk_path_when_available():
    """B6: when the backend exposes an engine bulk path, the slice is materialized
    via batched RPCs (batch_update) instead of one add_node per element."""

    class _Graph:
        def __init__(self):
            self.batches = []

        def batch_update(self, ops):
            self.batches.append(list(ops))
            return {"ok": True}

    class _BulkBackend:
        def __init__(self):
            self._graph = _Graph()
            self.per_item_nodes = 0

        def add_node(self, *a, **k):  # must NOT be hit when bulk works
            self.per_item_nodes += 1

        def add_edge(self, *a, **k):
            pass

    backend = _BulkBackend()
    proc = DocumentProcessor(
        _FakeFacade(backend),
        chunking=ChunkingConfig(chunk_size=200, overlap=50),
        embed_fn=_fake_embed,
        embedding_dim=EMBED_DIM,
    )
    result = proc.process(DOC_TEXT, source="memory://bulk")

    assert result.persisted is True
    assert backend._graph.batches, "batch_update should have been called"
    assert backend.per_item_nodes == 0, "must not fall back to per-item when bulk works"
    node_ops = [
        op
        for batch in backend._graph.batches
        for op in batch
        if op.get("op") == "add_node"
    ]
    assert len(node_ops) == 1 + result.chunk_count  # document + every chunk
