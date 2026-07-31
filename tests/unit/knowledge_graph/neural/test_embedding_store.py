"""Versioned tenant-scoped embedding production (CONCEPT:AU-KG.mining.governed-neural-layer, KG-6.6)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.neural import embedding_store


@pytest.fixture(autouse=True)
def _envelope_commit(monkeypatch):
    committed = []
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_envelope",
        lambda engine, env: committed.append(env) or {"status": "success"},
    )
    return committed


@pytest.fixture(autouse=True)
def _fake_embedder(monkeypatch):
    fake_model = MagicMock()
    fake_model.get_text_embedding = lambda text: [0.1, 0.2, 0.3]
    monkeypatch.setattr(
        "agent_utilities.core.embedding_utilities.create_embedding_model",
        lambda: fake_model,
    )
    return fake_model


def test_build_tenant_embedding_indexes_in_engine_and_commits_record(_envelope_commit):
    engine = MagicMock()
    calls = []
    engine.add_embedding = lambda node_id, vec: calls.append((node_id, vec))
    engine.graph = MagicMock(nodes={})

    rep = embedding_store.build_tenant_embedding(
        engine,
        tenant="acme",
        node_id="paper:1",
        node_type="ResearchPaper",
        text="self-improving agent harnesses",
    )

    assert rep.tenant == "acme"
    assert rep.target.node_id == "paper:1"
    assert rep.dimension == 3
    assert rep.artifact_ref == "paper:1"
    assert len(rep.content_hash) == 64
    assert calls == [("paper:1", [0.1, 0.2, 0.3])]
    assert len(_envelope_commit) == 1


def test_build_tenant_embedding_is_content_addressed():
    """Same text → same content_hash (the re-embed cache key), across runs."""
    engine = MagicMock()
    engine.add_embedding = MagicMock()
    engine.graph = MagicMock(nodes={})

    rep1 = embedding_store.build_tenant_embedding(
        engine, tenant="acme", node_id="p1", node_type="X", text="same text"
    )
    rep2 = embedding_store.build_tenant_embedding(
        engine, tenant="acme", node_id="p1", node_type="X", text="same text"
    )
    assert (
        rep1.content_hash
        == rep2.content_hash
        == embedding_store.content_hash("same text")
    )


def test_build_tenant_embedding_works_without_add_embedding_support(_envelope_commit):
    """An engine lacking native HNSW support degrades gracefully (record still committed)."""
    engine = MagicMock(spec=["graph"])
    engine.graph = MagicMock(nodes={})

    rep = embedding_store.build_tenant_embedding(
        engine, tenant="", node_id="p1", node_type="X", text="hello"
    )
    assert rep.artifact_ref == "p1"
    assert len(_envelope_commit) == 1
