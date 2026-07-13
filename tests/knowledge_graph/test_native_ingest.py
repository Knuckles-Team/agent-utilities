"""Shared native connector-ingestion primitive — fakes coverage.

Exercises ``ingest_entities`` / ``ingest_documents`` with a fake engine client (no
engine required). CONCEPT:AU-KG.ingest.enterprise-source-extractor.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.memory.native_ingest import (
    enrich_pending_documents,
    ingest_documents,
    ingest_entities,
)


class _FakeTxn:
    def __init__(self):
        self.nodes = {}
        self.committed = False

    def begin(self, graph=None):
        return "t"

    def add_node(self, txn, node_id, props):
        self.nodes[node_id] = props

    def commit(self, txn):
        self.committed = True
        return True


class _FakeEdges:
    def __init__(self):
        self.edges = []

    def add(self, s, d, p):
        self.edges.append((s, d, p))


class _FakeClient:
    def __init__(self):
        self.txn = _FakeTxn()
        self.edges = _FakeEdges()


def test_ingest_entities_writes_typed_nodes_and_stamps_provenance():
    c = _FakeClient()
    res = ingest_entities(
        [{"id": "erpnext:so:1", "type": "SalesOrder", "total": 10}],
        [{"source": "erpnext:so:1", "target": "erpnext:cust:1", "type": "orderedBy"}],
        source="erpnext-agent",
        domain="erpnext",
        client=c,
        graph="__commons__",
    )
    assert res == {"nodes": 1, "edges": 1}
    n = c.txn.nodes["erpnext:so:1"]
    assert n["type"] == "SalesOrder"
    assert n["source"] == "erpnext-agent" and n["domain"] == "erpnext"
    assert c.edges.edges[0] == ("erpnext:so:1", "erpnext:cust:1", {"type": "orderedBy"})


def test_ingest_documents_writes_document_nodes():
    c = _FakeClient()
    res = ingest_documents(
        [{"id": "sn:kb:1", "text": "reset vpn", "title": "VPN", "source_uri": "u"}],
        source="servicenow-api",
        domain="servicenow",
        client=c,
    )
    assert res == {"nodes": 1, "edges": 0}
    d = c.txn.nodes["sn:kb:1"]
    assert d["type"] == "Document"
    assert d["text"] == "reset vpn"
    assert d["title"] == "VPN"
    assert d["source"] == "servicenow-api"


def test_documents_without_text_are_skipped():
    c = _FakeClient()
    assert ingest_documents([{"id": "x"}], source="s", domain="d", client=c) is None


def test_noop_without_engine(monkeypatch):
    # Simulate no reachable engine so the test is deterministic regardless of
    # whether a live engine happens to be reachable from the test host.
    import agent_utilities.knowledge_graph.memory.native_ingest as ni

    monkeypatch.setattr(ni, "native_client", lambda: (None, ""))
    assert ingest_entities([{"id": "a", "type": "T"}], source="s", domain="d") is None
    assert ingest_documents([{"id": "a", "text": "t"}], source="s", domain="d") is None


def test_empty_is_noop():
    assert ingest_entities([], source="s", domain="d", client=_FakeClient()) is None


def test_ingest_documents_stamps_provenance_and_needs_enrichment_flag():
    """CONCEPT:AU-KG.enrichment.topic-classification-topology — every connector-written Document is
    flagged for the hub-side catch-up sweep and carries full provenance."""
    c = _FakeClient()
    res = ingest_documents(
        [{"id": "searxng:result:1", "text": "some search result body", "title": "R1"}],
        source="searxng-mcp",
        domain="searxng",
        client=c,
    )
    assert res == {"nodes": 1, "edges": 0}
    d = c.txn.nodes["searxng:result:1"]
    assert d["needs_enrichment"] is True
    assert d["fetched_at"]
    assert d["backend"] == "native_ingest"
    assert d["content_hash"]


class _FakeBackend:
    def __init__(self) -> None:
        self.add_node_calls: list[tuple[str, dict[str, Any]]] = []

    def add_node(self, node_id: str, **props: Any) -> None:
        self.add_node_calls.append((node_id, props))


class _FakeEngineForSweep:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.backend = _FakeBackend()

    def query_cypher(self, _q: str) -> list[dict[str, Any]]:
        return self._rows


@pytest.mark.asyncio
async def test_enrich_pending_documents_processes_and_clears_flag(monkeypatch):
    import agent_utilities.knowledge_graph.memory.native_ingest as ni

    rows = [
        {"d": {"id": "doc:a", "text": "hello world", "title": "A"}},
        {"d": {"id": "doc:b", "text": "goodbye world", "title": "B"}},
    ]
    engine = _FakeEngineForSweep(rows)

    processed: list[str] = []
    enriched_calls: list[str] = []

    class _FakeProcessor:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def process(self, _text: str, *, document_id: str, **_kw: Any) -> None:
            processed.append(document_id)

    class _FakeIngestionEngine:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        async def _enrich_text(
            self, source_id: str, _text: str, _source_type: str, _title: str = ""
        ) -> dict[str, int]:
            enriched_calls.append(source_id)
            return {"concepts": 1, "facts": 0, "topics": 1}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.document_processing.DocumentProcessor",
        _FakeProcessor,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.engine.IngestionEngine",
        _FakeIngestionEngine,
    )

    result = await enrich_pending_documents(engine, limit=10)

    assert result == {"scanned": 2, "enriched": 2, "failed": 0}
    assert processed == ["doc:a", "doc:b"]
    assert enriched_calls == ["doc:a", "doc:b"]
    cleared = {nid: props for nid, props in engine.backend.add_node_calls}
    assert cleared["doc:a"]["needs_enrichment"] is False
    assert cleared["doc:b"]["needs_enrichment"] is False


@pytest.mark.asyncio
async def test_enrich_pending_documents_one_bad_doc_does_not_abort_sweep(monkeypatch):
    rows = [
        {"d": {"id": "doc:bad", "text": "x"}},
        {"d": {"id": "doc:good", "text": "y"}},
    ]
    engine = _FakeEngineForSweep(rows)

    class _FailingProcessor:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def process(self, _text: str, *, document_id: str, **_kw: Any) -> None:
            if document_id == "doc:bad":
                raise RuntimeError("boom")

    class _FakeIngestionEngine:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        async def _enrich_text(self, *_a: Any, **_kw: Any) -> dict[str, int]:
            return {"concepts": 0, "facts": 0, "topics": 0}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.document_processing.DocumentProcessor",
        _FailingProcessor,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.engine.IngestionEngine",
        _FakeIngestionEngine,
    )

    result = await enrich_pending_documents(engine, limit=10)
    assert result == {"scanned": 2, "enriched": 1, "failed": 1}


@pytest.mark.asyncio
async def test_enrich_pending_documents_no_engine_capability_returns_zeroes():
    result = await enrich_pending_documents(object(), limit=10)
    assert result == {"scanned": 0, "enriched": 0, "failed": 0}


@pytest.mark.asyncio
async def test_enrich_pending_documents_empty_result_is_cheap_noop():
    engine = _FakeEngineForSweep([])
    result = await enrich_pending_documents(engine, limit=10)
    assert result == {"scanned": 0, "enriched": 0, "failed": 0}
