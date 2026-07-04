"""Shared native connector-ingestion primitive — fakes coverage.

Exercises ``ingest_entities`` / ``ingest_documents`` with a fake engine client (no
engine required). CONCEPT:AU-KG.ingest.enterprise-source-extractor.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.memory.native_ingest import (
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
