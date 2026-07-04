"""GraphComputeEngine RDF retract + SQL-write wrappers (CONCEPT:AU-KG.ingest.mirror-inbound).

The engine gained ``RemoveTriples`` / ``DropNamedGraph`` wire ops and a write-capable
SQL surface. These tests assert the Python wrappers call the right path: a typed
``client.rdf`` wrapper when present, else the raw wire op, and ``sql_exec`` through
``client.query.sql``. They construct a bare ``GraphComputeEngine`` instance without
running ``__init__`` so no real engine connection is needed.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

pytestmark = pytest.mark.concept("AU-KG.ingest.mirror-inbound")


def _engine_with_client(client):
    eng = GraphComputeEngine.__new__(GraphComputeEngine)
    eng._client = client
    return eng


class _RdfTyped:
    """An rdf namespace that DOES expose typed wrappers."""

    def __init__(self):
        self.calls = []

    def remove_triples(self, turtle=None, ntriples=None):
        self.calls.append(("remove_triples", turtle, ntriples))
        return {"removed": 3}

    def drop_named_graph(self, graph):
        self.calls.append(("drop_named_graph", graph))
        return {"dropped": graph}


def test_remove_triples_prefers_typed_wrapper():
    rdf = _RdfTyped()
    eng = _engine_with_client(type("C", (), {"rdf": rdf})())
    out = eng.remove_triples(turtle="@prefix : <x> . :a :b :c .")
    assert out == {"removed": 3}
    assert rdf.calls[0][0] == "remove_triples"


def test_drop_named_graph_prefers_typed_wrapper():
    rdf = _RdfTyped()
    eng = _engine_with_client(type("C", (), {"rdf": rdf})())
    assert eng.drop_named_graph("urn:g:1") == {"dropped": "urn:g:1"}


def test_remove_triples_falls_back_to_wire_op():
    """No typed wrapper → raw RemoveTriples wire op via _send_wire."""
    sent = {}

    eng = _engine_with_client(type("C", (), {"rdf": object()})())  # rdf has no methods

    def fake_send_wire(method, payload=None):
        sent["method"] = method
        sent["payload"] = payload
        return {"removed": 1}

    eng._send_wire = fake_send_wire  # type: ignore[assignment]
    out = eng.remove_triples(ntriples="<a> <b> <c> .")
    assert out == {"removed": 1}
    assert sent["method"] == "RemoveTriples"
    assert sent["payload"]["ntriples"] == "<a> <b> <c> ."


def test_drop_named_graph_falls_back_to_wire_op():
    sent = {}
    eng = _engine_with_client(type("C", (), {"rdf": object()})())
    eng._send_wire = lambda method, payload=None: sent.update(  # type: ignore[assignment]
        method=method, payload=payload
    )
    eng.drop_named_graph("urn:g:2")
    assert sent["method"] == "DropNamedGraph"
    assert sent["payload"] == {"graph": "urn:g:2"}


def test_sql_exec_routes_through_query_sql():
    seen = {}

    class _Query:
        def sql(self, statement):
            seen["stmt"] = statement
            return {"ok": True}

    eng = _engine_with_client(type("C", (), {"query": _Query()})())
    out = eng.sql_exec("CREATE TABLE t (id VARCHAR)")
    assert out == {"ok": True}
    assert seen["stmt"] == "CREATE TABLE t (id VARCHAR)"


def test_sql_exec_no_surface_raises():
    eng = _engine_with_client(type("C", (), {"query": object()})())
    with pytest.raises(RuntimeError, match="no epistemic-graph SQL surface"):
        eng.sql_exec("CREATE TABLE t (id VARCHAR)")
