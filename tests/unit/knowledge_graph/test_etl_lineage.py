"""Unit tests for ETL data lineage (CONCEPT:AU-KG.ontology.one-source).

Asserts record_etl_run creates a PROVENANCE_AGENT run node + WAS_DERIVED_FROM edges,
and query_lineage reads them back — using a fake engine/backend.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.etl.lineage import query_lineage, record_etl_run


class _FakeEngine:
    def __init__(self, rows=None):
        self.nodes = []
        self.edges = []
        self.backend = _FakeBackend(rows or [])

    def add_node(self, node_id, node_type, properties=None):
        self.nodes.append((node_id, str(node_type), dict(properties or {})))

    def link_nodes(self, s, t, rel):
        self.edges.append((s, t, str(rel)))


class _FakeBackend:
    def __init__(self, rows):
        self._rows = rows
        self.last_query = None
        self.last_params = None

    def execute(self, query, params=None):
        self.last_query = query
        self.last_params = params
        return self._rows


def test_record_etl_run_creates_run_node_and_edges():
    eng = _FakeEngine()
    run_id = record_etl_run(
        eng,
        source="servicenow",
        sink="leanix",
        direction="through",
        counts={"nodes": 5, "edges": 2},
        at=1_700_000_000.0,
    )
    assert run_id and run_id.startswith("etl-run:servicenow:leanix:")
    # run node + 2 system markers
    run_node = next(n for n in eng.nodes if n[0] == run_id)
    assert run_node[2]["kind"] == "etl_run"
    assert run_node[2]["source"] == "servicenow" and run_node[2]["sink"] == "leanix"
    assert run_node[2]["nodes"] == 5
    marker_ids = {n[0] for n in eng.nodes}
    assert "urn:source:servicenow" in marker_ids
    assert "urn:sink:leanix" in marker_ids
    # WAS_DERIVED_FROM chain: run→source, sink→run
    assert (run_id, "urn:source:servicenow", "was_derived_from") in eng.edges
    assert ("urn:sink:leanix", run_id, "was_derived_from") in eng.edges


def test_record_etl_run_inbound_only_no_sink_marker():
    eng = _FakeEngine()
    record_etl_run(eng, source="leanix", sink=None, direction="inbound", counts={})
    marker_ids = {n[0] for n in eng.nodes}
    assert "urn:source:leanix" in marker_ids
    assert not any(m.startswith("urn:sink:") for m in marker_ids)


def test_query_lineage_filters_and_reads():
    rows = [
        {
            "id": "etl-run:servicenow:leanix:1",
            "source": "servicenow",
            "sink": "leanix",
            "direction": "through",
            "nodes": 5,
            "edges": 2,
            "status": "ok",
            "at": 1_700_000_000.0,
        }
    ]
    eng = _FakeEngine(rows=rows)
    out = query_lineage(eng, source="servicenow")
    assert out == rows
    assert eng.backend.last_params["kind"] == "etl_run"
    assert eng.backend.last_params["source"] == "servicenow"
    assert "n.kind = $kind" in eng.backend.last_query


def test_query_lineage_no_backend_is_empty():
    class _NoBackend:
        backend = None

    assert query_lineage(_NoBackend()) == []
