#!/usr/bin/python
"""Feature lifecycle ledger + assimilation close-out (VU-5).

CONCEPT:AU-KG.query.vendor-agnostic-traversal
"""

import pytest

from agent_utilities.knowledge_graph.assimilation import (
    close_out,
    is_closed,
    ledger_state,
    open_features,
    promote_feature_ledger,
    record_feature,
    set_status,
)

pytestmark = pytest.mark.concept("AU-KG.query.vendor-agnostic-traversal")


class _Graph:
    def __init__(self):
        self._n: dict = {}
        self._out: dict = {}
        self._in: dict = {}

    def nodes(self, data=False):
        return list(self._n.items()) if data else list(self._n)

    def add_edge(self, src, dst, props):
        self._out.setdefault(src, []).append((src, dst, props))
        self._in.setdefault(dst, []).append((src, dst, props))

    def out_edges(self, nid, data=False):
        e = self._out.get(nid, [])
        return e if data else [(s, t) for s, t, _ in e]

    def in_edges(self, nid, data=False):
        e = self._in.get(nid, [])
        return e if data else [(s, t) for s, t, _ in e]


class _Engine:
    def __init__(self):
        self.graph = _Graph()

    def add_node(self, node_id, node_type, properties=None, ephemeral=False):
        self.graph._n[node_id] = {**(properties or {}), "type": node_type}

    def link_nodes(self, src, dst, rel_type, properties=None, ephemeral=False):
        self.graph.add_edge(src, dst, properties or {})


def test_record_and_set_status():
    engine = _Engine()
    record_feature(
        engine, feature_id="f1", name="exec-rag planner", concept_ids=["AU-KG.retrieval.memory-first-retrieval"]
    )
    st = ledger_state(engine)
    assert st["total"] == 1 and st["open"] == 1 and st["by_status"]["open"] == 1
    assert set_status(engine, "f1", "implemented")
    assert is_closed(engine, "f1", "implemented")
    assert set_status(engine, "missing", "x") is False


def test_close_out_writes_provenance_and_closes():
    engine = _Engine()
    record_feature(
        engine,
        feature_id="f1",
        name="ontology bootstrap",
        research_sources=["paper:productkg"],
        codebase="agent-utilities",
    )
    report = close_out(engine, "f1")
    assert report.derived_from == 1 and report.assimilated == 1
    # DERIVED_FROM_RESEARCH: feature → source
    der = [
        e
        for e in engine.graph._out["f1"]
        if e[2].get("_rel") == "DERIVED_FROM_RESEARCH"
    ]
    assert der and der[0][1] == "paper:productkg"
    # ASSIMILATED_INTO: source → codebase
    assim = [
        e
        for e in engine.graph._out["paper:productkg"]
        if e[2].get("_rel") == "ASSIMILATED_INTO"
    ]
    assert assim and assim[0][1] == "agent-utilities"
    # the feature is now closed and excluded from open_features
    assert is_closed(engine, "f1")
    assert "f1" not in set(open_features(engine))


def test_close_out_without_codebase_skips_assimilated():
    engine = _Engine()
    record_feature(engine, feature_id="f1", name="x", research_sources=["paper:y"])
    report = close_out(engine, "f1")
    assert report.derived_from == 1 and report.assimilated == 0


def test_promote_feature_ledger():
    engine = _Engine()
    rows = [
        {
            "id": "graph_pagerank",
            "name": "pagerank",
            "concept": "AU-KG.compute.spectral-cluster-navigator",
            "source": "graph.py:10",
            "status": "live",
        },
        {
            "id": "x_unknown",
            "name": "x",
            "concept": "UNKNOWN",
            "source": "x.py:1",
            "status": "stubbed-intended",
        },
        {"name": "no-id-skipped"},
    ]
    n = promote_feature_ledger(engine, rows)
    assert n == 2  # the id-less row is skipped
    data = dict(engine.graph.nodes(data=True))
    assert data["graph_pagerank"]["concept_ids"] == ["AU-KG.compute.spectral-cluster-navigator"]
    assert data["x_unknown"]["concept_ids"] == []  # UNKNOWN dropped


def test_ledger_state_buckets():
    engine = _Engine()
    record_feature(engine, feature_id="a", name="a", status="open")
    record_feature(engine, feature_id="b", name="b", status="open")
    close_out(engine, "b", codebase="agent-utilities", status="implemented")
    st = ledger_state(engine)
    assert st["total"] == 2
    assert st["open"] == 1 and st["closed"] == 1
    assert st["by_status"]["implemented"] == 1
