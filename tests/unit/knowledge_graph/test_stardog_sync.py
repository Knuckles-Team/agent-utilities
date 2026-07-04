"""Unit tests for explicit Stardog push/pull (CONCEPT:AU-KG.query.vendor-agnostic-traversal).

Uses a fake source graph + fake backend so the per-source partitioning and the
push/pull orchestration are asserted without pystardog or a live server.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.integrations.stardog_sync import (
    pull_from_stardog,
    push_to_stardog,
)


class _FakeGraph:
    """Minimal L1-compute shape that ``migration._iter_source_*`` understands."""

    def __init__(self, nodes, edges):
        self._nodes = nodes  # {id: props}
        self._edges = edges  # [(src, tgt, {"type":..., ...})]

    def _get_all_nodes(self):
        return list(self._nodes)

    def _get_node_properties(self, nid):
        return self._nodes.get(nid, {})

    def edges(self, data=False):
        return [(s, t, d) for (s, t, d) in self._edges]


class _FakeSource:
    def __init__(self, graph):
        self.graph = graph


class _FakeStardogBackend:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node_id, properties=None):
        self.nodes.append((node_id, dict(properties or {})))

    def add_edge(self, source_id, target_id, properties=None):
        self.edges.append((source_id, target_id, dict(properties or {})))


def test_push_all_sources_partitions_by_provenance():
    graph = _FakeGraph(
        nodes={
            "app:1": {"type": "Application", "source_system": "leanix"},
            "sn:1": {"type": "Request", "source_system": "servicenow"},
            "ep:1": {"type": "Episode"},  # internal, untagged → default graph
        },
        edges=[("app:1", "sn:1", {"type": "RELATES_TO", "source_system": "leanix"})],
    )
    be = _FakeStardogBackend()
    res = push_to_stardog(_FakeSource(graph), be)
    assert res["status"] == "ok"
    assert res["nodes"] == 3 and res["edges"] == 1
    assert res["graphs"]["urn:source:leanix"]["nodes"] == 1
    assert res["graphs"]["urn:source:servicenow"]["nodes"] == 1
    assert res["graphs"]["default"]["nodes"] == 1


def test_push_subset_filters_by_source():
    graph = _FakeGraph(
        nodes={
            "app:1": {"type": "Application", "source_system": "leanix"},
            "sn:1": {"type": "Request", "source_system": "servicenow"},
        },
        edges=[],
    )
    be = _FakeStardogBackend()
    res = push_to_stardog(_FakeSource(graph), be, sources=["leanix"])
    assert res["nodes"] == 1
    assert be.nodes[0][0] == "app:1"


class _PullBackend:
    """Fake Stardog returning canned SPARQL rows for the pull queries."""

    def __init__(self):
        self.calls = 0

    def execute_sparql_query(self, query, timeout_ms=30_000):
        q = query.upper()
        if " A ?T" in q and "FILTER(STRSTARTS" in query:
            return [
                {
                    "s": "http://agent-utilities.dev/kg#app:1",
                    "t": "http://agent-utilities.dev/kg#Application",
                }
            ]
        if "ISLITERAL(?O)" in q:
            return [{"p": "http://agent-utilities.dev/kg#name", "o": "Billing"}]
        if "ISIRI(?O)" in q:
            return []
        return []


class _RecordingEngine:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node_id, node_type, properties=None):
        self.nodes.append((node_id, node_type, dict(properties or {})))

    def link_nodes(self, s, t, rel):
        self.edges.append((s, t, rel))


def test_pull_materialises_nodes_and_props():
    eng = _RecordingEngine()
    res = pull_from_stardog(_PullBackend(), eng, source="leanix")
    assert res["status"] == "ok"
    assert res["graph"] == "urn:source:leanix"
    assert eng.nodes == [("app:1", "Application", {"name": "Billing"})]
