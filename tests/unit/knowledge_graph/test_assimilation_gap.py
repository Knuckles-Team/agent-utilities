#!/usr/bin/python
"""Gap-analysis helpers — open_features / is_closed (VU-3).

The feature→concept *matching* is now the ConceptMatcher (see
``test_concept_matcher.py``); this file covers the durable "what is still an open
gap?" query helpers that the matcher and the golden loop both rely on.

CONCEPT:KG-2.7
"""

import pytest

from agent_utilities.knowledge_graph.assimilation import is_closed, open_features

pytestmark = pytest.mark.concept("KG-2.7")


class _Graph:
    def __init__(self, nodes):
        self._n = nodes
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
    def __init__(self, nodes):
        self.graph = _Graph(nodes)

    def link_nodes(self, src, dst, rel_type, properties=None, ephemeral=False):
        self.graph.add_edge(src, dst, properties or {})


def _node(ntype, emb=None, status="open"):
    d = {"type": ntype, "status": status}
    if emb is not None:
        d["embedding"] = emb
    return d


def test_open_features_excludes_satisfied_superseded_and_status():
    engine = _Engine(
        {
            "open1": _node("capability", [0.0, 1.0]),  # stays open
            "sat1": _node("capability", [1.0, 0.0]),  # closed by SATISFIED_BY
            "done1": _node("sdd_feature", [0.2, 0.2], status="implemented"),
            "dup1": _node("capability", [0.5, 0.5]),  # superseded by an edge
            "c1": _node("concept", [1.0, 0.0]),
        }
    )
    # A SATISFIED_BY edge closes sat1; dup1 closed via a SUPERSEDES in-edge.
    engine.link_nodes("sat1", "c1", "SATISFIED_BY", properties={"_rel": "SATISFIED_BY"})
    engine.link_nodes(
        "survivor", "dup1", "SUPERSEDES", properties={"_rel": "SUPERSEDES"}
    )

    opens = set(open_features(engine))
    assert "open1" in opens
    assert "sat1" not in opens  # SATISFIED_BY
    assert "done1" not in opens  # status=implemented
    assert "dup1" not in opens  # superseded
    assert "c1" not in opens  # concept is not a feature type


def test_is_closed_by_status():
    engine = _Engine({"f": _node("capability", [1.0], status="rejected")})
    assert is_closed(engine, "f", "rejected")


class _BulkGraph(_Graph):
    """Adds an NX-style bulk ``edges`` view so the batched path is exercised."""

    def edges(self, data=False):
        out = []
        for lst in self._out.values():
            for s, d, p in lst:
                out.append((s, d, p) if data else (s, d))
        return out


class _BulkEngine(_Engine):
    def __init__(self, nodes):
        self.graph = _BulkGraph(nodes)
        self.deleted = []

    def delete_edge(self, src, dst, rel_type=None, ephemeral=False):
        self.deleted.append((src, dst, rel_type))
        self.graph._out[src] = [e for e in self.graph._out.get(src, []) if e[1] != dst]
        self.graph._in[dst] = [e for e in self.graph._in.get(dst, []) if e[0] != src]


def test_open_features_uses_bulk_edge_view():
    """Batched closed-index: a SATISFIED_BY edge closes a feature via the bulk view."""
    eng = _BulkEngine(
        {
            "f1": _node("capability"),
            "f2": _node("capability"),
            "c1": _node("concept"),
        }
    )
    eng.link_nodes("f1", "c1", "SATISFIED_BY", properties={"_rel": "SATISFIED_BY"})
    openf = open_features(eng, feature_types=("capability",))
    assert openf == ["f2"]  # f1 closed via the bulk-scanned edge
