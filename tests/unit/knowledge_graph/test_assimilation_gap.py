#!/usr/bin/python
"""Auto gap analysis — SATISFIED_BY + open_features (VU-3).

CONCEPT:KG-2.7
"""

import pytest

from agent_utilities.knowledge_graph.assimilation import (
    auto_satisfy,
    is_closed,
    open_features,
)

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


def test_auto_satisfy_writes_edge_for_match():
    engine = _Engine(
        {
            "f1": _node("capability", [1.0, 0.0, 0.0]),  # matches concept
            "f2": _node("capability", [0.0, 1.0, 0.0]),  # no match
            "c1": _node("concept", [1.0, 0.0, 0.0]),
        }
    )
    report = auto_satisfy(engine, threshold=0.85)
    assert report.features == 2 and report.concepts == 1
    assert report.satisfied == 1
    assert report.candidates[0][0] == "f1" and report.candidates[0][1] == "c1"
    # f1 now has a SATISFIED_BY closing edge
    assert is_closed(engine, "f1")
    assert not is_closed(engine, "f2")


def test_open_features_excludes_satisfied_superseded_and_status():
    engine = _Engine(
        {
            "open1": _node("capability", [0.0, 1.0]),  # stays open
            "sat1": _node("capability", [1.0, 0.0]),  # will be satisfied
            "done1": _node("sdd_feature", [0.2, 0.2], status="implemented"),
            "dup1": _node("capability", [0.5, 0.5]),  # superseded by an edge
            "c1": _node("concept", [1.0, 0.0]),
        }
    )
    # auto-satisfy closes sat1 (matches c1); dup1 closed via a SUPERSEDES in-edge.
    auto_satisfy(engine, threshold=0.85)
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


def test_no_concepts_means_nothing_satisfied():
    engine = _Engine({"f1": _node("capability", [1.0, 0.0])})
    report = auto_satisfy(engine)
    assert report.satisfied == 0
    assert open_features(engine) == ["f1"]


def test_dry_run_does_not_write():
    engine = _Engine(
        {"f1": _node("capability", [1.0, 0.0]), "c1": _node("concept", [1.0, 0.0])}
    )
    report = auto_satisfy(engine, write=False)
    assert report.satisfied == 1  # detected
    assert not is_closed(engine, "f1")  # but no edge written


def test_auto_satisfy_id_reference_beats_embedding():
    """An explicit concept-id reference wins over the embedding-closest concept.

    Regression for the calibration finding: pure cosine recognized 0/21 built
    capabilities (argmax wrong 71% of the time). The feature here is embedding-
    identical to the WRONG concept (KG-9.9) yet references KG-2.7 — the id signal
    must win, exact, score 1.0.
    """
    engine = _Engine(
        {
            "spec:foo": {
                "type": "capability",
                "status": "open",
                "embedding": [0.0, 1.0, 0.0],
                "concept_ids": ["KG-2.7"],
            },
            "KG-2.7": {
                "type": "concept",
                "status": "live",
                "concept_id": "KG-2.7",
                "embedding": [1.0, 0.0, 0.0],  # orthogonal to the feature
            },
            "KG-9.9": {
                "type": "concept",
                "status": "live",
                "concept_id": "KG-9.9",
                "embedding": [0.0, 1.0, 0.0],  # embedding-identical, but NOT referenced
            },
        }
    )
    report = auto_satisfy(engine)
    assert ("spec:foo", "KG-2.7", 1.0) in report.candidates
    assert is_closed(engine, "spec:foo")


def test_auto_satisfy_id_from_content_marker():
    """A CONCEPT:<ID> marker in the node's content text is recognized."""
    engine = _Engine(
        {
            "spec:bar": {
                "type": "capability",
                "status": "open",
                "embedding": [0.0, 1.0, 0.0],
                "content": "This implements CONCEPT:AHE-3.1 in-house training.",
            },
            "AHE-3.1": {
                "type": "concept",
                "status": "live",
                "concept_id": "AHE-3.1",
                "embedding": [1.0, 0.0, 0.0],
            },
        }
    )
    report = auto_satisfy(engine)
    assert ("spec:bar", "AHE-3.1", 1.0) in report.candidates
