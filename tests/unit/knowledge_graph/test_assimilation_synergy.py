#!/usr/bin/python
"""Synergy bundles + leverage ranking (VU-4).

CONCEPT:KG-2.7
"""

import pytest

from agent_utilities.knowledge_graph.assimilation import (
    rank_features,
    synergy_bundles,
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


def _f(concept, sources=(), status="open"):
    return {
        "type": "capability",
        "concept_ids": [concept],
        "research_sources": list(sources),
        "status": status,
    }


def test_cross_pillar_community_becomes_bundle():
    engine = _Engine(
        {"f_kg": _f("KG-2.1"), "f_orch": _f("ORCH-1.2"), "f_iso": _f("AHE-3.0")}
    )
    engine.link_nodes("f_kg", "f_orch", "SIMILAR_TO", properties={"_rel": "SIMILAR_TO"})
    report = synergy_bundles(engine, min_pillars=2)
    assert len(report.bundles) == 1
    b = report.bundles[0]
    assert b.members == ["f_kg", "f_orch"] and b.pillars == ["KG", "ORCH"]
    # HAS_SYNERGY_WITH written between the pair
    syn = [
        e
        for e in engine.graph._out.get("f_kg", [])
        if e[2].get("_rel") == "HAS_SYNERGY_WITH"
    ]
    assert syn and syn[0][1] == "f_orch"


def test_same_pillar_community_is_not_a_bundle():
    engine = _Engine({"a": _f("KG-2.1"), "b": _f("KG-2.5")})
    engine.link_nodes("a", "b", "SIMILAR_TO", properties={"_rel": "SIMILAR_TO"})
    report = synergy_bundles(engine, min_pillars=2)
    assert report.bundles == []  # single pillar → no synergy


def test_supersedes_edges_excluded_from_synergy():
    engine = _Engine({"a": _f("KG-2.1"), "b": _f("ORCH-1.2")})
    # only a duplicate edge connects them → not a synergy
    engine.link_nodes("a", "b", "SUPERSEDES", properties={"_rel": "SUPERSEDES"})
    report = synergy_bundles(engine, min_pillars=2)
    assert report.bundles == []


def test_rank_features_by_leverage():
    engine = _Engine(
        {
            "hi": _f("KG-2.1", sources=["p1", "p2", "p3"]),  # 3 sources
            "lo": _f("ORCH-1.2", sources=["p1"]),  # 1 source
        }
    )
    ranked = rank_features(engine)
    assert [r.feature_id for r in ranked] == ["hi", "lo"]
    assert ranked[0].source_count == 3 and ranked[0].score >= ranked[1].score


def test_rank_features_uses_open_only():
    engine = _Engine(
        {
            "open1": _f("KG-2.1", sources=["p1"]),
            "done1": _f("ORCH-1.2", sources=["p1", "p2"], status="implemented"),
        }
    )
    ranked = rank_features(engine)  # feature_ids=None → open_features()
    ids = {r.feature_id for r in ranked}
    assert ids == {"open1"}  # the implemented one is excluded despite more sources


def test_engine_pagerank_fast_path_used(monkeypatch):
    # Engine PageRank is opt-in (a full-graph op, too slow on a live backend for a
    # few-dozen-feature rank); enable it explicitly to exercise this path.
    monkeypatch.setenv("ASSIMILATION_ENGINE_PAGERANK", "1")

    class _PR(_Engine):
        def pagerank(self):
            return [("a", 0.9), ("b", 0.1)]

    engine = _PR({"a": _f("KG-2.1", sources=["p1"]), "b": _f("KG-2.5", sources=["p1"])})
    ranked = rank_features(engine, feature_ids=["a", "b"])
    # 'a' has higher centrality from the engine → ranks first at equal source_count
    assert ranked[0].feature_id == "a" and ranked[0].centrality == 0.9
