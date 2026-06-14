"""Unit tests for the robust ConceptMatcher (CONCEPT:KG-2.75)."""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.assimilation.concept_matcher import (
    ConceptMatcher,
    _decide,
    _parse_judge,
    _top_k_cosine,
)
from agent_utilities.knowledge_graph.assimilation.concept_matcher import (
    Match,
)


class _Graph:
    def __init__(self, nodes: dict[str, dict]):
        self._n = nodes
        self._edges: list[tuple[str, str, dict]] = []

    def nodes(self, data: bool = False):
        return list(self._n.items()) if data else list(self._n)

    def out_edges(self, nid: str, data: bool = False):
        rows = [(s, t, p) for s, t, p in self._edges if s == nid]
        return rows if data else [(s, t) for s, t, _ in rows]


class _Engine:
    """Minimal engine double: graph.nodes/out_edges + link_nodes/delete_edge."""

    def __init__(self, nodes: dict[str, dict]):
        self.graph = _Graph(nodes)

    def link_nodes(self, src: str, dst: str, rel: Any, properties=None):
        self.graph._edges.append((src, dst, {"_rel": str(getattr(rel, "value", rel) or rel), **(properties or {})}))

    def delete_edge(self, src: str, dst: str, rel: str):
        self.graph._edges = [
            (s, t, p) for s, t, p in self.graph._edges
            if not (s == src and t == dst and p.get("_rel") == rel)
        ]


def _emb(vec):
    return list(vec)


# --- parsing / retrieval / fusion units ------------------------------------ #
def test_parse_judge_handles_json_and_garbage():
    assert _parse_judge('{"verdict":"covered","confidence":0.9,"why":"x"}')[:2] == ("covered", 0.9)
    assert _parse_judge("noise covered noise")[0] == "covered"
    assert _parse_judge("")[0] == "unrelated"


def test_top_k_cosine_orders_and_thresholds():
    cvecs = [("a", [1.0, 0.0]), ("b", [0.0, 1.0]), ("c", [1.0, 1.0])]
    out = _top_k_cosine([1.0, 0.0], cvecs, k=2, threshold=0.5)
    assert out[0][0] == "a"  # exact match ranks first
    assert "b" not in [c for c, _ in out]  # orthogonal pruned by threshold


def test_decide_prefers_covered_over_related():
    ms = [
        Match("KG-2.1", 0.7, "related", 0.8, 0.76, "llm_judge"),
        Match("KG-2.2", 0.8, "covered", 0.9, 0.86, "llm_judge"),
    ]
    fm = _decide("f", ms, judge_accept=0.6)
    assert fm.decision == "covered" and fm.best.concept_id == "KG-2.2"


# --- explicit-id stage ----------------------------------------------------- #
def test_explicit_id_match_is_covered():
    nodes = {
        "concept:KG-2.7": {"type": "concept", "concept_id": "KG-2.7", "name": "Research Assimilation"},
        "article:p1": {"type": "article", "name": "A paper about KG-2.7", "concept_ids": ["KG-2.7"]},
    }
    eng = _Engine(nodes)
    m = ConceptMatcher(embed_fn=lambda t: [[0.0] for _ in t], use_llm=False)
    rep = m.satisfy(eng, feature_types=("article",), concept_types=("concept",))
    assert rep.satisfied == 1
    assert any(p.get("_rel") == "SATISFIED_BY" and p.get("match") == "id" for *_, p in eng.graph._edges)


# --- embedding retrieval + LLM judge --------------------------------------- #
def test_llm_judge_covered_writes_satisfied_by():
    nodes = {
        "concept:KG-2.7": {"type": "concept", "concept_id": "KG-2.7", "name": "Research Assimilation", "embedding": [1.0, 0.0]},
        "article:p1": {"type": "article", "name": "Self-improving assimilation", "embedding": [0.99, 0.1]},
    }
    eng = _Engine(nodes)
    judge = lambda _p: '{"verdict":"covered","confidence":0.95,"why":"same"}'
    m = ConceptMatcher(embed_fn=_emb, llm_judge_fn=judge, use_llm=True)
    rep = m.satisfy(eng, feature_types=("article",), concept_types=("concept",))
    assert rep.satisfied == 1 and rep.used_llm
    e = [p for *_, p in eng.graph._edges][0]
    assert e["_rel"] == "SATISFIED_BY" and e["match"] == "llm_judge"


def test_llm_judge_related_writes_relates_to_and_stays_novel():
    nodes = {
        "concept:KG-2.7": {"type": "concept", "concept_id": "KG-2.7", "name": "Assimilation", "embedding": [1.0, 0.0]},
        "article:p1": {"type": "article", "name": "Novel adjacent method", "embedding": [0.9, 0.2]},
    }
    eng = _Engine(nodes)
    judge = lambda _p: '{"verdict":"related","confidence":0.8,"why":"adjacent"}'
    m = ConceptMatcher(embed_fn=_emb, llm_judge_fn=judge)
    rep = m.satisfy(eng, feature_types=("article",), concept_types=("concept",))
    assert rep.related == 1 and rep.satisfied == 0
    e = [p for *_, p in eng.graph._edges][0]
    assert e["_rel"] == "RELATES_TO" and e["novelty"] > 0  # still an open gap


def test_no_llm_falls_back_to_cosine_verdict():
    nodes = {
        "concept:KG-2.7": {"type": "concept", "concept_id": "KG-2.7", "name": "Assimilation", "embedding": [1.0, 0.0]},
        "article:p1": {"type": "article", "name": "near-duplicate", "embedding": [1.0, 0.0]},
    }
    eng = _Engine(nodes)
    m = ConceptMatcher(embed_fn=_emb, use_llm=False)  # no judge
    rep = m.satisfy(eng, feature_types=("article",), concept_types=("concept",))
    assert rep.satisfied == 1  # cosine 1.0 ≥ COVERED_COSINE → covered deterministically


def test_idempotent_reconcile_replaces_not_accumulates():
    nodes = {
        "concept:KG-2.7": {"type": "concept", "concept_id": "KG-2.7", "name": "A", "embedding": [1.0, 0.0]},
        "article:p1": {"type": "article", "name": "x", "embedding": [1.0, 0.0]},
    }
    eng = _Engine(nodes)
    m = ConceptMatcher(embed_fn=_emb, use_llm=False)
    m.satisfy(eng, feature_types=("article",), concept_types=("concept",))
    m.satisfy(eng, feature_types=("article",), concept_types=("concept",))
    sat = [p for *_, p in eng.graph._edges if p.get("_rel") == "SATISFIED_BY"]
    assert len(sat) == 1  # re-run replaced, did not duplicate
