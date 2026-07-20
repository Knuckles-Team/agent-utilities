"""Tests for confidence aggregation + support-count edge weighting (KG-2.257)."""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.extraction.fact_extractor import (
    ExtractedFact,
    aggregate_confidence,
    persist_facts,
)


class FakeStore:
    """Records add_node / add_edge calls for assertion."""

    def __init__(self) -> None:
        self.nodes: list[tuple[str, dict[str, Any]]] = []
        self.edges: list[tuple[str, str, dict[str, Any]]] = []

    def add_node(self, key: str, **props: Any) -> None:
        self.nodes.append((key, props))

    def add_edge(self, source: str, target: str, **props: Any) -> None:
        self.edges.append((source, target, props))


def test_aggregate_confidence_product_complement():
    assert aggregate_confidence([]) == 0.0
    assert aggregate_confidence([0.8]) == pytest.approx(0.8)
    assert aggregate_confidence([0.5, 0.5]) == pytest.approx(0.75)
    assert aggregate_confidence([0.5, 0.5, 0.5]) == pytest.approx(0.875)
    # clamps out-of-range inputs
    assert aggregate_confidence([1.5, -0.2]) == pytest.approx(1.0)


def _fact(s: str, p: str, o: str, conf: int, src: str = "", tags=None) -> ExtractedFact:
    return ExtractedFact(
        subject=s,
        predicate=p,
        object=o,
        confidence=conf,
        source_file=src,
        tags=tags or [],
    )


def test_repeated_triple_merges_into_one_weighted_edge():
    store = FakeStore()
    facts = [
        _fact("Bob", "works_for", "Acme", 50, src="doc1", tags=["a"]),
        _fact("Bob", "works_for", "Acme", 50, src="doc2", tags=["b"]),
        _fact("Bob", "leads", "Acme", 90, src="doc1"),
    ]
    out = persist_facts(store, facts)
    # two distinct edges (works_for merged, leads separate)
    assert out["edges"] == 2
    assert len(store.edges) == 2

    works = next(e for e in store.edges if e[2]["rel_type"] == "works_for")
    _s, _o, props = works
    assert props["support_count"] == 2
    assert props["weight"] == 2.0
    # product-complement of two 0.5 mentions
    assert props["confidence"] == pytest.approx(0.75)
    # tags unioned, two distinct source documents
    assert set(props["tags"]) == {"a", "b"}
    assert props["support_documents"] == 2


def test_singleton_edge_weight_one():
    store = FakeStore()
    persist_facts(store, [_fact("Bob", "leads", "Acme", 90, src="doc1")])
    _s, _o, props = store.edges[0]
    assert props["support_count"] == 1
    assert props["weight"] == 1.0
    assert props["confidence"] == pytest.approx(0.9)


def test_duplicates_skipped():
    store = FakeStore()
    f = _fact("Bob", "works_for", "Acme", 50)
    f.is_duplicate = True
    out = persist_facts(store, [f])
    assert out["edges"] == 0
    assert store.edges == []


def test_corroborated_edge_outranks_singleton():
    """A 2×0.5 corroborated edge should carry higher confidence than a 0.6 singleton."""
    store = FakeStore()
    persist_facts(
        store,
        [
            _fact("Bob", "works_for", "Acme", 50, src="d1"),
            _fact("Bob", "works_for", "Acme", 50, src="d2"),
            _fact("Carl", "works_for", "Acme", 60, src="d1"),
        ],
    )
    by_subj = {e[0]: e[2] for e in store.edges}
    bob = by_subj[ExtractedFact.normalize_key("Bob")]
    carl = by_subj[ExtractedFact.normalize_key("Carl")]
    assert bob["confidence"] > carl["confidence"]
    assert bob["weight"] > carl["weight"]
