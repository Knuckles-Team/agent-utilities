#!/usr/bin/python
"""Tests for the hierarchical (tree-navigation) retriever (CONCEPT:AU-KG.retrieval.tree-navigation).

Deterministic + offline: an in-memory section tree is searched with the default
dependency-free lexical scorer (no model/network). A fake engine exercises the
graph load path (Section-node rows -> rebuilt tree).
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.ontology.document_processing import (
    SectionTreeConfig,
    build_section_tree,
    section_nodes_and_edges,
)
from agent_utilities.knowledge_graph.retrieval.hierarchical_document_retriever import (
    HierarchicalDocumentRetriever,
    SectionMatch,
    build_hierarchical_retriever,
    content_for_ranges,
    structure_view,
)

MD = """# Product Manual
High-level overview of the product and this manual.

## Installation
Install with pip; configure the vector database connection and credentials.

## Retrieval
Vectorless reasoning-tree navigation walks the section tree and avoids the
embedder recall ceiling that hurts long single documents.

## Billing and refunds
Payment methods, invoices, refunds, and subscription tier changes are covered.
"""


def _tree():
    # thin=False keeps each heading a node (the doc is short); summaries off to
    # also exercise the retriever's body-text scoring fallback.
    return build_section_tree(MD, config=SectionTreeConfig(thin=False))


def test_retrieve_ranks_relevant_section_top():
    r = HierarchicalDocumentRetriever()
    matches = r.retrieve(
        "how does tree navigation avoid the recall ceiling", tree=_tree(), top_k=3
    )
    assert matches
    assert isinstance(matches[0], SectionMatch)
    assert matches[0].title == "Retrieval"
    # Cited range slices the source.
    assert MD[matches[0].char_start : matches[0].char_end].strip()


def test_similar_not_relevant_prefers_the_right_branch():
    r = HierarchicalDocumentRetriever()
    matches = r.retrieve("refund my subscription payment", tree=_tree(), top_k=2)
    assert matches[0].title == "Billing and refunds"


def test_structure_view_is_text_free():
    view = structure_view(_tree())
    flat = _flatten(view)
    assert all("range" in v and "title" in v for v in flat)
    assert all("text" not in v and "content" not in v for v in flat)


def test_content_for_ranges_returns_intersecting_sections():
    tree = _tree()
    target = tree[0].children[-1]  # Billing
    hits = content_for_ranges(tree, [(target.char_start, target.char_end)])
    assert [h["title"] for h in hits] == ["Billing and refunds"]
    assert "refunds" in hits[0]["content"].lower()


def test_load_tree_from_fake_engine():
    tree = _tree()
    nodes, _edges = section_nodes_and_edges("doc:9", tree)

    class _FakeEngine:
        def query_cypher(self, cypher, params):
            assert params["doc"] == "doc:9"
            return nodes

    r = build_hierarchical_retriever(_FakeEngine())
    loaded = r.load_tree("doc:9")
    assert loaded and loaded[0].title == "Product Manual"
    matches = r.retrieve("installation credentials", document_id="doc:9", top_k=2)
    assert matches[0].title == "Installation"


def test_empty_tree_returns_no_matches():
    r = HierarchicalDocumentRetriever()
    assert r.retrieve("anything", tree=[], top_k=3) == []


def _flatten(view):
    out = []
    for v in view:
        out.append(v)
        out.extend(_flatten(v.get("children", [])))
    return out
