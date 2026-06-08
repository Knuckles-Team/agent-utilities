#!/usr/bin/python
"""Tests for hierarchical (global→local) community retrieval (b2-04).

CONCEPT:KG-2.5
"""

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.core.hierarchical_retrieval import (
    HierarchicalCommunityRetriever,
    HierarchicalResult,
)
from agent_utilities.knowledge_graph.core.topological_analysis_engine import (
    TopologicalAnalysisEngine,
)

pytestmark = pytest.mark.concept("KG-2.5")


def _graph():
    g = GraphComputeEngine(backend_type="rust")
    # Community A — retrieval/vector topic
    g.add_node("a1", name="vector index hnsw tuning")
    g.add_node("a2", name="graph database vector search")
    g.add_edge("a1", "a2", type="related")
    # Community B — cooking topic
    g.add_node("b1", name="cooking pasta recipe")
    g.add_node("b2", name="tomato sauce simmer")
    g.add_edge("b1", "b2", type="related")
    return g


_COMMS = [{"a1", "a2"}, {"b1", "b2"}]


def test_global_then_local_ranks_relevant_community_first():
    r = HierarchicalCommunityRetriever(_graph())
    res = r.retrieve(
        "vector index", communities=_COMMS, top_communities=1, top_entities=5
    )
    assert isinstance(res, HierarchicalResult)
    assert res.communities_searched == 1
    assert res.hits
    assert all(h.id in {"a1", "a2"} for h in res.hits)  # drilled into community A only
    assert res.hits[0].community_score > 0


def test_parent_context_boost_orders_within_relevant_community():
    r = HierarchicalCommunityRetriever(_graph(), parent_weight=0.3)
    res = r.retrieve(
        "vector index", communities=_COMMS, top_communities=2, top_entities=10
    )
    a_hits = [h for h in res.hits if h.community_index == 0]
    b_hits = [h for h in res.hits if h.community_index == 1]
    # entities in the query-relevant community outrank the irrelevant community
    assert min(h.score for h in a_hits) >= max((h.score for h in b_hits), default=0.0)


def test_empty_communities_returns_no_hits():
    r = HierarchicalCommunityRetriever(_graph())
    res = r.retrieve("anything", communities=[], top_entities=5)
    assert res.hits == []


def test_embedder_injection():
    embedder = type(
        "E",
        (),
        {
            "score": staticmethod(
                lambda q, t: 1.0 if "a1" in t or "vector" in t else 0.0
            )
        },
    )()
    r = HierarchicalCommunityRetriever(_graph(), embedder=embedder)
    res = r.retrieve("q", communities=_COMMS, top_communities=1)
    assert res.hits and res.hits[0].community_index == 0


# --- live engine method -----------------------------------------------------


def test_engine_hierarchical_retrieve_is_live():
    eng = TopologicalAnalysisEngine(graph=_graph())
    res = eng.hierarchical_retrieve("vector index", top_communities=2, top_entities=5)
    assert isinstance(res, HierarchicalResult)
    assert res.query == "vector index"
    # with edge-formed communities, any returned top hit is from the vector community
    if res.hits:
        assert res.hits[0].id in {"a1", "a2"}


def test_engine_no_graph_returns_empty():
    eng = TopologicalAnalysisEngine(graph=None)
    res = eng.hierarchical_retrieve("q")
    assert isinstance(res, HierarchicalResult)
    assert res.hits == []
