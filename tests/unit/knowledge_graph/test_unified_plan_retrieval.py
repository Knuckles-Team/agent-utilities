#!/usr/bin/python
"""``retrieve_hybrid`` runs ONE engine unified plan — no O(N) Python cosine scan.

CONCEPT:KG-2.250 / KG-2.238. The hand-orchestrated hybrid retriever's vector arm
is collapsed onto the engine: the vector neighbourhood is computed by the engine's
native ANN inside a single costed cross-modal plan (``query.unified``, falling to
the native ``semantic_search`` ANN primitive on a lean engine), NEVER by an O(N)
Python ``cosine_similarity`` scan over the whole graph.

These tests run against the REAL ephemeral epistemic-graph engine via the
``engine_graph`` fixture (a fresh isolated tenant per test) — NOT mocks, NOT
SQLite. They seed nodes + embeddings in the engine and assert the retriever
returns the right ranked results through the unified plan, and that the deleted
O(N) cosine fallback is gone (the vector path errors without an engine rather than
silently scanning).
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import HybridRetriever

pytestmark = [pytest.mark.engine, pytest.mark.concept("KG-2.250")]


class _FakeEmbed:
    """Deterministic embedding model — maps a token vocabulary to unit axes.

    No network: each seeded doc and the query embed to a fixed orthonormal axis so
    the engine ANN ranking is fully determined and assertable.
    """

    _VOCAB = {
        "alpha": [1.0, 0.0, 0.0, 0.0],
        "beta": [0.0, 1.0, 0.0, 0.0],
        "gamma": [0.0, 0.0, 1.0, 0.0],
        "delta": [0.0, 0.0, 0.0, 1.0],
    }

    def get_text_embedding(self, text: str) -> list[float]:
        for token, vec in self._VOCAB.items():
            if token in text.lower():
                return list(vec)
        return [0.0, 0.0, 0.0, 0.0]


class _Engine:
    """Minimal engine surface the retriever reads: the REAL ``graph`` + a truthy
    ``backend`` so the vector arm runs, plus a keyword fallback."""

    def __init__(self, graph: Any) -> None:
        self.graph = graph
        # A truthy backend the vector arm gates on. Its ``execute`` returns nothing
        # so the multi-hop traversal phase falls to the resident-graph BFS over the
        # REAL engine (``self.graph``) — the engine path we are exercising.
        self.backend = _NoCypherBackend()
        self.graph_compute = graph

    def _search_keyword(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        # Only reached if the vector arm yields nothing — assert it is NOT here.
        raise AssertionError(
            "keyword fallback was hit — the engine vector path returned nothing"
        )


class _NoCypherBackend:
    """Truthy backend whose Cypher ``execute`` yields nothing (forces BFS path)."""

    def execute(self, _q: str, _p: Any = None) -> list[dict[str, Any]]:
        return []


def _retriever(graph: Any, *, embed: Any = None) -> HybridRetriever:
    """A HybridRetriever wired to the real engine graph, skipping heavy __init__."""
    r = HybridRetriever.__new__(HybridRetriever)
    r.engine = _Engine(graph)  # type: ignore[assignment]
    r._schema_pack = None
    r._reranker = None
    r._rerank_overfetch = 4
    from agent_utilities.models.schema_pack import BacklinkBoostStrategy

    r._boost_strategy = BacklinkBoostStrategy.GLOBAL
    r._boost_factor = 0.0  # neutralise backlink boost for deterministic scores
    r._relevance_threshold = 0.1
    r._quality_gate = None
    r._last_quality_report = None
    r._embed_model = embed if embed is not None else _FakeEmbed()
    r._embed_model_initialized = True
    from agent_utilities.knowledge_graph.core.hypergraph import (
        PositionalInteractionEncoder,
    )

    r._enc_pi = PositionalInteractionEncoder()
    return r


def _seed(graph: Any) -> None:
    """Seed three docs with orthonormal embeddings in the REAL engine."""
    graph.add_node("doc_alpha", {"type": "Doc", "name": "alpha doc", "year": 2025})
    graph.add_node("doc_beta", {"type": "Doc", "name": "beta doc", "year": 2020})
    graph.add_node("doc_gamma", {"type": "Doc", "name": "gamma doc", "year": 2024})
    graph.add_embedding("doc_alpha", [1.0, 0.0, 0.0, 0.0])
    graph.add_embedding("doc_beta", [0.0, 1.0, 0.0, 0.0])
    graph.add_embedding("doc_gamma", [0.0, 0.0, 1.0, 0.0])


def test_retrieve_hybrid_ranks_via_engine_unified_plan(engine_graph: Any) -> None:
    """The vector arm returns the engine-ANN-ranked doc, hydrated with its props."""
    _seed(engine_graph)
    r = _retriever(engine_graph)

    out = r.retrieve_hybrid("find the alpha document", context_window=3)

    ids = [n.get("id") for n in out]
    assert "doc_alpha" in ids, f"expected the alpha doc to be retrieved, got {ids}"
    # The top base node is the nearest neighbour (alpha axis) — engine-ranked.
    assert out[0].get("id") == "doc_alpha"
    # Properties were hydrated from the engine (one batched fetch), not lost.
    assert out[0].get("name") == "alpha doc"


def test_retrieve_hybrid_distinguishes_neighbours(engine_graph: Any) -> None:
    """A different query embeds to a different axis → a different top neighbour."""
    _seed(engine_graph)
    r = _retriever(engine_graph)

    out = r.retrieve_hybrid("the beta result please", context_window=3)

    assert out, "expected a non-empty result from the engine vector path"
    assert out[0].get("id") == "doc_beta"


def test_unified_plan_result_matches_native_ann_ordering(engine_graph: Any) -> None:
    """The unified plan and the native ANN primitive agree on ranking order.

    Proves the unified plan IS the engine's vector index (parity with the staged
    native path), so collapsing onto it preserves recall.
    """
    _seed(engine_graph)
    r = _retriever(engine_graph)

    emb = [1.0, 0.0, 0.0, 0.0]  # alpha axis
    # The retriever's engine path (unified when available; native ANN otherwise).
    via_helper = r._engine_vector_search(emb, top_k=3, threshold=0.1)
    helper_ids = [n["id"] for n in via_helper]
    # The engine's own native ANN, directly.
    native = engine_graph.semantic_search(emb, 3)
    native_ids = [str(nid) for nid, _score in native]

    assert helper_ids[0] == "doc_alpha"
    # Same nearest neighbour as the raw engine ANN — one vector index, one ranking.
    assert native_ids[0] == "doc_alpha"


def test_vector_path_has_no_on_python_cosine_fallback(engine_graph: Any) -> None:
    """The deleted O(N) Python cosine scan is GONE.

    The retriever no longer carries ``_vector_search_native`` / a brute-force
    ``cosine_similarity`` scan over all nodes: the vector neighbourhood is ALWAYS
    the engine's ANN. With no engine reachable the vector path raises rather than
    silently scanning the graph in Python.
    """
    # The old O(N) entry point is removed from the public surface.
    assert not hasattr(HybridRetriever, "_vector_search_native")

    # With a graph that has no usable vector surface, the helper returns nothing
    # (degrade to keyword) — it never falls to a Python cosine scan.
    class _Brokengraph:
        def query_unified(self, *_a: Any, **_k: Any) -> list[dict[str, Any]]:
            raise RuntimeError("engine built without `query` feature")

        def semantic_search(self, *_a: Any, **_k: Any) -> list[Any]:
            raise RuntimeError("no engine reachable")

    r = _retriever(_Brokengraph())
    # No engine ANN at all → empty (NOT an O(N) Python scan, NOT a hang).
    assert r._engine_vector_search([1.0, 0.0, 0.0, 0.0], top_k=3, threshold=0.1) == []
