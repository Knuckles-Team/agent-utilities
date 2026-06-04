"""Plan 08 Synergy 1: KG-driven specialist designation wired into the router.

Verifies the enricher builds an ANN capability index from the engine's callable
nodes and designates by query embedding, and that it degrades gracefully (returns
None → router falls back) when embeddings/model are unavailable.
"""

from __future__ import annotations

import types

from agent_utilities.graph.routing.enrichers.capability_designation import (
    build_designation_index,
    designate_specialists,
)


def _make_engine(nodes: dict[str, dict]):
    """Fake engine exposing graph.node_ids() + graph._get_node_properties()."""
    graph = types.SimpleNamespace(
        node_ids=lambda: list(nodes.keys()),
        _get_node_properties=lambda nid: nodes.get(nid, {}),
    )
    return types.SimpleNamespace(graph=graph, backend=None)


# Two callable tools, near-orthogonal embeddings; one non-callable node ignored.
NODES = {
    "tool:search": {
        "type": "tool",
        "embedding": [1.0, 0.0, 0.0],
        "capabilities": ["web_search"],
    },
    "tool:math": {
        "type": "tool",
        "embedding": [0.0, 1.0, 0.0],
        "capabilities": ["arithmetic"],
    },
    "concept:foo": {"type": "concept", "embedding": [0.0, 0.0, 1.0]},  # not callable
}


def test_index_built_only_from_callable_nodes_with_embeddings():
    engine = _make_engine(NODES)
    index = build_designation_index(engine)
    assert index is not None
    assert len(index) == 2  # the concept node is excluded


def test_designate_returns_best_specialist():
    engine = _make_engine(NODES)
    # Query embedding closest to tool:search.
    out = designate_specialists(
        engine, "find me a search", k=1, embed_fn=lambda q: [0.95, 0.05, 0.0]
    )
    assert out == ["tool:search"]


def test_capability_filter_restricts_candidates():
    engine = _make_engine(NODES)
    out = designate_specialists(
        engine,
        "anything",
        k=5,
        required_caps=["arithmetic"],
        embed_fn=lambda q: [0.1, 0.9, 0.0],
    )
    assert out == ["tool:math"]


def test_graceful_fallback_when_no_embeddings():
    engine = _make_engine(
        {"tool:x": {"type": "tool", "capabilities": ["c"]}}  # no embedding
    )
    assert designate_specialists(engine, "q", embed_fn=lambda q: [1.0]) is None


def test_graceful_fallback_when_no_model_and_no_embed_fn():
    engine = _make_engine(NODES)
    # No embed_fn and create_embedding_model unavailable in-test -> None, not raise.
    out = designate_specialists(engine, "q", embed_fn=lambda q: None)
    assert out is None


def test_index_cached_on_engine():
    engine = _make_engine(NODES)
    designate_specialists(engine, "q", embed_fn=lambda q: [1.0, 0.0, 0.0])
    assert getattr(engine, "_designation_index", None) is not None
