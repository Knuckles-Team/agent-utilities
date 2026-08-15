"""Tests for KnowledgeGraph.populate_capability_index (Plan 08 Synergy 1).

This is the bridge that feeds real graph nodes into the L2 capability index so
the live router can call ``facade.designate(...)``. All vectors are synthetic —
no embedding model, no network, no running store.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.facade import KnowledgeGraph
from agent_utilities.models.company_brain import DataClassification, NodeACL
# The compiled epistemic_graph.numeric kernel must be built for these tests; skip the whole module cleanly when it isn't, rather than erroring out collection (CONCEPT:AU-KG.compute.numeric-kernel).
pytest.importorskip("epistemic_graph.numeric")

from agent_utilities.numeric import xp as np

DIM = 16


@pytest.fixture(autouse=True)
def _clean_company_brain():
    reset_company_brain()
    yield
    reset_company_brain()


def _grant_public(*node_ids: str) -> None:
    """Grant a PUBLIC-classification ACL for each id.

    ``KnowledgeGraph.designate`` runs its results through the real,
    fail-closed ACL gate (AU-P0-4 — a node with no ACL is denied outright);
    these synthetic capability-index entries never exist in a real graph.
    """
    permissions = get_company_brain().permissions
    for node_id in node_ids:
        permissions.set_acl(
            NodeACL(node_id=node_id, classification=DataClassification.PUBLIC)
        )


def _basis(i: int, dim: int = DIM, scale: float = 1.0) -> list[float]:
    """One-hot basis vector along axis ``i`` (controllable similarities)."""
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = scale
    return v.tolist()


def _nodes() -> list[dict]:
    return [
        # Planted relevant node: provides "web"+"search", sits on axis 0.
        {
            "id": "web_search",
            "embedding": _basis(0),
            "capabilities": ["web", "search"],
            "swappable_with": ["serp_api"],
        },
        # Distractor in the SAME embedding neighbourhood but without "search".
        {
            "id": "web_fetch",
            "embedding": _basis(0, scale=0.95),
            "capabilities": ["web", "fetch"],
        },
        {
            "id": "calculator",
            "embedding": _basis(2),
            "capabilities": ["math"],
        },
        # No embedding -> must be skipped.
        {"id": "no_embedding", "capabilities": ["web", "search"]},
        # Empty embedding -> must be skipped.
        {"id": "empty_embedding", "embedding": [], "capabilities": ["web"]},
        # Missing capabilities key -> tolerated (added with no caps).
        {"id": "bare", "embedding": _basis(5)},
    ]


def test_populate_returns_count_skipping_missing_embeddings():
    kg = KnowledgeGraph(embedding_dim=DIM)
    added = kg.populate_capability_index(_nodes())
    # 6 nodes in, 2 skipped (no embedding + empty embedding) -> 4 added.
    assert added == 4
    assert len(kg.retrieval) == 4
    # Skipped ids are genuinely absent from the index.
    assert "no_embedding" not in kg.retrieval._id_to_vec
    assert "empty_embedding" not in kg.retrieval._id_to_vec


def test_designate_after_population_returns_planted_id():
    kg = KnowledgeGraph(embedding_dim=DIM)
    kg.populate_capability_index(_nodes())
    _grant_public("web_search", "web_fetch", "calculator", "bare")

    # Query the web_search neighbourhood, requiring "search" which only
    # web_search provides -> the planted relevant id must be designated.
    results = kg.designate(_basis(0), required_caps=["search"], k=3)
    ids = [d.id for d in results]
    assert ids, "expected at least one designation"
    assert ids[0] == "web_search"
    # The distractor sharing the embedding space but lacking "search" is gone.
    assert "web_fetch" not in ids


def test_swappable_edges_preserved_through_population():
    kg = KnowledgeGraph(embedding_dim=DIM)
    kg.populate_capability_index(_nodes())
    # swappable_with edge from the node dict should be wired into the index.
    assert "serp_api" in kg.retrieval.alternatives("web_search")


def test_empty_population_is_noop():
    kg = KnowledgeGraph(embedding_dim=DIM)
    assert kg.populate_capability_index([]) == 0
    assert len(kg.retrieval) == 0
