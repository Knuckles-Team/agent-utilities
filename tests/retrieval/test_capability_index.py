#!/usr/bin/python
"""Tests for the capability-aware designation index and facade delegation.

All tests use synthetic embedding vectors — no embedding model, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent_utilities.knowledge_graph.facade import KnowledgeGraph
from agent_utilities.knowledge_graph.retrieval.capability_index import (
    CapabilityIndex,
    Designation,
)

DIM = 16


def _unit(seed: int, dim: int = DIM) -> list[float]:
    """Deterministic pseudo-random unit-ish vector for a given seed."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v.tolist()


def _basis(i: int, dim: int = DIM, scale: float = 1.0) -> list[float]:
    """One-hot basis vector along axis ``i`` (so similarities are controllable)."""
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = scale
    return v.tolist()


def _populated_index(prefer_backend: str | None = None) -> CapabilityIndex:
    idx = CapabilityIndex(dim=DIM, prefer_backend=prefer_backend)
    # web_search and web_fetch both provide "web"; only web_search provides "search".
    idx.add("web_search", _basis(0), ["web", "search"], swappable_with=["serp_api"])
    idx.add("serp_api", _basis(0, scale=0.9), ["web", "search"])
    idx.add("web_fetch", _basis(1), ["web", "fetch"])
    idx.add("calculator", _basis(2), ["math"])
    idx.add("python_exec", _basis(3), ["code", "math"])
    return idx


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
def test_backend_is_reported():
    idx = CapabilityIndex(dim=DIM)
    assert idx.backend in {"hnsw", "numpy"}


# ---------------------------------------------------------------------------
# 1. Capability filtering reduces the candidate set
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prefer", ["numpy", "hnsw"])
def test_capability_filter_shrinks_results(prefer):
    idx = _populated_index(prefer)

    # No filter: query near axis-0 should be able to surface many entities.
    unfiltered = idx.designate(_basis(0), required_caps=None, k=10)
    unfiltered_ids = {d.id for d in unfiltered}

    # Require both "web" and "search": only web_search + serp_api qualify.
    filtered = idx.designate(_basis(0), required_caps=["web", "search"], k=10)
    filtered_ids = {d.id for d in filtered}

    assert filtered_ids == {"web_search", "serp_api"}
    assert filtered_ids < unfiltered_ids
    assert len(filtered_ids) < len(unfiltered_ids)
    # Provenance reflects the filtering.
    assert all(d.provenance["capability_filtered"] for d in filtered)
    assert all(d.provenance["required_caps"] == ["search", "web"] for d in filtered)


def test_required_caps_with_no_provider_returns_empty():
    idx = _populated_index("numpy")
    assert idx.designate(_basis(0), required_caps=["nonexistent"], k=5) == []
    # Conjunction that no single entity satisfies.
    assert idx.designate(_basis(0), required_caps=["math", "web"], k=5) == []


# ---------------------------------------------------------------------------
# 2. ANN ranking returns the planted most-similar id at rank 1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prefer", ["numpy", "hnsw"])
def test_ranking_returns_planted_top1(prefer):
    idx = CapabilityIndex(dim=DIM, prefer_backend=prefer)
    # Plant a clear winner along axis 5.
    idx.add("winner", _basis(5), ["web"])
    idx.add("near", _basis(5), ["web"])  # identical direction, added second
    idx.add("other_a", _basis(0), ["web"])
    idx.add("other_b", _basis(1), ["web"])
    idx.add("other_c", _basis(2), ["web"])

    # Query exactly along axis 5 — winner/near are the most similar.
    results = idx.designate(_basis(5), required_caps=None, k=3)
    assert results, "expected designations"
    top_ids = {results[0].id, results[1].id} if len(results) > 1 else {results[0].id}
    assert "winner" in {d.id for d in results[:2]} or "near" in {
        d.id for d in results[:2]
    }
    assert results[0].id in {"winner", "near"}
    assert results[0].score == pytest.approx(1.0, abs=1e-3)
    assert "other_c" not in top_ids


# ---------------------------------------------------------------------------
# 3. alternatives() returns a swappableWith partner
# ---------------------------------------------------------------------------
def test_alternatives_returns_swappable_partner():
    idx = _populated_index("numpy")
    assert idx.alternatives("web_search") == ["serp_api"]
    # Symmetric edge.
    assert idx.alternatives("serp_api") == ["web_search"]
    # Designation provenance surfaces alternatives.
    res = idx.designate(_basis(0), required_caps=["web", "search"], k=5)
    web_search = next(d for d in res if d.id == "web_search")
    assert web_search.provenance.get("alternatives") == ["serp_api"]


def test_build_from_edges_dicts():
    idx = CapabilityIndex(dim=DIM, prefer_backend="numpy")
    idx.build_from_edges(
        [
            {
                "id": "a",
                "embedding": _basis(0),
                "capabilities": ["x"],
                "swappable_with": ["b"],
            },
            {"id": "b", "embedding": _basis(1), "providesCapability": ["x"]},
        ]
    )
    assert len(idx) == 2
    assert idx.alternatives("a") == ["b"]
    assert {d.id for d in idx.designate(_basis(0), required_caps=["x"], k=5)} == {
        "a",
        "b",
    }


# ---------------------------------------------------------------------------
# 4. save()/load() round-trips and returns identical top-k
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prefer", ["numpy", "hnsw"])
def test_save_load_roundtrip_identical_topk(prefer, tmp_path):
    idx = _populated_index(prefer)
    query = _unit(seed=42)

    before = idx.designate(query, required_caps=None, k=5)
    before_ids = [d.id for d in before]

    save_dir = tmp_path / "capidx"
    idx.save(save_dir)
    reloaded = CapabilityIndex.load(save_dir)

    assert reloaded.backend == idx.backend
    assert reloaded.dim == idx.dim
    assert len(reloaded) == len(idx)

    after = reloaded.designate(query, required_caps=None, k=5)
    after_ids = [d.id for d in after]
    assert after_ids == before_ids

    # Filtered query round-trips too.
    fb = [d.id for d in idx.designate(query, required_caps=["web"], k=5)]
    fa = [d.id for d in reloaded.designate(query, required_caps=["web"], k=5)]
    assert fa == fb
    # Swappable + capability maps survive.
    assert reloaded.alternatives("web_search") == ["serp_api"]


# ---------------------------------------------------------------------------
# 5. Facade designate() delegates correctly
# ---------------------------------------------------------------------------
def test_facade_designate_delegates():
    idx = _populated_index("numpy")
    kg = KnowledgeGraph(retrieval=idx)

    # Same index object is exposed.
    assert kg.retrieval is idx

    via_facade = kg.designate(_basis(0), required_caps=["web", "search"], k=5)
    via_index = idx.designate(_basis(0), required_caps=["web", "search"], k=5)

    assert [d.id for d in via_facade] == [d.id for d in via_index]
    assert all(isinstance(d, Designation) for d in via_facade)
    assert {d.id for d in via_facade} == {"web_search", "serp_api"}


def test_facade_lazy_retrieval_when_none_provided():
    kg = KnowledgeGraph(embedding_dim=DIM)
    # No retrieval supplied — facade builds an empty usable index lazily.
    assert isinstance(kg.retrieval, CapabilityIndex)
    assert kg.designate(_basis(0), required_caps=None, k=5) == []
    kg.retrieval.add("t", _basis(0), ["web"])
    out = kg.designate(_basis(0), required_caps=["web"], k=5)
    assert [d.id for d in out] == ["t"]


def test_facade_construction_is_side_effect_free():
    # Constructing the facade must not require any running service.
    kg = KnowledgeGraph(retrieval=_populated_index("numpy"))
    assert kg is not None
    # retrieval works without ever touching store/compute/semantic.
    assert kg.designate(_basis(0), required_caps=["web"], k=10)
