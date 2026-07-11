#!/usr/bin/python
"""Tests for Seam 6 — ContextCompiler x KV-cache reuse.

CONCEPT:AU-KG.retrieval.context-compiler-kv-seam. The X-7 ``ContextCompiler`` (CONCEPT:AU-KG.retrieval.context-compiler)
can optionally route its assembled bundle through the existing shared,
content-addressed KV-cache surface (``agent_utilities.kvcache.EpistemicGraphKVBackend``
/ the ``graph_kvcache`` MCP tool) via ``compile(..., kv_backend=...)``. These tests
prove: (1) a stable cache key derived from the evidence-id set + policy_version +
token_budget, (2) a second compile with identical inputs reuses the cached bundle
instead of re-running MMR/scoring/proof-graph assembly, (3) a different evidence
set or policy version never false-reuses, and (4) citations/calibration scores
survive the serialize/deserialize round trip.

Uses a small in-memory ``FakeKVBackend`` implementing the exact duck-typed
``get(key) -> bytes | None`` / ``put(key, bytes) -> bool`` shape
``EpistemicGraphKVBackend`` exposes — no network, fully offline/deterministic.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.knowledge_graph.ontology.permissioning import clear_markings
from agent_utilities.knowledge_graph.retrieval.context_compiler import (
    ContextCompiler,
    compute_bundle_cache_key,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext


@pytest.fixture(autouse=True)
def _clean_state():
    reset_company_brain()
    clear_markings()
    yield
    reset_company_brain()
    clear_markings()


class FakeRetriever:
    """Duck-typed stand-in for ``HybridRetriever``/the engine's ``search_hybrid``.

    Returns a fixed candidate pool regardless of query — the seam under test is
    the cache key + reuse, not retrieval itself.
    """

    def __init__(self, nodes: list[dict]) -> None:
        self._nodes = nodes

    def retrieve_hybrid(self, query, context_window=10, **kwargs):
        return list(self._nodes)[:context_window]


class FakeKVBackend:
    """In-memory stand-in for ``EpistemicGraphKVBackend``'s ``get``/``put`` shape.

    Deliberately minimal — just enough to prove the compiler talks to the
    EXISTING KV-cache interface (get/put by opaque key, raw bytes) rather than a
    second cache being invented for this seam.
    """

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.get_calls = 0
        self.put_calls = 0

    def get(self, key: str) -> bytes | None:
        self.get_calls += 1
        return self.store.get(key)

    def put(self, key: str, value: bytes) -> bool:
        self.put_calls += 1
        self.store[key] = value
        return True


def _actor(**kw) -> ActorContext:
    return ActorContext(actor_id="agent:test", actor_type=ActorType.AI_AGENT, **kw)


def _session(*, policy_version: str = "v1", **kw) -> GraphSession:
    return GraphSession(actor=_actor(**kw), policy_version=policy_version)


_NODES = [
    {
        "id": "claim:a",
        "type": "Claim",
        "name": "Claim A",
        "description": "The premise claim, well sourced.",
        "score": 0.9,
        "confidence": 0.9,
        "source_refs": ["doc:1"],
    },
    {
        "id": "claim:b",
        "type": "Claim",
        "name": "Claim B",
        "description": "The dependent claim, worded very differently indeed.",
        "score": 0.85,
        "confidence": 0.7,
        "proof_ids": ["claim:a"],
        "contradiction_ids": ["claim:c"],
    },
]


# --------------------------------------------------------------------------
# Cache key: stable, order-independent, sensitive to every named axis
# --------------------------------------------------------------------------


def test_cache_key_is_order_independent_and_stable():
    key_ab = compute_bundle_cache_key(["a", "b"], policy_version="v1", token_budget=100)
    key_ba = compute_bundle_cache_key(["b", "a"], policy_version="v1", token_budget=100)
    assert key_ab == key_ba

    key_again = compute_bundle_cache_key(
        ["a", "b"], policy_version="v1", token_budget=100
    )
    assert key_ab == key_again


def test_cache_key_differs_on_evidence_set_policy_version_and_budget():
    base = compute_bundle_cache_key(["a", "b"], policy_version="v1", token_budget=100)
    diff_evidence = compute_bundle_cache_key(
        ["a", "c"], policy_version="v1", token_budget=100
    )
    diff_policy = compute_bundle_cache_key(
        ["a", "b"], policy_version="v2", token_budget=100
    )
    diff_budget = compute_bundle_cache_key(
        ["a", "b"], policy_version="v1", token_budget=200
    )
    assert len({base, diff_evidence, diff_policy, diff_budget}) == 4


# --------------------------------------------------------------------------
# Reuse: second compile of the same inputs hits the KV-cache path
# --------------------------------------------------------------------------


def test_second_compile_reuses_bundle_via_kv_layer(monkeypatch):
    compiler = ContextCompiler(FakeRetriever(_NODES))
    kv = FakeKVBackend()
    session = _session()

    # Spy on the MMR similarity step — the thing a cache HIT must skip — to
    # prove the second compile does NOT redo the assembly, not merely that a
    # flag says so.
    calls = {"n": 0}
    original = ContextCompiler._max_similarity

    def _spy(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ContextCompiler, "_max_similarity", _spy)

    bundle1 = compiler.compile(
        "test query", session=session, top_k=2, candidate_pool=2, kv_backend=kv
    )
    assert bundle1.kv_cache_hit is False
    assert bundle1.cache_key
    first_call_count = calls["n"]
    assert first_call_count > 0, "first compile should have run MMR scoring"

    bundle2 = compiler.compile(
        "test query", session=session, top_k=2, candidate_pool=2, kv_backend=kv
    )
    assert bundle2.kv_cache_hit is True
    assert bundle2.cache_key == bundle1.cache_key
    assert calls["n"] == first_call_count, (
        "second compile hit the KV-cache path and must NOT re-run MMR assembly"
    )

    # The store happened exactly once (on the miss); both compiles checked the
    # cache (a real interaction with the KV layer, not a bypass).
    assert kv.put_calls == 1
    assert kv.get_calls == 2

    # Citations + calibration scores survive the serialize/deserialize round trip.
    assert [c.node_id for c in bundle1.citations] == [
        c.node_id for c in bundle2.citations
    ]
    assert [c.source_refs for c in bundle1.citations] == [
        c.source_refs for c in bundle2.citations
    ]
    for it1, it2 in zip(bundle1.items, bundle2.items, strict=True):
        assert it1.id == it2.id
        assert it1.composite_score == pytest.approx(it2.composite_score)
        assert it1.evidence_quality == pytest.approx(it2.evidence_quality)
        assert it1.freshness == pytest.approx(it2.freshness)
        assert it1.citation.confidence == pytest.approx(it2.citation.confidence)
    relations1 = {(e.src, e.dst, e.relation) for e in bundle1.proof_graph}
    relations2 = {(e.src, e.dst, e.relation) for e in bundle2.proof_graph}
    assert relations1 == relations2
    assert ("claim:a", "claim:b", "supports") in relations2


def test_no_kv_backend_leaves_default_behavior_unchanged():
    compiler = ContextCompiler(FakeRetriever(_NODES))
    bundle = compiler.compile(
        "test query", session=_session(), top_k=2, candidate_pool=2
    )
    assert bundle.cache_key == ""
    assert bundle.kv_cache_hit is False


# --------------------------------------------------------------------------
# No false reuse: different evidence set / policy version never collide
# --------------------------------------------------------------------------


def test_different_evidence_set_does_not_false_reuse():
    kv = FakeKVBackend()
    session = _session()

    other_nodes = [
        {
            "id": "claim:x",
            "type": "Claim",
            "name": "Claim X",
            "description": "A totally unrelated claim about a different topic.",
            "score": 0.9,
            "confidence": 0.9,
        },
        {
            "id": "claim:y",
            "type": "Claim",
            "name": "Claim Y",
            "description": "Another unrelated claim, distinct wording throughout.",
            "score": 0.85,
            "confidence": 0.7,
        },
    ]

    bundle_a = ContextCompiler(FakeRetriever(_NODES)).compile(
        "test query", session=session, top_k=2, candidate_pool=2, kv_backend=kv
    )
    bundle_b = ContextCompiler(FakeRetriever(other_nodes)).compile(
        "test query", session=session, top_k=2, candidate_pool=2, kv_backend=kv
    )

    assert bundle_a.cache_key != bundle_b.cache_key
    assert bundle_a.kv_cache_hit is False
    assert bundle_b.kv_cache_hit is False
    assert {it.id for it in bundle_b.items} == {"claim:x", "claim:y"}
    assert kv.put_calls == 2


def test_different_policy_version_does_not_false_reuse():
    kv = FakeKVBackend()
    compiler = ContextCompiler(FakeRetriever(_NODES))

    bundle_v1 = compiler.compile(
        "test query",
        session=_session(policy_version="v1"),
        top_k=2,
        candidate_pool=2,
        kv_backend=kv,
    )
    bundle_v2 = compiler.compile(
        "test query",
        session=_session(policy_version="v2"),
        top_k=2,
        candidate_pool=2,
        kv_backend=kv,
    )

    assert bundle_v1.cache_key != bundle_v2.cache_key
    assert bundle_v1.kv_cache_hit is False
    assert bundle_v2.kv_cache_hit is False
    assert kv.put_calls == 2

    # Same policy version again *does* reuse — isolates policy_version as the
    # only thing that changed above.
    bundle_v1_again = compiler.compile(
        "test query",
        session=_session(policy_version="v1"),
        top_k=2,
        candidate_pool=2,
        kv_backend=kv,
    )
    assert bundle_v1_again.cache_key == bundle_v1.cache_key
    assert bundle_v1_again.kv_cache_hit is True
