#!/usr/bin/python
"""Tests for ContextCompiler — policy-aware LLM context assembly (CONCEPT:AU-KG.retrieval.context-compiler).

Deterministic + offline: a fake retriever hands back canned node dicts (no
embedding/model calls); policy enforcement reuses the real, live
``permissioning`` gate (CONCEPT:AU-KG.ontology.redact-object-materialize-restricted) rather than a mock.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.knowledge_graph.ontology.permissioning import (
    Marking,
    apply_marking,
    clear_markings,
)
from agent_utilities.knowledge_graph.retrieval.context_compiler import (
    ContextCompiler,
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

    Returns a fixed candidate pool regardless of query — the compiler's
    scoring/selection is what's under test, not retrieval itself.
    """

    def __init__(self, nodes: list[dict]) -> None:
        self._nodes = nodes

    def retrieve_hybrid(self, query, context_window=10, **kwargs):
        return list(self._nodes)[:context_window]


def _actor(**kw) -> ActorContext:
    return ActorContext(actor_id="agent:test", actor_type=ActorType.AI_AGENT, **kw)


def _session(**kw) -> GraphSession:
    return GraphSession(actor=_actor(**kw))


# --------------------------------------------------------------------------
# Token budget
# --------------------------------------------------------------------------


def test_compile_fits_within_token_budget():
    long_text = "word " * 400  # ~530 estimated tokens each
    nodes = [
        {"id": f"n{i}", "type": "Doc", "name": f"Doc {i}", "description": long_text, "score": 0.9 - i * 0.01}
        for i in range(10)
    ]
    compiler = ContextCompiler(FakeRetriever(nodes))
    bundle = compiler.compile(
        "test query", session=_session(), top_k=10, candidate_pool=10, token_budget=1000
    )
    assert bundle.tokens_used <= bundle.token_budget
    assert len(bundle.items) < len(nodes)  # budget forced some drops
    assert bundle.dropped_budget > 0


# --------------------------------------------------------------------------
# Evidence quality + freshness preference
# --------------------------------------------------------------------------


def test_higher_evidence_quality_is_preferred():
    # Same relevance score; one is well-evidenced, one is a bare, unscored hit.
    nodes = [
        {
            "id": "strong",
            "type": "Claim",
            "name": "Strong claim",
            "description": "A well-sourced claim about X.",
            "score": 0.8,
            "confidence": 0.95,
            "source_refs": ["doc:1", "doc:2"],
            "evidence_refs": ["span:1"],
        },
        {
            "id": "weak",
            "type": "Claim",
            "name": "Weak claim",
            "description": "An unsourced claim about Y, unrelated wording entirely.",
            "score": 0.8,
        },
    ]
    compiler = ContextCompiler(FakeRetriever(nodes))
    bundle = compiler.compile("test query", session=_session(), top_k=2, candidate_pool=2)
    by_id = {it.id: it for it in bundle.items}
    assert by_id["strong"].evidence_quality > by_id["weak"].evidence_quality
    assert by_id["strong"].composite_score > by_id["weak"].composite_score


def test_contested_claim_scored_lower():
    nodes = [
        {
            "id": "clean",
            "type": "Claim",
            "name": "Clean claim",
            "description": "An uncontested claim, entirely distinct phrasing here.",
            "score": 0.8,
            "confidence": 0.8,
        },
        {
            "id": "contested",
            "type": "Claim",
            "name": "Contested claim",
            "description": "A disputed claim with totally different wording too.",
            "score": 0.8,
            "confidence": 0.8,
            "contradiction_ids": ["other:1"],
        },
    ]
    compiler = ContextCompiler(FakeRetriever(nodes))
    bundle = compiler.compile("test query", session=_session(), top_k=2, candidate_pool=2)
    by_id = {it.id: it for it in bundle.items}
    assert by_id["clean"].evidence_quality > by_id["contested"].evidence_quality


def test_fresher_claim_is_preferred():
    nodes = [
        {
            "id": "fresh",
            "type": "Doc",
            "name": "Fresh doc",
            "description": "Recent information, distinct content entirely.",
            "score": 0.8,
            "timestamp": "2026-07-10T00:00:00Z",
        },
        {
            "id": "stale",
            "type": "Doc",
            "name": "Stale doc",
            "description": "Old information, totally different words used.",
            "score": 0.8,
            "timestamp": "2020-01-01T00:00:00Z",
        },
    ]
    compiler = ContextCompiler(FakeRetriever(nodes))
    bundle = compiler.compile(
        "test query",
        session=_session(),
        top_k=2,
        candidate_pool=2,
        as_of="2026-07-10T00:00:00Z",
    )
    by_id = {it.id: it for it in bundle.items}
    assert by_id["fresh"].freshness > by_id["stale"].freshness
    assert by_id["fresh"].composite_score > by_id["stale"].composite_score


# --------------------------------------------------------------------------
# Policy — drop/redact what the actor can't see
# --------------------------------------------------------------------------


def test_policy_restricted_item_dropped_for_unauthorized_actor():
    apply_marking("restricted:1", Marking("pii", requires_audit=True))
    nodes = [
        {"id": "public:1", "type": "Doc", "name": "Public", "description": "Open info.", "score": 0.9},
        {"id": "restricted:1", "type": "Doc", "name": "Secret", "description": "Sensitive info.", "score": 0.95},
    ]
    compiler = ContextCompiler(FakeRetriever(nodes))
    # A low-clearance actor holding no markings.
    bundle = compiler.compile(
        "test query",
        session=_session(roles=("analyst",)),
        top_k=5,
        candidate_pool=5,
    )
    ids = {it.id for it in bundle.items}
    assert "restricted:1" not in ids
    assert "public:1" in ids
    assert bundle.dropped_policy == 1


def test_policy_cleared_actor_sees_restricted_item():
    apply_marking("restricted:1", Marking("pii", requires_audit=True))
    nodes = [
        {"id": "public:1", "type": "Doc", "name": "Public", "description": "Open info.", "score": 0.9},
        {"id": "restricted:1", "type": "Doc", "name": "Secret", "description": "Sensitive info.", "score": 0.95},
    ]
    compiler = ContextCompiler(FakeRetriever(nodes))
    bundle = compiler.compile(
        "test query",
        session=_session(roles=("marking:pii",)),
        top_k=5,
        candidate_pool=5,
    )
    ids = {it.id for it in bundle.items}
    assert "restricted:1" in ids
    assert bundle.dropped_policy == 0


# --------------------------------------------------------------------------
# Citations + proof graph
# --------------------------------------------------------------------------


def test_bundle_returns_citations_and_proof_graph():
    nodes = [
        {
            "id": "claim:a",
            "type": "Claim",
            "name": "Claim A",
            "description": "The premise claim.",
            "score": 0.9,
            "confidence": 0.9,
            "source_refs": ["doc:1"],
        },
        {
            "id": "claim:b",
            "type": "Claim",
            "name": "Claim B",
            "description": "The dependent claim, worded very differently.",
            "score": 0.85,
            "confidence": 0.7,
            "proof_ids": ["claim:a"],
            "contradiction_ids": ["claim:c"],
        },
    ]
    compiler = ContextCompiler(FakeRetriever(nodes))
    bundle = compiler.compile("test query", session=_session(), top_k=2, candidate_pool=2)

    assert len(bundle.citations) == len(bundle.items)
    assert any(c.source_refs for c in bundle.citations)

    relations = {(e.src, e.dst, e.relation) for e in bundle.proof_graph}
    assert ("claim:a", "claim:b", "supports") in relations
    assert ("claim:b", "claim:c", "contradicts") in relations

    text = bundle.as_text()
    assert "Citations" in text
    assert "Proof graph" in text


# --------------------------------------------------------------------------
# MMR diversity
# --------------------------------------------------------------------------


def test_mmr_reduces_redundancy():
    # Two near-duplicate high-scoring hits plus one diverse, slightly lower one.
    nodes = [
        {
            "id": "dup1",
            "type": "Doc",
            "name": "Dup 1",
            "description": "the quick brown fox jumps over the lazy dog today",
            "score": 0.95,
        },
        {
            "id": "dup2",
            "type": "Doc",
            "name": "Dup 2",
            "description": "the quick brown fox jumps over the lazy dog now",
            "score": 0.94,
        },
        {
            "id": "diverse",
            "type": "Doc",
            "name": "Diverse",
            "description": "completely unrelated topic about deep sea marine biology",
            "score": 0.80,
        },
    ]
    compiler = ContextCompiler(FakeRetriever(nodes))

    # Pure relevance (lambda=1.0): both near-duplicates always win over diverse.
    bundle_relevance_only = compiler.compile(
        "test query", session=_session(), top_k=2, candidate_pool=3, diversity_lambda=1.0
    )
    ids_relevance_only = {it.id for it in bundle_relevance_only.items}
    assert ids_relevance_only == {"dup1", "dup2"}

    # MMR (lambda=0.5): the second near-duplicate is redundant against the
    # first, so the diverse item displaces it despite the lower raw score.
    bundle_mmr = compiler.compile(
        "test query", session=_session(), top_k=2, candidate_pool=3, diversity_lambda=0.5
    )
    ids_mmr = {it.id for it in bundle_mmr.items}
    assert "diverse" in ids_mmr
    assert ids_mmr != ids_relevance_only

    # The kept near-duplicate carries a nonzero diversity penalty entry (dup2
    # excluded before it, dup1 selected first with 0 penalty since nothing is
    # selected yet).
    dup1_item = next(it for it in bundle_mmr.items if it.id == "dup1")
    assert dup1_item.diversity_penalty == 0.0


# --------------------------------------------------------------------------
# Observability / determinism
# --------------------------------------------------------------------------


def test_decisions_log_is_observable_and_deterministic():
    nodes = [
        {"id": "a", "type": "Doc", "name": "A", "description": "alpha content", "score": 0.9},
        {"id": "b", "type": "Doc", "name": "B", "description": "beta content", "score": 0.5},
    ]
    compiler = ContextCompiler(FakeRetriever(nodes))
    bundle1 = compiler.compile("q", session=_session(), top_k=1, candidate_pool=2)
    bundle2 = compiler.compile("q", session=_session(), top_k=1, candidate_pool=2)

    assert bundle1.decisions == bundle2.decisions
    assert [it.id for it in bundle1.items] == [it.id for it in bundle2.items]
    assert any(d.get("stage") == "select" for d in bundle1.decisions)
    included = [d for d in bundle1.decisions if d.get("included")]
    assert included and "composite_score" in included[0]
