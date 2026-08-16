"""Unit tests for the unified EvidenceBundle envelope (Epistemic Substrate Program, C1).

Exercises each ``from_*`` classmethod against realistic fixture inputs shaped exactly
like the wrapped surfaces (``CodeContextAnswer``, ``RagResult``, the ``nl_to_query``/
``nl_planner.nl_query`` payload dict), asserting: field mapping is correct, nothing
from the source payload is silently dropped, and ``confidence`` is NEVER fabricated
(stays ``None`` unless a caller explicitly threads one in via an override).
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.retrieval.code_context import CodeContextAnswer
from agent_utilities.knowledge_graph.retrieval.executable_rag import (
    RagResult,
    StepOp,
    StepTrace,
)
from agent_utilities.models.evidence_bundle import EvidenceBundle

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------


def test_defaults_are_safe_and_empty():
    b = EvidenceBundle()
    assert b.answer_candidate == ""
    assert b.claims == []
    assert b.evidence_spans == []
    assert b.source_authority == {}
    assert b.contradictions == []
    assert b.confidence is None
    assert b.freshness == {}
    assert b.policy_exclusions == []
    assert b.reasoning_trace == []
    assert b.next_actions == []


# ---------------------------------------------------------------------------
# from_code_context_answer
# ---------------------------------------------------------------------------


def _code_context_answer(**overrides) -> CodeContextAnswer:
    defaults: dict = dict(
        query="how does run_agent work",
        intent="how",
        answer="`run_agent` (function) is defined at a.py:10. It calls `create_model`.",
        citations=[
            {
                "id": "code:a.py::run_agent",
                "symbol": "run_agent",
                "file": "a.py",
                "line": 10,
                "kind": "function",
                "language": "python",
                "source_system": "agent-utilities",
            }
        ],
        sections={"calls": [{"symbol": "create_model", "file": "b.py", "line": 5}]},
        anchors=[{"symbol": "run_agent", "file": "a.py", "line": 10}],
        capability_id="code_context:how:run_agent",
        used_primitives=["call_graph"],
        cross_repo=False,
        coverage={"anchors": 1, "citations": 1, "sections": {"calls": 1}},
    )
    defaults.update(overrides)
    return CodeContextAnswer(**defaults)


def test_from_code_context_answer_maps_fields():
    ans = _code_context_answer()
    b = EvidenceBundle.from_code_context_answer(ans)

    assert b.answer_candidate == ans.answer
    assert b.evidence_spans == ans.citations
    assert b.freshness == ans.coverage
    assert b.confidence is None
    # used_primitives surfaced as reasoning_trace entries
    assert {"primitive": "call_graph"} in b.reasoning_trace
    # nothing with no dedicated slot (capability_id/intent/cross_repo/query/sections)
    # is silently dropped — round-trip through the full bundle dump.
    dumped = json.dumps(b.model_dump(), default=str)
    assert ans.capability_id in dumped
    assert ans.intent in dumped
    assert ans.query in dumped
    assert "create_model" in dumped  # from `sections`
    # claims derived from sentence-splitting the answer
    assert b.claims
    assert all("id" in c and "text" in c for c in b.claims)


def test_from_code_context_answer_accepts_dict():
    ans = _code_context_answer()
    b_from_obj = EvidenceBundle.from_code_context_answer(ans)
    b_from_dict = EvidenceBundle.from_code_context_answer(ans.as_dict())
    assert b_from_obj.answer_candidate == b_from_dict.answer_candidate
    assert b_from_obj.evidence_spans == b_from_dict.evidence_spans


def test_from_code_context_answer_no_anchors_suggests_resync():
    ans = _code_context_answer(
        answer="No resolved code symbol matched 'foo'.",
        citations=[],
        sections={},
        anchors=[],
        coverage={"anchors": 0},
    )
    b = EvidenceBundle.from_code_context_answer(ans)
    assert b.next_actions  # a concrete, grounded follow-up is suggested
    assert any("source_sync" in a for a in b.next_actions)
    assert b.confidence is None


def test_bug004_from_code_context_answer_degraded_status_becomes_real_error():
    """BUG-004: `status: "degraded"` must survive the wrap as a real
    `EvidenceBundle.error`, not a silently-empty-but-`error=None` bundle — the
    exact shape that made an engine outage indistinguishable from a genuine
    empty result at the caller-facing envelope."""
    ans = _code_context_answer(
        answer=(
            "The knowledge graph engine was unavailable while answering "
            "'foo'. This is NOT evidence the symbol is unindexed — retry "
            "shortly rather than re-ingesting."
        ),
        citations=[],
        sections={},
        anchors=[],
        coverage={"anchors": 0},
        status="degraded",
        error={"code": "engine_degraded", "cause_type": "EngineCircuitOpenError"},
    )
    b = EvidenceBundle.from_code_context_answer(ans)
    assert b.error == {"code": "engine_degraded", "cause_type": "EngineCircuitOpenError"}
    # The misleading re-ingest suggestion must NOT appear for a degraded read.
    assert not any("source_sync" in a for a in b.next_actions)
    assert any("retry" in a.lower() for a in b.next_actions)


def test_from_code_context_answer_source_authority_only_when_known():
    ans = _code_context_answer()
    b = EvidenceBundle.from_code_context_answer(ans)
    assert b.source_authority == {
        "strategy": "source_authority_wins",
        "by_source_system": {"agent-utilities": 1},
    }

    ans_unknown = _code_context_answer(
        citations=[
            {
                "id": "code:a.py::run_agent",
                "symbol": "run_agent",
                "file": "a.py",
                "line": 10,
                "source_system": None,
            }
        ]
    )
    b_unknown = EvidenceBundle.from_code_context_answer(ans_unknown)
    assert b_unknown.source_authority == {}  # no fabricated ranking


# ---------------------------------------------------------------------------
# from_rag_result
# ---------------------------------------------------------------------------


def test_from_rag_result_id_only_when_no_evidence_passed():
    res = RagResult(
        answer="the fused answer.",
        evidence_ids=["n1", "n2"],
        trace=[
            StepTrace(op=StepOp.RETRIEVE, query="q", mode="vector", n_results=2),
            StepTrace(op=StepOp.ANSWER, query="q", n_results=2),
        ],
        success=True,
    )
    b = EvidenceBundle.from_rag_result(res)

    assert b.answer_candidate == "the fused answer."
    assert b.evidence_spans == [{"id": "n1"}, {"id": "n2"}]
    assert b.confidence is None
    # StepTrace entries + the final success marker are all present
    assert len(b.reasoning_trace) == 3
    assert b.reasoning_trace[-1] == {"step": "final", "success": True}
    assert b.next_actions == []  # success=True — nothing to suggest


def test_from_rag_result_uses_richer_evidence_when_provided():
    res = RagResult(answer="answer", evidence_ids=["n1"], trace=[], success=True)
    evidence = [{"id": "n1", "content": "n1 full content", "score": 0.8}]
    b = EvidenceBundle.from_rag_result(res, evidence=evidence)
    assert b.evidence_spans == evidence
    assert b.claims == evidence


def test_from_rag_result_failure_suggests_retry_and_confidence_stays_none():
    res = RagResult(answer="", evidence_ids=[], trace=[], success=False)
    b = EvidenceBundle.from_rag_result(res)
    assert b.confidence is None  # success=False is NOT converted into a number
    assert b.next_actions
    assert b.reasoning_trace[-1] == {"step": "final", "success": False}


# ---------------------------------------------------------------------------
# from_nl_query
# ---------------------------------------------------------------------------


def test_from_nl_query_maps_results_and_citations():
    payload = {
        "question": "which agents call run_agent?",
        "dialect": "cypher",
        "generated_query": "MATCH (a:Agent)-[:CALLS]->(f) WHERE f.name='run_agent' RETURN a",
        "results": [{"id": "agent:foo", "name": "foo"}],
        "row_count": 1,
        "citations": ["agent:foo"],
        "schema": {"node_labels": ["Agent"]},
    }
    b = EvidenceBundle.from_nl_query(payload)

    assert b.claims == payload["results"]
    assert b.evidence_spans == [{"ref": "agent:foo"}]
    assert b.answer_candidate == "1 row(s) for: which agents call run_agent?"
    assert b.confidence is None
    dumped = json.dumps(b.model_dump(), default=str)
    assert payload["generated_query"] in dumped
    assert "Agent" in dumped  # schema not silently dropped


def test_from_nl_query_accepts_nl_planner_request_key():
    # nl_planner.nl_query keys the question as "request" and adds "planner".
    payload = {
        "request": "how many agents exist?",
        "dialect": "sql",
        "generated_query": "SELECT count(*) FROM agents",
        "planner": "agent-utilities-fleet-llm",
        "results": [{"count": 5}],
        "row_count": 1,
        "citations": [],
        "schema": {},
    }
    b = EvidenceBundle.from_nl_query(payload)
    assert "how many agents exist?" in b.answer_candidate
    assert any(
        t.get("planner") == "agent-utilities-fleet-llm" for t in b.reasoning_trace
    )


def test_from_nl_query_error_path():
    payload = {"error": "query execution failed: boom", "schema": {}}
    b = EvidenceBundle.from_nl_query(payload)
    assert b.answer_candidate == ""
    assert b.confidence is None
    assert b.next_actions
    assert any(t.get("error") for t in b.reasoning_trace)


# ---------------------------------------------------------------------------
# from_engine_wire (legacy shape / forward-compat passthrough)
# ---------------------------------------------------------------------------


def test_from_engine_wire_passthrough():
    ws = {
        "answer_candidate": "engine answer",
        "claims": [{"id": "c1", "text": "x"}],
        "evidence_spans": [{"id": "e1"}],
        "confidence": 0.42,
        "next_actions": ["do the thing"],
    }
    b = EvidenceBundle.from_engine_wire(ws)
    assert b.answer_candidate == "engine answer"
    assert b.claims == ws["claims"]
    assert b.evidence_spans == ws["evidence_spans"]
    assert b.confidence == 0.42
    assert b.next_actions == ["do the thing"]


def test_from_engine_wire_degrades_cleanly_on_empty_dict():
    b = EvidenceBundle.from_engine_wire({})
    assert b.answer_candidate == ""
    assert b.confidence is None
    assert b.claims == []


# ---------------------------------------------------------------------------
# from_engine_wire — real E3 KnowledgeSet row mapping (D11)
# ---------------------------------------------------------------------------


def _knowledge_set_wire(**overrides) -> dict:
    payload = {
        "query": "what is run_agent's confidence?",
        "rows": [
            {
                "id": "belief:run_agent_stable",
                "kind": "Belief",
                "score": 0.91,
                "confidence": 0.8,
                "valid_time": "2026-01-01T00:00:00Z",
                "tx_time": "2026-01-02T00:00:00Z",
                "source_refs": ["src:code_context"],
                "evidence_refs": ["ev:a.py::run_agent"],
                "policy_labels": ["internal"],
                "text": "run_agent is stable",
            },
            {
                "id": "belief:run_agent_flaky",
                "kind": "Belief",
                "score": 0.4,
                "confidence": 0.3,
                "valid_time": "2025-12-01T00:00:00Z",
                "tx_time": "2025-12-02T00:00:00Z",
                "source_refs": [],
                "evidence_refs": ["ev:flaky_report"],
                "policy_labels": ["internal", "pii"],
                "text": "run_agent is flaky",
            },
        ],
    }
    payload.update(overrides)
    return payload


def test_from_engine_wire_maps_knowledge_set_rows():
    ws = _knowledge_set_wire()
    b = EvidenceBundle.from_engine_wire(ws)

    assert len(b.claims) == 2
    assert {c["id"] for c in b.claims} == {
        "belief:run_agent_stable",
        "belief:run_agent_flaky",
    }
    assert any(c["text"] == "run_agent is stable" for c in b.claims)

    # every source_ref/evidence_ref surfaces as its own evidence_span
    refs = {(s["ref"], s["type"]) for s in b.evidence_spans}
    assert ("src:code_context", "source_ref") in refs
    assert ("ev:a.py::run_agent", "evidence_ref") in refs
    assert ("ev:flaky_report", "evidence_ref") in refs

    # confidence = the TOP-SCORED row's own confidence (score 0.91 > 0.4)
    assert b.confidence == pytest.approx(0.8)

    # policy_labels dedup across rows, order-preserving
    assert b.policy_exclusions == ["internal", "pii"]

    # bitemporal coverage signal
    assert b.freshness["valid_time"] == {
        "min": "2025-12-01T00:00:00Z",
        "max": "2026-01-01T00:00:00Z",
    }
    assert b.freshness["tx_time"]["min"] == "2025-12-02T00:00:00Z"

    # nothing dropped — every row's raw fields recoverable from the trace
    row_steps = [t for t in b.reasoning_trace if t.get("step") == "knowledge_set_row"]
    assert len(row_steps) == 2

    # templated (never fabricated) answer_candidate when the wire has none
    assert b.answer_candidate == "2 row(s) for: what is run_agent's confidence?"


def test_from_engine_wire_knowledge_set_prefers_explicit_answer_and_confidence():
    ws = _knowledge_set_wire(answer_candidate="engine-composed answer", confidence=0.55)
    ws["rows"][0]["confidence"] = None
    ws["rows"][1]["confidence"] = None
    b = EvidenceBundle.from_engine_wire(ws)
    assert b.answer_candidate == "engine-composed answer"
    # no row carries a confidence — falls back to the wire-level one
    assert b.confidence == pytest.approx(0.55)


def test_from_engine_wire_knowledge_set_degrades_on_uncomparable_timestamps():
    ws = _knowledge_set_wire()
    ws["rows"][0]["valid_time"] = 123  # int, incomparable against the other row's str
    b = EvidenceBundle.from_engine_wire(ws)
    assert b.freshness == {}  # degrades cleanly rather than raising


def test_from_engine_wire_knowledge_set_carries_unmapped_meta_fields():
    ws = _knowledge_set_wire(dialect="cypher")
    b = EvidenceBundle.from_engine_wire(ws)
    dumped = json.dumps(b.model_dump(), default=str)
    assert "cypher" in dumped  # nothing silently dropped


# ---------------------------------------------------------------------------
# contradiction scanning
# ---------------------------------------------------------------------------


def test_contradiction_detector_fires_on_opposing_claims():
    ans = _code_context_answer(
        answer=(
            "The cache increases lookup latency for cold reads. "
            "The cache decreases lookup latency for cold reads."
        ),
    )
    b = EvidenceBundle.from_code_context_answer(ans)
    assert len(b.claims) == 2
    assert b.contradictions  # opposing polarity over the same topic is flagged
    finding = b.contradictions[0]
    assert {"new_id", "conflict_id", "similarity", "reason", "severity"} <= set(finding)


def test_contradiction_detector_silent_on_single_or_no_claims():
    ans = _code_context_answer(answer="Only one sentence here.")
    b = EvidenceBundle.from_code_context_answer(ans)
    assert b.contradictions == []


# ---------------------------------------------------------------------------
# U-124 / U-131 — contradiction detection is opt-in, not generic-row-wide
# ---------------------------------------------------------------------------
#
# Regression coverage for the defect where EVERY query row (not just explicit
# assertions) was scanned for contradictions: sibling metadata rows differing
# only in ordinary values (path/size/hash/...) were falsely reported as
# high-severity FRICTION at ~0.81-0.91 similarity, because the synthesized
# claim text fell back to a generic templated repr of the row's own
# id/kind/score/confidence/time/evidence fields — never real assertion
# content or declared contradiction semantics.


def _artifact_rows() -> list[dict]:
    """Three sibling `SourceArtifact`-shaped rows: same schema, different values."""
    return [
        {
            "id": "artifact:1",
            "kind": "SourceArtifact",
            "path": "/repo/a.py",
            "size": 1234,
            "hash": "aaa111",
        },
        {
            "id": "artifact:2",
            "kind": "SourceArtifact",
            "path": "/repo/b.py",
            "size": 5678,
            "hash": "bbb222",
        },
        {
            "id": "artifact:3",
            "kind": "SourceArtifact",
            "path": "/repo/c.py",
            "size": 9012,
            "hash": "ccc333",
        },
    ]


def test_sibling_metadata_rows_with_different_values_are_not_contradictions():
    rows = _artifact_rows()
    b = EvidenceBundle.from_payload({"rows": rows})
    assert len(b.claims) == 3  # every source row is retained regardless
    assert b.contradictions == []


def test_repeated_identical_structured_rows_are_not_contradictions():
    row = {
        "id": "e1",
        "comparison_key": "primary_region",
        "comparison_mode": "exclusive",
        "comparison_value": "us-east-1",
    }
    other = dict(row, id="e2")
    b = EvidenceBundle.from_payload({"rows": [row, other]})
    assert len(b.claims) == 2
    assert b.contradictions == []


def test_unrelated_prose_assertions_are_not_contradictions():
    rows = [
        {"id": "n1", "semantic_role": "assertion", "text": "The sky is blue today."},
        {
            "id": "n2",
            "semantic_role": "assertion",
            "text": "Bananas are a yellow fruit.",
        },
    ]
    b = EvidenceBundle.from_payload({"rows": rows})
    assert len(b.claims) == 2
    assert b.contradictions == []


def test_explicit_opposing_assertions_yield_exactly_one_finding():
    rows = [
        {
            "id": "c1",
            "semantic_role": "assertion",
            "text": "The cache increases lookup latency for cold reads.",
        },
        {
            "id": "c2",
            "semantic_role": "assertion",
            "text": "The cache decreases lookup latency for cold reads.",
        },
    ]
    b = EvidenceBundle.from_payload({"rows": rows})
    assert len(b.claims) == 2
    assert len(b.contradictions) == 1
    finding = b.contradictions[0]
    assert {finding["new_id"], finding["conflict_id"]} == {"c1", "c2"}
    assert finding["comparison_rule"] == "natural_language_opposition"


def test_exclusive_structured_values_under_one_key_yield_exactly_one_finding():
    rows = [
        {
            "id": "e1",
            "comparison_key": "primary_region",
            "comparison_mode": "exclusive",
            "comparison_value": "us-east-1",
        },
        {
            "id": "e2",
            "comparison_key": "primary_region",
            "comparison_mode": "exclusive",
            "comparison_value": "us-west-2",
        },
    ]
    b = EvidenceBundle.from_payload({"rows": rows})
    assert len(b.claims) == 2
    assert len(b.contradictions) == 1
    finding = b.contradictions[0]
    assert {finding["new_id"], finding["conflict_id"]} == {"e1", "e2"}
    assert finding["comparison_rule"] == "structured_mutual_exclusivity"
    assert finding["comparison_key"] == "primary_region"


def test_polarity_conflict_yields_one_finding():
    rows = [
        {
            "id": "p1",
            "semantic_role": "assertion",
            "text": "The migration is healthy.",
        },
        {
            "id": "p2",
            "semantic_role": "assertion",
            "text": "The migration is not healthy.",
        },
    ]
    b = EvidenceBundle.from_payload({"rows": rows})
    assert len(b.claims) == 2
    assert len(b.contradictions) == 1
    finding = b.contradictions[0]
    assert {finding["new_id"], finding["conflict_id"]} == {"p1", "p2"}


def test_missing_comparison_key_yields_zero_contradictions():
    rows = [
        {"id": "m1", "comparison_mode": "exclusive", "comparison_value": "a"},
        {"id": "m2", "comparison_mode": "exclusive", "comparison_value": "b"},
    ]
    b = EvidenceBundle.from_payload({"rows": rows})
    assert len(b.claims) == 2
    assert b.contradictions == []


def test_mixed_eligible_and_ineligible_claims_only_eligible_participate():
    rows = _artifact_rows()[:2] + [
        {
            "id": "e1",
            "comparison_key": "primary_region",
            "comparison_mode": "exclusive",
            "comparison_value": "us-east-1",
        },
        {
            "id": "e2",
            "comparison_key": "primary_region",
            "comparison_mode": "exclusive",
            "comparison_value": "us-west-2",
        },
    ]
    b = EvidenceBundle.from_payload({"rows": rows})
    assert len(b.claims) == 4  # every row retained, ineligible or not
    assert len(b.contradictions) == 1
    finding = b.contradictions[0]
    assert {finding["new_id"], finding["conflict_id"]} == {"e1", "e2"}
    ineligible_ids = {"artifact:1", "artifact:2"}
    assert not ineligible_ids & {finding["new_id"], finding["conflict_id"]}


def test_all_source_rows_remain_intact_across_every_scenario():
    scenarios = [
        _artifact_rows(),
        [
            {
                "id": "e1",
                "comparison_key": "k",
                "comparison_mode": "exclusive",
                "comparison_value": "a",
            },
            {
                "id": "e2",
                "comparison_key": "k",
                "comparison_mode": "exclusive",
                "comparison_value": "a",
            },
        ],
        [
            {"id": "n1", "semantic_role": "assertion", "text": "unrelated prose one."},
            {"id": "n2", "semantic_role": "assertion", "text": "unrelated prose two."},
        ],
        [
            {
                "id": "c1",
                "semantic_role": "assertion",
                "text": "The cache increases lookup latency for cold reads.",
            },
            {
                "id": "c2",
                "semantic_role": "assertion",
                "text": "The cache decreases lookup latency for cold reads.",
            },
        ],
    ]
    for rows in scenarios:
        b = EvidenceBundle.from_payload({"rows": rows})
        assert [c["id"] for c in b.claims] == [r["id"] for r in rows]
