"""Unit tests for the unified EvidenceBundle envelope (Epistemic Substrate Program, C1).

Exercises each ``from_*`` classmethod against realistic fixture inputs shaped exactly
like the wrapped surfaces (``CodeContextAnswer``, ``RagResult``, the ``nl_to_query``/
``nl_planner.nl_query`` payload dict), asserting: field mapping is correct, nothing
from the source payload is silently dropped, and ``confidence`` is NEVER fabricated
(stays ``None`` unless a caller explicitly threads one in via an override).
"""

from __future__ import annotations

import json

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
    assert any(t.get("planner") == "agent-utilities-fleet-llm" for t in b.reasoning_trace)


def test_from_nl_query_error_path():
    payload = {"error": "query execution failed: boom", "schema": {}}
    b = EvidenceBundle.from_nl_query(payload)
    assert b.answer_candidate == ""
    assert b.confidence is None
    assert b.next_actions
    assert any(t.get("error") for t in b.reasoning_trace)


# ---------------------------------------------------------------------------
# from_engine_wire (stub / forward-compat passthrough)
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
    assert {"new_id", "conflict_id", "similarity", "reason", "severity"} <= set(
        finding
    )


def test_contradiction_detector_silent_on_single_or_no_claims():
    ans = _code_context_answer(answer="Only one sentence here.")
    b = EvidenceBundle.from_code_context_answer(ans)
    assert b.contradictions == []
