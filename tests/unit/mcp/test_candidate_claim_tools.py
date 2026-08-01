"""Tests for the graph_candidate_claims MCP tool + REST twin (D-CE-2).

CONCEPT:AU-KG.enrichment.candidate-claim-extraction,
CONCEPT:AU-KG.identity.entity-resolution-candidates.

Proves the dedicated MCP/REST surface this item asked for: 'propose' really
calls ``CandidateClaimExtractor.propose`` (a fake stream_fn stands in for the
LLM the same way ``test_candidate_claims_wiring.py`` already proves the
extractor itself works — this file's job is the DISPATCH wiring, not
re-proving extraction internals), and 'resolve_identities' really calls the
REAL (unmocked, pure) ``resolve_identity_candidates`` plus, when
``persist=true``, ``write_candidate`` against a real double engine.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from agent_utilities.mcp import kg_server

pytestmark = pytest.mark.concept("AU-KG.enrichment.candidate-claim-extraction")


class _EdgeStubEngine:
    """Minimal ``link_nodes`` double so ``write_candidate`` (the ONE ungated
    write this pipeline performs) can be proven for real."""

    def __init__(self) -> None:
        self.edges: list[tuple[str, str, str, dict[str, Any]]] = []

    def link_nodes(
        self, source: str, target: str, relationship: Any, *, properties=None
    ) -> None:
        rel = relationship.value if hasattr(relationship, "value") else str(relationship)
        self.edges.append((source, target, rel, dict(properties or {})))


@pytest.fixture
def registered():
    kg_server.ensure_tools_registered()
    return kg_server


@pytest.fixture
def stub_engine(monkeypatch):
    engine = _EdgeStubEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    return engine


def _fact_json(subject: str, predicate: str, obj: str, evidence_span: str) -> str:
    return json.dumps(
        {
            "title": f"{subject} {predicate} {obj}",
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "evidence_span": evidence_span,
            "confidence": 88,
            "tags": ["t"],
        }
    )


def _one_shot_stream(chunk: str):
    async def _stream(_prompt: str, _seed: int) -> AsyncGenerator[str, None]:
        yield chunk

    return _stream


def test_graph_candidate_claims_registered_on_both_surfaces(registered):
    assert "graph_candidate_claims" in registered.REGISTERED_TOOLS
    assert (
        registered.ACTION_TOOL_ROUTES.get("graph_candidate_claims")
        == "/graph/candidate-claims"
    )


@pytest.mark.asyncio
async def test_propose_extracts_a_real_candidate_with_resolvable_evidence(
    registered, monkeypatch
):
    """A real prose fragment, run through the SAME extraction machinery
    test_candidate_claims_wiring.py proves works, dispatched through the MCP
    action core (kg_server._execute_tool) — the live path a real caller uses."""
    import agent_utilities.knowledge_graph.extraction.candidate_claims as cc

    quote = "Acme Corp acquired Globex in 2024"
    stream = _one_shot_stream(
        _fact_json("Acme Corp", "acquired", "Globex", quote)
    )
    monkeypatch.setattr(
        cc,
        "make_streaming_extract_fn",
        lambda *a, **k: stream,
    )

    fragments_json = json.dumps(
        [
            {
                "fragment_id": "frag:doc-1:p3",
                "text": "Acme Corp acquired Globex in 2024 for an undisclosed sum.",
            }
        ]
    )
    res = json.loads(
        await kg_server._execute_tool(
            "graph_candidate_claims",
            action="propose",
            text="Acme Corp acquired Globex in 2024 for an undisclosed sum.",
            source_id="source:doc-1",
            fragments_json=fragments_json,
            dedup=False,
        )
    )

    assert res["action"] == "propose"
    assert res["counts"]["accepted"] == 1
    assert len(res["candidates"]) == 1
    candidate = res["candidates"][0]
    assert candidate["subject"] == "Acme Corp"
    assert candidate["predicate"] == "acquired"
    assert candidate["object"] == "Globex"
    assert candidate["model_confidence"] == pytest.approx(0.88)
    assert candidate["review_bucket"] == "accepted"
    assert candidate["evidence"][0]["fragment_id"] == "frag:doc-1:p3"
    assert candidate["evidence"][0]["quote"] == quote
    assert res["extraction_run_id"]
    assert res["unresolved_evidence"] == 0


@pytest.mark.asyncio
async def test_propose_requires_text_and_source_id(registered):
    res = json.loads(
        await kg_server._execute_tool("graph_candidate_claims", action="propose")
    )
    assert "requires text and source_id" in res["error"]


@pytest.mark.asyncio
async def test_propose_rejects_malformed_fragments_json(registered):
    res = json.loads(
        await kg_server._execute_tool(
            "graph_candidate_claims",
            action="propose",
            text="x",
            source_id="s1",
            fragments_json="not json",
        )
    )
    assert "invalid fragments_json" in res["error"]


@pytest.mark.asyncio
async def test_resolve_identities_returns_a_real_candidate_never_a_merge(
    registered, stub_engine
):
    """Real, unmocked resolve_identity_candidates — near-identical names ->
    ONE candidate, status 'candidate', never persisted unless asked."""
    records_json = json.dumps(
        [
            {"id": "entity:a", "name": "payments platform"},
            {"id": "entity:b", "name": "payments-platform"},
        ]
    )
    res = json.loads(
        await kg_server._execute_tool(
            "graph_candidate_claims",
            action="resolve_identities",
            records_json=records_json,
        )
    )

    assert res["action"] == "resolve_identities"
    assert len(res["candidates"]) == 1
    candidate = res["candidates"][0]
    assert candidate["status"] == "candidate"
    assert {candidate["entity_a"], candidate["entity_b"]} == {
        "entity:a",
        "entity:b",
    }
    assert res["persisted"] == 0
    assert stub_engine.edges == []  # persist=false (default): nothing written


@pytest.mark.asyncio
async def test_resolve_identities_persist_writes_possible_same_as_edges(
    registered, stub_engine
):
    records_json = json.dumps(
        [
            {"id": "entity:a", "name": "payments platform"},
            {"id": "entity:b", "name": "payments-platform"},
        ]
    )
    res = json.loads(
        await kg_server._execute_tool(
            "graph_candidate_claims",
            action="resolve_identities",
            records_json=records_json,
            persist=True,
        )
    )

    assert res["persisted"] == 1
    assert len(stub_engine.edges) == 1
    source, target, rel, props = stub_engine.edges[0]
    assert {source, target} == {"entity:a", "entity:b"}
    assert rel == "possible_same_as"
    assert props["status"] == "candidate"


@pytest.mark.asyncio
async def test_resolve_identities_no_evidence_yields_no_candidates(registered):
    records_json = json.dumps(
        [
            {"id": "entity:x", "name": "Aurora Freight Logistics"},
            {"id": "entity:y", "name": "Nimbus Data Analytics"},
        ]
    )
    res = json.loads(
        await kg_server._execute_tool(
            "graph_candidate_claims",
            action="resolve_identities",
            records_json=records_json,
        )
    )
    assert res["candidates"] == []


@pytest.mark.asyncio
async def test_resolve_identities_rejects_malformed_records_json(registered):
    res = json.loads(
        await kg_server._execute_tool(
            "graph_candidate_claims",
            action="resolve_identities",
            records_json="not json",
        )
    )
    assert "invalid records_json" in res["error"]


@pytest.mark.asyncio
async def test_unknown_action(registered):
    res = json.loads(
        await kg_server._execute_tool("graph_candidate_claims", action="bogus")
    )
    assert "unknown action" in res["error"]
