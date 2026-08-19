"""NE-151 UQL property-reference planner contract.

The fixture mirrors the served code-context ask: a planner response is routed
through the AU schema probe, canonical UQL boundary, engine surface and evidence
shape.  It never contacts a model or a live GraphOS process.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.core import nl_planner
from agent_utilities.models.evidence_bundle import EvidenceBundle

pytestmark = pytest.mark.concept("AU-KG.query.ask-gateway-rest-twin")


class _CodeContextEngine:
    """A live-shaped read-only engine double with a strict UQL seam."""

    expected = "MATCH (:Code) WHERE name = 'build_code_context' |> LIMIT 8"

    def __init__(self) -> None:
        self.uql_calls: list[str] = []

    def query_cypher(self, query: str, *_args, **_kwargs):
        if query.startswith("MATCH (n) RETURN") and "n.node_type AS nt" in query:
            return [{"nt": "Code"}]
        return []

    def uql(self, query: str):
        self.uql_calls.append(query)
        if query != self.expected:
            raise RuntimeError("UQL parse error: expected the canonical code query")
        return [
            {
                "id": "code:agent-utilities:code_context",
                "name": "build_code_context",
                "source_uri": "repo://agent-utilities/agent_utilities/knowledge_graph/retrieval/code_context.py",
            }
        ]


def _planner(payloads: list[dict[str, str]]):
    responses = iter(json.dumps(payload) for payload in payloads)
    prompts: list[str] = []

    def run(prompt: str, _system: str) -> str:
        prompts.append(prompt)
        return next(responses)

    return nl_planner.AuNlPlanner(run=run), prompts


def test_code_context_ask_lowers_props_wrapper_to_parseable_evidence_plan():
    """The reproduced ``props.name`` response never reaches the UQL parser."""
    engine = _CodeContextEngine()
    planner, _prompts = _planner(
        [
            {
                "dialect": "uql",
                "query": "MATCH (:Code) WHERE props.name = 'build_code_context' |> LIMIT 8",
            }
        ]
    )

    result = nl_planner.nl_query(
        engine,
        "Where is build_code_context defined?",
        dialect="uql",
        planner=planner,
        limit=8,
    )

    assert "error" not in result, result
    assert engine.uql_calls == [_CodeContextEngine.expected]
    assert result["generated_query"] == _CodeContextEngine.expected
    assert result["results"]
    assert result["citations"] == ["code:agent-utilities:code_context"]
    assert result["schema"]["node_labels"] == ["Code"]

    plan = result["plan"]
    assert plan["bounded"] is True
    assert plan["grammar_version"] == nl_planner.UQL_GRAMMAR_VERSION
    assert plan["corrections"] == [
        {
            "kind": "property_reference",
            "from": "props.name",
            "to": "name",
            "reason": "UQL v1 predicates use bare property identifiers",
        }
    ]
    bundle = EvidenceBundle.from_nl_query(result)
    assert bundle.reasoning_trace[0]["plan"] == plan
    assert bundle.reasoning_trace[0]["attempts"] == []


def test_unknown_dotted_property_replans_once_without_empty_answer():
    """An untranslatable qualifier is corrected or returned as an explicit error."""
    engine = _CodeContextEngine()
    planner, prompts = _planner(
        [
            {
                "dialect": "uql",
                "query": "MATCH (:Code) WHERE node.name = 'build_code_context' |> LIMIT 8",
            },
            {"dialect": "uql", "query": _CodeContextEngine.expected},
        ]
    )

    result = nl_planner.nl_query(
        engine,
        "Where is build_code_context defined?",
        dialect="uql",
        planner=planner,
        max_corrections=1,
    )

    assert "error" not in result, result
    assert engine.uql_calls == [_CodeContextEngine.expected]
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["phase"] == "planning"
    assert nl_planner.UQL_GRAMMAR_VERSION in result["attempts"][0]["error"]
    assert len(prompts) == 2
    assert nl_planner.UQL_GRAMMAR_VERSION in prompts[1]
    assert result["results"]  # malformed UQL was not disguised as an empty answer


def test_malformed_uql_exhausts_bounded_replan_with_no_engine_call():
    engine = _CodeContextEngine()
    planner, _prompts = _planner(
        [
            {
                "dialect": "uql",
                "query": "MATCH (:Code) WHERE node.name = 'build_code_context' |> LIMIT 8",
            },
            {
                "dialect": "uql",
                "query": "MATCH (:Code) WHERE node.name = 'build_code_context' |> LIMIT 8",
            },
        ]
    )

    result = nl_planner.nl_query(
        engine,
        "Where is build_code_context defined?",
        dialect="uql",
        planner=planner,
        max_corrections=1,
    )

    assert "error" in result
    assert "planning failed" in result["error"]
    assert nl_planner.UQL_GRAMMAR_VERSION in result["error"]
    assert "results" not in result
    assert len(result["attempts"]) == 2
    assert engine.uql_calls == []
