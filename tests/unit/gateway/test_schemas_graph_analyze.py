"""Contract tests for agent_utilities.gateway.schemas.graph_analyze.

Covers every model backing the ``/api/graph/analyze/*`` and ``/api/graph/search/*``
REST route families (CONCEPT:AU-KG.query.typed-rest-schemas): each request model
accepts a realistic payload the live handler in
``agent_utilities/mcp/kg_server.py`` would currently accept, rejects a payload with a
wrong-typed field, and every field on every model in the module carries a non-empty
description (so the generated OpenAPI docs cannot silently regress to undocumented
fields).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from agent_utilities.gateway.schemas import graph_analyze as schemas
from agent_utilities.models.evidence_bundle import EvidenceBundle

# ══════════════════════════════════════════════════════════════════
# Every model this module defines, for the blanket "descriptions never rot" sweep.
# ══════════════════════════════════════════════════════════════════

ALL_MODEL_NAMES = list(schemas.__all__)

# RoutesQuery (GET /api/graph/analyze/routes) and NoParamsRequest (POST
# /evolve-model|/forecast|/causal|/invariant) are DELIBERATELY field-less — their
# routes' handlers read no request data at all (see their docstrings) — so they are
# exempt from "declares no fields" but still covered by every other assertion below.
_INTENTIONALLY_FIELDLESS = {"RoutesQuery", "NoParamsRequest"}


def _model_classes() -> list[type[BaseModel]]:
    return [getattr(schemas, name) for name in ALL_MODEL_NAMES]


@pytest.mark.parametrize("model_cls", _model_classes(), ids=ALL_MODEL_NAMES)
def test_every_field_has_a_nonempty_description(model_cls: type[BaseModel]) -> None:
    """Loop over model_fields so a future field addition without a description fails
    loudly instead of silently shipping undocumented into the OpenAPI schema."""
    if model_cls.__name__ in _INTENTIONALLY_FIELDLESS:
        assert model_cls.model_fields == {}
        return
    assert model_cls.model_fields, f"{model_cls.__name__} declares no fields at all"
    for field_name, field_info in model_cls.model_fields.items():
        description = field_info.description
        assert description and description.strip(), (
            f"{model_cls.__name__}.{field_name} has no description "
            f"(got {description!r})"
        )


# ══════════════════════════════════════════════════════════════════
# Response envelopes
# ══════════════════════════════════════════════════════════════════


def test_analyze_response_accepts_a_real_evidence_bundle_shaped_result() -> None:
    bundle = EvidenceBundle(answer_candidate="graph-os has 47 registered tools.")
    payload = {"status": "success", "result": bundle.model_dump()}
    parsed = schemas.AnalyzeResponse.model_validate(payload)
    assert parsed.status == "success"
    assert parsed.result.answer_candidate == "graph-os has 47 registered tools."


def test_analyze_response_rejects_non_dict_result() -> None:
    with pytest.raises(ValidationError):
        schemas.AnalyzeResponse.model_validate({"status": "success", "result": "oops"})


def test_search_text_response_accepts_free_form_text_result() -> None:
    payload = {
        "status": "success",
        "result": "[Concept] Widget (ID: n1) - Score: 0.87\nA widget.",
    }
    parsed = schemas.SearchTextResponse.model_validate(payload)
    assert parsed.result.startswith("[Concept]")


def test_search_text_response_rejects_non_string_result() -> None:
    with pytest.raises(ValidationError):
        schemas.SearchTextResponse.model_validate(
            {"status": "success", "result": {"not": "a string"}}
        )


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/analyze  (GraphAnalyzeRequest)
# ══════════════════════════════════════════════════════════════════


def test_graph_analyze_request_accepts_realistic_payload() -> None:
    payload = {
        "action": "inspect",
        "query": "why is the engine degraded",
        "top_k": 5,
        "node_id": "code:foo.py::bar",
        "depth": 3,
        "target": "engine",
    }
    parsed = schemas.GraphAnalyzeRequest.model_validate(payload)
    assert parsed.action == "inspect"
    assert parsed.top_k == 5


def test_graph_analyze_request_accepts_empty_body() -> None:
    # Every field defaults, matching `body.get(...)`-style handler tolerance for a
    # missing key — mirrored here since the tool call resolves the same defaults.
    parsed = schemas.GraphAnalyzeRequest.model_validate({})
    assert parsed.action == "inspect"
    assert parsed.top_k == 10


def test_graph_analyze_request_rejects_wrong_typed_top_k() -> None:
    with pytest.raises(ValidationError):
        schemas.GraphAnalyzeRequest.model_validate({"top_k": "not-an-int"})


def test_graph_analyze_request_rejects_unknown_field() -> None:
    # U-74 (_validate_tool_kwargs_against_signature) already 400s an unrecognized
    # field for this route today — extra='forbid' mirrors that, not a new rejection.
    with pytest.raises(ValidationError):
        schemas.GraphAnalyzeRequest.model_validate({"bogus_field": "x"})


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/analyze/synthesize, /deep-extract  (ResearchQueryTopKRequest)
# ══════════════════════════════════════════════════════════════════


def test_research_query_topk_request_accepts_realistic_payload() -> None:
    parsed = schemas.ResearchQueryTopKRequest.model_validate(
        {"query": "assimilate arXiv:2606.12087", "top_k": 15}
    )
    assert parsed.query == "assimilate arXiv:2606.12087"
    assert parsed.top_k == 15


def test_research_query_topk_request_allows_unknown_field() -> None:
    # The handler reads only query/top_k via body.get(...) and silently ignores the
    # rest — extra='allow' preserves that a caller sending a legacy/extra field
    # keeps working.
    parsed = schemas.ResearchQueryTopKRequest.model_validate(
        {"query": "x", "legacy_field": "kept-but-unused"}
    )
    assert parsed.query == "x"


def test_research_query_topk_request_rejects_wrong_typed_top_k() -> None:
    with pytest.raises(ValidationError):
        schemas.ResearchQueryTopKRequest.model_validate({"top_k": ["not", "an", "int"]})


# ══════════════════════════════════════════════════════════════════
# GET /api/graph/analyze/blast-radius  (BlastRadiusQuery)
# ══════════════════════════════════════════════════════════════════


def test_blast_radius_query_accepts_realistic_payload() -> None:
    parsed = schemas.BlastRadiusQuery.model_validate(
        {"id": "code:foo.py::bar", "depth": 4}
    )
    assert parsed.depth == 4


def test_blast_radius_query_rejects_wrong_typed_depth() -> None:
    with pytest.raises(ValidationError):
        schemas.BlastRadiusQuery.model_validate({"depth": "deep"})


# ══════════════════════════════════════════════════════════════════
# GET /api/graph/analyze/inspect  (InspectQuery)
# ══════════════════════════════════════════════════════════════════


def test_inspect_query_accepts_realistic_payload() -> None:
    parsed = schemas.InspectQuery.model_validate({"target": "engine"})
    assert parsed.target == "engine"


def test_inspect_query_rejects_wrong_typed_target() -> None:
    with pytest.raises(ValidationError):
        schemas.InspectQuery.model_validate({"target": 12345})


# ══════════════════════════════════════════════════════════════════
# GET /api/graph/analyze/call-graph  (CallGraphQuery)
# ══════════════════════════════════════════════════════════════════


def test_call_graph_query_accepts_realistic_payload() -> None:
    parsed = schemas.CallGraphQuery.model_validate(
        {"id": "code:foo.py::bar", "direction": "callers"}
    )
    assert parsed.direction == "callers"


def test_call_graph_query_accepts_legacy_target_alias() -> None:
    parsed = schemas.CallGraphQuery.model_validate(
        {"id": "code:foo.py::bar", "target": "inherits"}
    )
    assert parsed.target == "inherits"
    assert parsed.direction is None


def test_call_graph_query_rejects_wrong_typed_id() -> None:
    with pytest.raises(ValidationError):
        schemas.CallGraphQuery.model_validate({"id": {"not": "a string"}})


# ══════════════════════════════════════════════════════════════════
# GET /api/graph/analyze/similar-code  (SimilarCodeQuery)
# ══════════════════════════════════════════════════════════════════


def test_similar_code_query_accepts_realistic_payload() -> None:
    parsed = schemas.SimilarCodeQuery.model_validate(
        {"id": "code:foo.py::bar", "top_k": 5}
    )
    assert parsed.top_k == 5


def test_similar_code_query_rejects_wrong_typed_top_k() -> None:
    with pytest.raises(ValidationError):
        schemas.SimilarCodeQuery.model_validate({"top_k": "five"})


# ══════════════════════════════════════════════════════════════════
# GET /api/graph/analyze/routes  (RoutesQuery)
# ══════════════════════════════════════════════════════════════════


def test_routes_query_accepts_empty_payload() -> None:
    parsed = schemas.RoutesQuery.model_validate({})
    assert isinstance(parsed, schemas.RoutesQuery)


def test_routes_query_allows_unknown_field_but_still_validates_extras_shape() -> None:
    # extra='allow' + no declared fields: passing a mapping still round-trips fine.
    parsed = schemas.RoutesQuery.model_validate({"anything": "ignored-by-the-handler"})
    assert (
        parsed.model_dump(exclude_none=True).get("anything") == "ignored-by-the-handler"
    )


# ══════════════════════════════════════════════════════════════════
# GET /api/graph/analyze/code-metrics, /arch-report  (ScopeTopKQuery)
# ══════════════════════════════════════════════════════════════════


def test_scope_topk_query_accepts_realistic_payload() -> None:
    parsed = schemas.ScopeTopKQuery.model_validate(
        {"scope": "agent_utilities/gateway", "top_k": 20}
    )
    assert parsed.scope == "agent_utilities/gateway"
    assert parsed.top_k == 20


def test_scope_topk_query_accepts_legacy_target_alias() -> None:
    parsed = schemas.ScopeTopKQuery.model_validate(
        {"target": "agent_utilities/gateway"}
    )
    assert parsed.target == "agent_utilities/gateway"
    assert parsed.scope is None


def test_scope_topk_query_rejects_wrong_typed_top_k() -> None:
    with pytest.raises(ValidationError):
        schemas.ScopeTopKQuery.model_validate({"top_k": "lots"})


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/analyze/adr  (AdrRequest)
# ══════════════════════════════════════════════════════════════════


def test_adr_request_accepts_realistic_payload() -> None:
    parsed = schemas.AdrRequest.model_validate(
        {
            "title": "Adopt typed gateway schemas",
            "status": "accepted",
            "decision": "Model every /api/graph/* route with Pydantic.",
        }
    )
    assert parsed.title == "Adopt typed gateway schemas"


def test_adr_request_accepts_empty_body_to_list_adrs() -> None:
    parsed = schemas.AdrRequest.model_validate({})
    assert parsed.title == ""


def test_adr_request_rejects_wrong_typed_status() -> None:
    with pytest.raises(ValidationError):
        schemas.AdrRequest.model_validate({"status": 42})


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/analyze/harness-gate  (HarnessGateRequest)
# ══════════════════════════════════════════════════════════════════


def test_harness_gate_request_accepts_realistic_payload() -> None:
    parsed = schemas.HarnessGateRequest.model_validate(
        {
            "edits": [{"path": "prompts/foo.md", "diff": "..."}],
            "variants": ["baseline", "candidate"],
            "pathologies": [],
        }
    )
    assert parsed.edits[0]["path"] == "prompts/foo.md"


def test_harness_gate_request_accepts_empty_body() -> None:
    parsed = schemas.HarnessGateRequest.model_validate({})
    assert parsed.edits == []
    assert parsed.variants is None


def test_harness_gate_request_rejects_wrong_typed_edits() -> None:
    with pytest.raises(ValidationError):
        schemas.HarnessGateRequest.model_validate({"edits": "not-a-list"})


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/analyze/code-context  (CodeContextRequest)
# ══════════════════════════════════════════════════════════════════


def test_code_context_request_accepts_realistic_payload() -> None:
    parsed = schemas.CodeContextRequest.model_validate(
        {
            "query": "how does _execute_tool dispatch a sync tool",
            "intent": "how",
            "node_id": "code:kg_server.py::_execute_tool",
            "top_k": 8,
            "depth": 2,
            "cross_repo": True,
        }
    )
    assert parsed.cross_repo is True


def test_code_context_request_rejects_wrong_typed_cross_repo() -> None:
    with pytest.raises(ValidationError):
        schemas.CodeContextRequest.model_validate({"cross_repo": "yes-please"})


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/analyze/explain  (ExplainRequest)
# ══════════════════════════════════════════════════════════════════


def test_explain_request_accepts_realistic_payload() -> None:
    parsed = schemas.ExplainRequest.model_validate(
        {"query": "is my change live", "domain": "deploy", "intent": "status"}
    )
    assert parsed.domain == "deploy"


def test_explain_request_rejects_wrong_typed_top_k() -> None:
    with pytest.raises(ValidationError):
        schemas.ExplainRequest.model_validate({"top_k": "many"})


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/analyze/context  (ContextRequest)
# ══════════════════════════════════════════════════════════════════


def test_context_request_accepts_realistic_payload() -> None:
    parsed = schemas.ContextRequest.model_validate(
        {"target": "engine", "query": "why did engine restart", "top_k": 5}
    )
    assert parsed.target == "engine"


def test_context_request_rejects_wrong_typed_top_k() -> None:
    with pytest.raises(ValidationError):
        schemas.ContextRequest.model_validate({"top_k": "several"})


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/analyze/evaluate  (EvaluateRequest)
# ══════════════════════════════════════════════════════════════════


def test_evaluate_request_accepts_realistic_payload() -> None:
    parsed = schemas.EvaluateRequest.model_validate({"target": "agent:sonnet-5"})
    assert parsed.target == "agent:sonnet-5"


def test_evaluate_request_rejects_wrong_typed_target() -> None:
    with pytest.raises(ValidationError):
        schemas.EvaluateRequest.model_validate({"target": ["not", "a", "string"]})


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/analyze/evolve-model, /forecast, /causal, /invariant (NoParamsRequest)
# ══════════════════════════════════════════════════════════════════


def test_no_params_request_accepts_empty_body() -> None:
    parsed = schemas.NoParamsRequest.model_validate({})
    assert isinstance(parsed, schemas.NoParamsRequest)


def test_no_params_request_accepts_arbitrary_body_since_handler_never_reads_it() -> (
    None
):
    parsed = schemas.NoParamsRequest.model_validate({"anything": "at-all", "n": 1})
    dumped = parsed.model_dump(exclude_none=True)
    assert dumped.get("anything") == "at-all"


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/search  (GraphSearchRequest)
# ══════════════════════════════════════════════════════════════════


def test_graph_search_request_accepts_realistic_payload() -> None:
    parsed = schemas.GraphSearchRequest.model_validate(
        {
            "query": "AU-KG.retrieval.context-compiler",
            "mode": "concept",
            "top_k": 5,
            "self_correct": True,
            "as_of": "2026-08-01T00:00:00Z",
            "connection": "default",
            "graph": "",
            "token_budget": 0,
        }
    )
    assert parsed.mode == "concept"
    assert parsed.self_correct is True


def test_graph_search_request_requires_query() -> None:
    # The tool declares `query` with no default — it is the one genuinely required
    # field across this whole module.
    with pytest.raises(ValidationError):
        schemas.GraphSearchRequest.model_validate({})


def test_graph_search_request_rejects_wrong_typed_self_correct() -> None:
    with pytest.raises(ValidationError):
        schemas.GraphSearchRequest.model_validate(
            {"query": "x", "self_correct": "definitely"}
        )


def test_graph_search_request_rejects_unknown_field() -> None:
    # Same U-74 enforcement as GraphAnalyzeRequest — the tool has no **kwargs.
    with pytest.raises(ValidationError):
        schemas.GraphSearchRequest.model_validate({"query": "x", "bogus_field": "y"})


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/search/concept, /analogy, /memory, /dci  (SearchQueryTopKRequest)
# ══════════════════════════════════════════════════════════════════


def test_search_query_topk_request_accepts_realistic_payload() -> None:
    parsed = schemas.SearchQueryTopKRequest.model_validate(
        {"query": "AU-KG.retrieval.context-compiler", "top_k": 5}
    )
    assert parsed.top_k == 5


def test_search_query_topk_request_rejects_wrong_typed_top_k() -> None:
    with pytest.raises(ValidationError):
        schemas.SearchQueryTopKRequest.model_validate({"top_k": "a lot"})


# ══════════════════════════════════════════════════════════════════
# POST /api/graph/search/discover  (SearchDiscoverRequest)
# ══════════════════════════════════════════════════════════════════


def test_search_discover_request_accepts_realistic_payload() -> None:
    parsed = schemas.SearchDiscoverRequest.model_validate({"query": "deploy status"})
    assert parsed.query == "deploy status"


def test_search_discover_request_rejects_wrong_typed_query() -> None:
    with pytest.raises(ValidationError):
        schemas.SearchDiscoverRequest.model_validate({"query": 123})
