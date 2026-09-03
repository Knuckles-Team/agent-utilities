"""Tests for `agent_utilities.gateway.schemas.graph_core` (SCHEMA LANE 3/3).

Per model: a realistic currently-valid payload is accepted, an invalid one is
rejected, and every declared field carries a non-empty description. Plus two
contract-pinning tests called out by the lane brief: the `/tools` flat-list
shape agent-terminal-ui depends on, and the canonical field name
`/graph/query` accepts.
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel, ValidationError

from agent_utilities.gateway.schemas import graph_core as schemas
from agent_utilities.models.evidence_bundle import EvidenceBundle

# ── every BaseModel this module defines (used by the blanket description
#    sweep below) — collected by introspection so a new model added later is
#    automatically covered without editing this list. ──────────────────────
ALL_MODELS: list[type[BaseModel]] = [
    obj
    for _name, obj in vars(schemas).items()
    if inspect.isclass(obj)
    and issubclass(obj, BaseModel)
    and obj.__module__ == schemas.__name__
]


def test_every_model_field_has_a_non_empty_description() -> None:
    """Undescribed fields are a failure of this lane per the brief."""
    offenders = []
    for model in ALL_MODELS:
        for field_name, field_info in model.model_fields.items():
            desc = field_info.description
            if not desc or not desc.strip():
                offenders.append(f"{model.__name__}.{field_name}")
    assert not offenders, f"fields missing a description: {offenders}"


def test_collected_at_least_one_model_per_route_family() -> None:
    """Sanity check the introspection sweep itself found real models (guards
    against a silent import/collection failure making the sweep above vacuous).
    """
    assert len(ALL_MODELS) >= 35


# ══════════════════════════════════════════════════════════════════
# graph_query/federated
# ══════════════════════════════════════════════════════════════════


def test_graph_query_federated_accepts_dict_params() -> None:
    """Unlike /graph/query, the federated twin accepts a native JSON object
    for `params` (converted server-side via `_to_json_str`)."""
    req = schemas.GraphQueryFederatedRequest(
        query="MATCH (n) RETURN n", params={"limit": 5}, reference_id="ref-1"
    )
    assert req.params == {"limit": 5}


def test_graph_query_federated_accepts_string_params() -> None:
    req = schemas.GraphQueryFederatedRequest(query="MATCH (n) RETURN n", params="{}")
    assert req.params == "{}"


def test_graph_query_federated_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        schemas.GraphQueryFederatedRequest(query="MATCH (n) RETURN n", bogus_field="x")


def test_evidence_bundle_envelope_accepts_a_real_bundle() -> None:
    bundle = EvidenceBundle(answer_candidate="42 nodes matched.")
    env = schemas.EvidenceBundleEnvelope(status="success", result=bundle)
    assert env.result.answer_candidate == "42 nodes matched."


def test_evidence_bundle_envelope_rejects_bad_status() -> None:
    with pytest.raises(ValidationError):
        schemas.EvidenceBundleEnvelope(status="failed", result=EvidenceBundle())


# ══════════════════════════════════════════════════════════════════
# graph_code / graph_research / graph_evaluate / graph_explain / graph_observe
# ══════════════════════════════════════════════════════════════════


def test_graph_code_request_valid() -> None:
    req = schemas.GraphCodeRequest(
        action="call_graph", node_id="Code:foo", target="callers"
    )
    assert req.action == "call_graph"


def test_graph_code_request_rejects_unknown_action() -> None:
    with pytest.raises(ValidationError):
        schemas.GraphCodeRequest(action="not_a_real_action")


def test_graph_code_request_rejects_unknown_field() -> None:
    """Proves the factory's blind `**body` splat is strict in practice — the
    target tool has no `**kwargs`, so `_execute_tool` 400s an unknown field."""
    with pytest.raises(ValidationError):
        schemas.GraphCodeRequest(action="code_context", connection="default")


def test_graph_research_request_valid() -> None:
    req = schemas.GraphResearchRequest(action="night_shift", target="/vault")
    assert req.action == "night_shift"


def test_graph_research_request_rejects_unknown_action() -> None:
    with pytest.raises(ValidationError):
        schemas.GraphResearchRequest(action="does_not_exist")


def test_graph_evaluate_request_valid() -> None:
    req = schemas.GraphEvaluateRequest(action="forecast", query="revenue-q3")
    assert req.action == "forecast"


def test_graph_evaluate_request_rejects_unknown_action() -> None:
    with pytest.raises(ValidationError):
        schemas.GraphEvaluateRequest(action="quant_bogus")


def test_graph_explain_request_valid() -> None:
    req = schemas.GraphExplainRequest(
        action="explain", target="ops:why", query="why did it fail"
    )
    assert req.target == "ops:why"


def test_graph_explain_request_rejects_unknown_action() -> None:
    with pytest.raises(ValidationError):
        schemas.GraphExplainRequest(action="summarize")


def test_graph_observe_request_valid() -> None:
    req = schemas.GraphObserveRequest(
        action="trace_rootcause", query="graph_orchestrate", top_k=5
    )
    assert req.top_k == 5


def test_graph_observe_request_rejects_unknown_action() -> None:
    with pytest.raises(ValidationError):
        schemas.GraphObserveRequest(action="not_an_action")


def test_graph_observe_response_accepts_dict_result() -> None:
    resp = schemas.GraphObserveResponse(status="success", result={"rows": []})
    assert resp.result == {"rows": []}


def test_graph_observe_response_accepts_string_result() -> None:
    """graph_observe's declared return type is `str`; a non-JSON message
    (e.g. a permission-denial or unknown-action string) passes through
    `safe_json_load` unparsed."""
    resp = schemas.GraphObserveResponse(
        status="success", result="Error: Unknown observe action 'x'"
    )
    assert isinstance(resp.result, str)


# ══════════════════════════════════════════════════════════════════
# graph_orchestrate
# ══════════════════════════════════════════════════════════════════


def test_graph_orchestrate_request_valid_minimal() -> None:
    req = schemas.GraphOrchestrateRequest(task="fix the failing test")
    assert req.execution_mode == "auto"


def test_graph_orchestrate_request_valid_full() -> None:
    req = schemas.GraphOrchestrateRequest(
        task="run the eval suite",
        agent_name="",
        skill_name="my-skill",
        tool_server="my-server",
        execution_mode="pydantic_graph",
        max_steps=10,
        allowed_tools=["graph_query", "graph_code"],
        required_tools=["graph_query"],
        response_format="json",
        grounding="best_effort",
    )
    assert req.allowed_tools == ["graph_query", "graph_code"]


def test_graph_orchestrate_request_rejects_bad_execution_mode() -> None:
    with pytest.raises(ValidationError):
        schemas.GraphOrchestrateRequest(task="x", execution_mode="nonsense")


def test_graph_orchestrate_request_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        schemas.GraphOrchestrateRequest(task="x", agent="not-a-real-field")


def test_graph_orchestrate_response_free_form_result() -> None:
    resp = schemas.GraphOrchestrateResponse(
        status="success", result={"output": "done", "mermaid": None, "run_id": "abc"}
    )
    assert resp.result["output"] == "done"


# ══════════════════════════════════════════════════════════════════
# graph_configure + doctor/secret/vault-sync
# ══════════════════════════════════════════════════════════════════


def test_graph_configure_request_valid() -> None:
    req = schemas.GraphConfigureRequest(action="doctor")
    assert req.action == "doctor"


def test_graph_configure_request_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        schemas.GraphConfigureRequest(action="doctor", extra_thing="nope")


def test_graph_configure_response_permissive_result() -> None:
    resp = schemas.GraphConfigureResponse(status="success", result={"anything": True})
    assert resp.result == {"anything": True}


def test_graph_configure_secret_request_valid() -> None:
    req = schemas.GraphConfigureSecretRequest(
        config_key="MY_API_KEY", config_value="s3cr3t"
    )
    assert req.config_key == "MY_API_KEY"


def test_graph_configure_secret_request_allows_unknown_field() -> None:
    """PROVEN permissive: the handler reads only config_key/config_value off
    the body; anything else is ignored, not rejected."""
    req = schemas.GraphConfigureSecretRequest(
        config_key="K", config_value="V", irrelevant="ignored"
    )
    assert req.irrelevant == "ignored"


def test_set_secret_result_never_carries_the_value() -> None:
    result = schemas.SetSecretResult(status="success", action="set_secret", stored=True)
    assert "config_value" not in type(result).model_fields
    assert set(type(result).model_fields) == {"status", "action", "stored"}


def test_set_secret_result_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        schemas.SetSecretResult(
            status="success", action="set_secret", stored=True, value="leak"
        )


def test_graph_configure_vault_sync_request_valid() -> None:
    req = schemas.GraphConfigureVaultSyncRequest(
        config_key="my-service",
        config_value='{"env_keys": ["API_TOKEN"], "overwrite": false}',
    )
    assert req.config_key == "my-service"


def test_vault_sync_result_never_carries_raw_values() -> None:
    result = schemas.VaultSyncResult(
        status="success",
        action="vault_sync",
        service="my-service",
        refs={"API_TOKEN": "vault://my-service/API_TOKEN"},
        present=[],
        written=["API_TOKEN"],
        missing=[],
    )
    assert set(type(result).model_fields) == {
        "status",
        "action",
        "service",
        "refs",
        "present",
        "written",
        "missing",
    }
    assert result.refs["API_TOKEN"].startswith("vault://")


def test_hook_doctor_entry_valid() -> None:
    entry = schemas.HookDoctorEntry(
        name="claude-code",
        path="/home/user/.claude.json",
        status="healthy",
        size_bytes=128,
    )
    assert entry.status == "healthy"


def test_hook_doctor_entry_rejects_bad_status() -> None:
    with pytest.raises(ValidationError):
        schemas.HookDoctorEntry(name="x", path=None, status="bogus")


def test_graph_configure_doctor_response_valid() -> None:
    resp = schemas.GraphConfigureDoctorResponse(
        status="success",
        result={
            "claude-code": {
                "name": "Claude Code",
                "path": "/x/.claude.json",
                "status": "healthy",
                "size_bytes": 42,
            },
            "agent-terminal-ui": {
                "name": "agent-terminal-ui",
                "path": None,
                "status": "integrated",
            },
        },
    )
    assert resp.result["agent-terminal-ui"].status == "integrated"


# ══════════════════════════════════════════════════════════════════
# /tools — the two-handler duality
# ══════════════════════════════════════════════════════════════════


def test_tools_catalog_response_valid() -> None:
    resp = schemas.ToolsCatalogResponse(
        mcp_tools=[
            {
                "name": "graph-os",
                "type": "MCP Server",
                "launch_mode": "remote",
                "command": "",
                "args": [],
                "status": "active",
                "enabled": True,
            }
        ],
        builtin_tools=[
            {
                "name": "web_search",
                "type": "Built-in Tool",
                "file_path": "tool://web_search",
                "status": "enabled",
                "enabled": True,
            }
        ],
        skills=[
            {"id": "s1", "name": "some-skill", "type": "Agent Skill", "enabled": True}
        ],
        skill_graphs=[],
        skill_workflows=[],
        section_status={"mcp_tools": "ok", "builtin_tools": "ok"},
    )
    assert resp.mcp_tools[0].name == "graph-os"


def test_tools_catalog_response_rejects_bad_launch_mode() -> None:
    with pytest.raises(ValidationError):
        schemas.McpServerToolInfo(
            name="x",
            type="MCP Server",
            launch_mode="carrier_pigeon",
            command="",
            args=[],
            status="active",
            enabled=True,
        )


def test_skill_catalog_entry_allows_extra_frontmatter_fields() -> None:
    entry = schemas.SkillCatalogEntry(
        id="s1",
        name="foo",
        type="Agent Skill",
        enabled=True,
        domain="ops",
        tags=["a", "b"],
    )
    assert entry.domain == "ops"


def test_tool_catalog_item_pins_the_flat_list_contract_agent_terminal_ui_depends_on() -> (
    None
):
    """Pins the exact shape `agent-terminal-ui/agent_terminal_ui/client.py:534`'s
    `AgentClient.list_tools()` depends on: a bare `list[dict]`, each with
    id/name/description/source_name/type — served by
    `agent_utilities/server/routers/core.py::list_tools` (a DIFFERENT
    handler than the one `_mount_rest_routes` binds to the same leaf path;
    see the module docstring above `McpServerToolInfo`)."""
    payload = [
        {
            "id": "tool-1",
            "name": "graph_query",
            "description": "Execute a read-only Cypher query.",
            "source_name": "graph-os",
            "type": "tool",
        },
        {
            "id": "skill-1",
            "name": "some-skill",
            "description": "Does a thing.",
            "source_name": "ops",
            "type": "skill",
        },
    ]
    items = [schemas.ToolCatalogItem(**row) for row in payload]
    assert isinstance(items, list)
    assert items[0].type == "tool"
    assert items[1].type == "skill"
    # every item is a flat dict, not an envelope
    for item in items:
        assert set(type(item).model_fields) == {
            "id",
            "name",
            "description",
            "source_name",
            "type",
        }


def test_tool_catalog_item_rejects_bad_type() -> None:
    with pytest.raises(ValidationError):
        schemas.ToolCatalogItem(id="x", name="y", type="widget")


# ══════════════════════════════════════════════════════════════════
# /goals, /goals/{id}/cancel, /goals/{id}/iterations
# ══════════════════════════════════════════════════════════════════


def test_goal_record_from_in_memory_producer_has_created_at() -> None:
    rec = schemas.GoalRecord(
        goal_id="g1",
        session_id="s1",
        status="running",
        objective="ship the feature",
        owner_host="worker-1",
        created_at=1_700_000_000.0,
        iterations=[],
        total_iterations=2,
        total_duration_ms=1200,
        total_tool_calls=5,
        summary="in progress",
        error="",
    )
    assert rec.created_at == 1_700_000_000.0


def test_goal_record_from_kg_fallback_producer_omits_created_at() -> None:
    """The `_goal_row_to_entry` KG-derived shape never populates `created_at`
    — this must validate without it."""
    rec = schemas.GoalRecord(
        goal_id="g2",
        session_id="s2",
        status="submitted",
        objective="ship the other feature",
        owner_host="",
        iterations=[],
        total_iterations=0,
        total_duration_ms=0,
        total_tool_calls=0,
        summary="",
        error="",
    )
    assert rec.created_at is None


def test_goal_record_rejects_missing_required_field() -> None:
    with pytest.raises(ValidationError):
        schemas.GoalRecord(session_id="s3", status="running", objective="x")


def test_create_goal_request_requires_objective() -> None:
    with pytest.raises(ValidationError):
        schemas.CreateGoalRequest()


def test_create_goal_request_valid() -> None:
    req = schemas.CreateGoalRequest(
        objective="Green up the failing suite",
        max_iterations=5,
        validation_cmd="pytest -q",
        constraints=["no force-push"],
    )
    assert req.max_iterations == 5


def test_create_goal_request_allows_unknown_field() -> None:
    req = schemas.CreateGoalRequest(objective="x", unexpected="ignored")
    assert req.unexpected == "ignored"


def test_create_goal_response_valid() -> None:
    resp = schemas.CreateGoalResponse(
        status="success",
        goal_id="g1",
        session_id="s1",
        objective="ship it",
        validation_cmd="",
        dispatch={"queued": True},
    )
    assert resp.goal_id == "g1"


def test_list_goals_response_valid() -> None:
    resp = schemas.ListGoalsResponse(
        goals=[
            schemas.GoalRecord(
                goal_id="g1",
                session_id="s1",
                status="running",
                objective="x",
                owner_host="",
                iterations=[],
                total_iterations=0,
                total_duration_ms=0,
                total_tool_calls=0,
                summary="",
                error="",
            )
        ]
    )
    assert len(resp.goals) == 1


def test_cancel_goal_response_valid() -> None:
    resp = schemas.CancelGoalResponse(
        status="success", message="Goal cancelled successfully."
    )
    assert resp.message


def test_cancel_goal_response_rejects_missing_message() -> None:
    with pytest.raises(ValidationError):
        schemas.CancelGoalResponse(status="success")


# ══════════════════════════════════════════════════════════════════
# /connector/run, /connector/sources
# ══════════════════════════════════════════════════════════════════


def test_connector_run_request_valid() -> None:
    req = schemas.ConnectorRunRequest(
        source_type="filesystem",
        config={"root": "/docs"},
        connector_id="fs-1",
        contextual=True,
        incremental=True,
    )
    assert req.source_type == "filesystem"


def test_connector_run_request_defaults_and_allows_unknown_field() -> None:
    req = schemas.ConnectorRunRequest(bogus="ignored")
    assert req.source_type == ""
    assert req.contextual is True
    assert req.bogus == "ignored"


def test_connector_run_result_valid_with_extra_details() -> None:
    result = schemas.ConnectorRunResult(
        status="success",
        error=None,
        nodes_created=3,
        edges_created=5,
        documents_ingested=2,
    )
    assert result.documents_ingested == 2


def test_connector_run_response_valid() -> None:
    resp = schemas.ConnectorRunResponse(
        status="success",
        result={
            "status": "success",
            "error": None,
            "nodes_created": 1,
            "edges_created": 1,
        },
    )
    assert resp.result.nodes_created == 1


def test_connector_sources_result_valid() -> None:
    result = schemas.ConnectorSourcesResult(connectors=["filesystem", "web", "rest"])
    assert "filesystem" in result.connectors


def test_connector_sources_result_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        schemas.ConnectorSourcesResult(connectors=[], extra="nope")


def test_connector_sources_response_valid() -> None:
    resp = schemas.ConnectorSourcesResponse(
        status="success", result={"connectors": ["filesystem"]}
    )
    assert resp.result.connectors == ["filesystem"]


# ══════════════════════════════════════════════════════════════════
# /sessions
# ══════════════════════════════════════════════════════════════════


def test_session_record_valid() -> None:
    rec = schemas.SessionRecord(
        id="sess-1",
        title="Goal: ship it",
        created_at=1_700_000_000.0,
        updated_at=1_700_000_100.0,
        model="gpt-4o",
        mode="ask",
        workspace="",
        turn_count=1,
        status="queued",
        background=True,
        needs_input=False,
        last_response_preview="Goal queued for dispatch...",
        goal_id="g1",
        metadata_json="{}",
        tenant_id="",
    )
    assert rec.background is True


def test_session_record_rejects_missing_required_field() -> None:
    with pytest.raises(ValidationError):
        schemas.SessionRecord(id="sess-1", created_at=1.0, updated_at=1.0)


def test_session_record_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        schemas.SessionRecord(
            id="sess-1",
            created_at=1.0,
            updated_at=1.0,
            background=False,
            needs_input=False,
            extra_column="nope",
        )


def test_sessions_list_query_defaults() -> None:
    q = schemas.SessionsListQuery()
    assert q.limit == 500
    assert q.offset == 0


def test_sessions_list_query_accepts_custom_values() -> None:
    q = schemas.SessionsListQuery(limit=50, offset=100)
    assert q.limit == 50
    assert q.offset == 100
