"""Contract tests for the privacy-safe bundled-skill runtime harness."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastmcp.exceptions import ToolError
from fastmcp.utilities.json_schema_type import json_schema_to_type
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

import agent_utilities.skills.runtime_validation as runtime_harness
from agent_utilities.security.persistence_privacy import persistence_reference
from agent_utilities.skills.runtime_validation import (
    CaseResult,
    SemanticOutput,
    TraceRecord,
    ValidationCase,
    ValidationChildToolError,
    _contract_instruction,
    _direct_case_minimum_authority_ttl,
    _direct_execution_prompt,
    _direct_semantic_output_type,
    _extract_semantic_payload,
    _ReadOnlyValidationMarkingStore,
    _renew_delegated_validation_session,
    _renew_direct_validation_session,
    _run_delegated_case,
    _skill_instruction_digest,
    _SkillValidationEvidenceSource,
    _validate_delegated_runtime_evidence,
    _validate_result_set,
    _validation_reasoning_effort,
    _verified_validation_session,
    _wait_for_run_completion,
    build_evidence,
    load_matrix,
    render_report,
    sign_and_verify_evidence,
    validate_semantic_output,
)
from agent_utilities.usage.privacy import normalize_run_id

_RUN_ID = "run:" + "a" * 32
_TENANT_ID = "tenant:synthetic"
_RUN_REF = normalize_run_id(_RUN_ID, tenant_id=_TENANT_ID)
_TRACE_NAME = f"graph_run:{_RUN_REF}"
_MODEL_REF = "pref_model_" + "b" * 64
_SKILL_REF = persistence_reference(
    "skill", "graph-query-and-explanation", namespace="execution-trace"
)
_SKILL_BODY_REF = "pref_skill_body_" + "c" * 64
_TRACE_EVIDENCE = {
    "run_ref": _RUN_REF,
    "model_ref": _MODEL_REF,
    "model_class": "economy",
    "skill_ref": _SKILL_REF,
    "skill_body_ref": _SKILL_BODY_REF,
}


def _delegated_case() -> ValidationCase:
    return ValidationCase(
        case_id="synthetic-delegated",
        skill="graph-query-and-explanation",
        mode="delegated",
        model_class="economy",
        task="Synthetic task.",
        expected_routes=("graph_orchestrate", "graph_query"),
        allowed_tools=("graph_query",),
        read_only=True,
    )


def _matrix_case(case_id: str) -> ValidationCase:
    _defaults, cases = load_matrix()
    return next(case for case in cases if case.case_id == case_id)


def _direct_semantic_payload(
    case: ValidationCase, *, selected_routes: list[str]
) -> dict[str, Any]:
    return {
        "skill": case.skill,
        "mode": "direct",
        "selected_routes": selected_routes,
        "read_only": True,
        "privacy_safe": True,
        "acceptance_summary": "Synthetic validation passed.",
    }


def _semantic_payload() -> dict[str, Any]:
    return {
        "skill": "graph-query-and-explanation",
        "mode": "delegated",
        "selected_routes": ["graph_orchestrate", "graph_query"],
        "read_only": True,
        "privacy_safe": True,
        "acceptance_summary": "Synthetic validation.",
    }


def test_contract_instruction_binds_exact_skill_identity() -> None:
    instruction = _contract_instruction(_delegated_case())

    assert "Set skill to 'graph-query-and-explanation'." in instruction
    assert "acceptance_summary to one plain sentence of at most 240 characters" in (
        instruction
    )


def test_direct_execution_prompt_places_closed_contract_after_original_task() -> None:
    case = _matrix_case("orchestration-direct")
    contract = _contract_instruction(case)

    prompt = _direct_execution_prompt(case)

    assert prompt == f"{case.task}\n\n{contract}"
    assert prompt.startswith(case.task)
    assert prompt.endswith(contract)
    assert prompt.rsplit("\n\n", maxsplit=1) == [case.task, contract]
    assert "only one JSON object" in prompt.rsplit("\n\n", maxsplit=1)[-1]


def test_direct_semantic_contract_uses_prompted_json_not_tool_output() -> None:
    from pydantic_ai import PromptedOutput

    output_type = _direct_semantic_output_type(_matrix_case("engine-direct"))

    assert isinstance(output_type, PromptedOutput)
    assert issubclass(output_type.outputs, SemanticOutput)
    assert output_type.template is not None
    assert "{schema}" in output_type.template
    assert "Do not wrap it in Markdown" in output_type.template


def test_direct_prompted_output_accepts_exact_case_route_set() -> None:
    case = _matrix_case("query-direct")
    output_type = _direct_semantic_output_type(case)

    output = output_type.outputs.model_validate(
        _direct_semantic_payload(
            case,
            selected_routes=list(reversed(case.expected_routes)),
        )
    )

    assert set(output.selected_routes) == set(case.expected_routes)


def test_direct_prompted_output_rejects_engine_route_expansion() -> None:
    case = _matrix_case("engine-direct")
    output_type = _direct_semantic_output_type(case)
    expanded_routes = [*case.expected_routes, "engine_datascience"]

    with pytest.raises(ValidationError):
        output_type.outputs.model_validate(
            _direct_semantic_payload(case, selected_routes=expanded_routes)
        )


def test_direct_validation_evidence_is_bounded_and_privacy_safe() -> None:
    source = _SkillValidationEvidenceSource("graph-query-and-explanation")

    rows = source.search_hybrid("synthetic query", top_k=40)

    assert len(rows) == 1
    assert rows[0]["kind"] == "skill_instruction"
    assert rows[0]["source_refs"] == ["skill://graph-query-and-explanation"]
    assert rows[0]["id"].startswith("pref_skill_")
    assert "graph-query-and-explanation" not in rows[0]["id"]
    assert source.search_hybrid("synthetic query", top_k=0) == []
    assert source.retrieve_epistemic_view("synthetic query") == {}


def test_direct_validation_marking_store_is_read_only() -> None:
    store = _ReadOnlyValidationMarkingStore()

    assert store.execute("MATCH (m) RETURN m") == []
    with pytest.raises(
        PermissionError, match="skill_validation_marking_store_is_read_only"
    ):
        store.execute("MERGE (m {id: $id})", {"id": "synthetic"})


def test_direct_authority_restores_exact_state_on_cancellation() -> None:
    import agent_utilities.core.contextual_model as contextual_model
    import agent_utilities.knowledge_graph.ontology.permissioning as permissioning
    from agent_utilities.core.contextual_model import use_context_compiler_engine
    from agent_utilities.knowledge_graph.core.company_brain_runtime import (
        get_company_brain,
    )
    from agent_utilities.knowledge_graph.ontology.permissioning import (
        use_marking_authority,
    )
    from agent_utilities.models.company_brain import (
        ActorType,
        DataClassification,
        NodeACL,
    )

    sentinel_engine = object()
    sentinel_store = object()
    source = _SkillValidationEvidenceSource("graph-query-and-explanation")
    sentinel_acl = NodeACL(
        node_id=source.node_id,
        classification=DataClassification.CONFIDENTIAL,
        read_roles=["kg:admin"],
        data_owner="sentinel-authority",
        data_owner_type=ActorType.SYSTEM,
    )
    permissions = get_company_brain().permissions

    with (
        use_context_compiler_engine(sentinel_engine),
        use_marking_authority(sentinel_store),
        permissions.use_acl(sentinel_acl),
    ):
        permissioning.MARKING_REGISTRY[("tenant:synthetic", "node:synthetic")] = {
            "controlled"
        }
        with pytest.raises(asyncio.CancelledError):
            with runtime_harness._direct_evidence_authority(
                "graph-query-and-explanation"
            ):
                raise asyncio.CancelledError

        assert contextual_model._compiler_engine is sentinel_engine
        assert permissioning._marking_store is sentinel_store
        assert permissioning._marking_store_resolved is True
        assert permissioning._markings_hydrated is False
        assert permissioning.MARKING_REGISTRY == {
            ("tenant:synthetic", "node:synthetic"): {"controlled"}
        }
        assert permissions.get_acl(source.node_id) is sentinel_acl


@pytest.mark.asyncio
async def test_validation_session_is_minted_from_verified_bearer(monkeypatch) -> None:
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security import request_identity
    from agent_utilities.security.brain_context import ActorContext

    seen: list[str] = []
    ttl_checks: list[int] = []

    async def verify(token: str) -> ActorContext:
        seen.append(token)
        return ActorContext(
            actor_id="subject:opaque:synthetic",
            actor_type=ActorType.SYSTEM,
            roles=("kg:admin",),
            tenant_id=_TENANT_ID,
            authenticated=True,
        )

    class Session:
        def __init__(self, actor: ActorContext) -> None:
            self.actor = actor
            self.tenant = actor.tenant_id

        def engine_verified_context(self) -> dict[str, str]:
            return {"principal": self.actor.actor_id}

        def ensure_authority_current(self, *, minimum_ttl_seconds: int) -> None:
            ttl_checks.append(minimum_ttl_seconds)

    monkeypatch.setattr(request_identity, "actor_from_bearer_token", verify)
    monkeypatch.setattr(request_identity, "mint_graph_session", Session)

    session = await _verified_validation_session(
        {"Authorization": "Bearer synthetic-token"}, minimum_ttl_seconds=47
    )

    assert seen == ["synthetic-token"]
    assert ttl_checks == [47]
    assert session.actor.authenticated is True
    assert session.tenant == _TENANT_ID


def test_direct_authority_ttl_covers_the_complete_bounded_case() -> None:
    required = _direct_case_minimum_authority_ttl(
        case_timeout=120.0, trace_timeout=30.0
    )

    assert required == 215


@pytest.mark.asyncio
async def test_direct_session_forces_refresh_when_cached_lease_is_too_short(
    monkeypatch,
) -> None:
    from agent_utilities.knowledge_graph.core.session import SessionExpiredError
    from agent_utilities.mcp import client_credentials

    headers_seen: list[str] = []
    force_values: list[bool] = []
    current_token = ["cached-token"]
    authority = {"principal": "subject:opaque:synthetic", "tenant": _TENANT_ID}
    renewed = SimpleNamespace(
        tenant=_TENANT_ID,
        actor=object(),
        engine_verified_context=lambda: authority,
    )

    def child_header(_existing):
        return {"Authorization": f"Bearer {current_token[0]}"}

    class Provider:
        def get_token(self, *, force: bool = False) -> str:
            force_values.append(force)
            assert force is True
            current_token[0] = "renewed-token"
            return current_token[0]

    async def verify(headers, *, minimum_ttl_seconds):
        assert minimum_ttl_seconds == 215
        authorization = headers["Authorization"]
        headers_seen.append(authorization)
        if authorization.endswith("cached-token"):
            raise SessionExpiredError("synthetic")
        return renewed

    monkeypatch.setattr(client_credentials, "child_auth_header", child_header)
    monkeypatch.setattr(client_credentials, "get_provider", lambda: Provider())
    monkeypatch.setattr(runtime_harness, "_verified_validation_session", verify)

    session = await _renew_direct_validation_session(
        expected_authority=authority, minimum_ttl_seconds=215
    )

    assert session is renewed
    assert headers_seen == ["Bearer cached-token", "Bearer renewed-token"]
    assert force_values == [True]


@pytest.mark.asyncio
async def test_direct_session_rejects_any_authority_change(monkeypatch) -> None:
    from agent_utilities.mcp import client_credentials

    expected = {
        "principal": "subject:opaque:synthetic",
        "roles": ["kg:read"],
        "tenant": _TENANT_ID,
    }
    changed = SimpleNamespace(
        tenant=_TENANT_ID,
        actor=object(),
        engine_verified_context=lambda: {**expected, "roles": ["kg:admin"]},
    )

    monkeypatch.setattr(
        client_credentials,
        "child_auth_header",
        lambda _existing: {"Authorization": "Bearer current-token"},
    )

    async def verify(_headers, *, minimum_ttl_seconds):
        assert minimum_ttl_seconds == 60
        return changed

    monkeypatch.setattr(runtime_harness, "_verified_validation_session", verify)

    with pytest.raises(RuntimeError, match="direct_identity_authority_changed"):
        await _renew_direct_validation_session(
            expected_authority=expected, minimum_ttl_seconds=60
        )


@pytest.mark.asyncio
async def test_delegated_session_always_force_refreshes_and_revalidates(
    monkeypatch,
) -> None:
    from agent_utilities.mcp import client_credentials

    force_values: list[bool] = []
    current_token = ["cached-token"]
    authority = {"principal": "subject:opaque:synthetic", "tenant": _TENANT_ID}
    renewed = SimpleNamespace(
        tenant=_TENANT_ID,
        actor=object(),
        engine_verified_context=lambda: authority,
    )

    class Provider:
        def get_token(self, *, force: bool = False) -> str:
            force_values.append(force)
            current_token[0] = "renewed-token"
            return current_token[0]

    monkeypatch.setattr(client_credentials, "get_provider", lambda: Provider())
    monkeypatch.setattr(
        client_credentials,
        "child_auth_header",
        lambda _existing: {"Authorization": f"Bearer {current_token[0]}"},
    )

    async def verify(headers, *, minimum_ttl_seconds):
        assert headers == {"Authorization": "Bearer renewed-token"}
        assert minimum_ttl_seconds == 215
        return renewed

    monkeypatch.setattr(runtime_harness, "_verified_validation_session", verify)

    session = await _renew_delegated_validation_session(
        expected_authority=authority, minimum_ttl_seconds=215
    )

    assert session is renewed
    assert force_values == [True]


@pytest.mark.asyncio
async def test_delegated_session_rejects_changed_authority(monkeypatch) -> None:
    from agent_utilities.mcp import client_credentials

    expected = {
        "principal": "subject:opaque:synthetic",
        "roles": ["kg:read"],
        "tenant": _TENANT_ID,
    }
    changed = SimpleNamespace(
        engine_verified_context=lambda: {**expected, "roles": ["kg:admin"]}
    )

    class Provider:
        def get_token(self, *, force: bool = False) -> str:
            assert force is True
            return "renewed-token"

    monkeypatch.setattr(client_credentials, "get_provider", lambda: Provider())
    monkeypatch.setattr(
        client_credentials,
        "child_auth_header",
        lambda _existing: {"Authorization": "Bearer renewed-token"},
    )

    async def verify(_headers, *, minimum_ttl_seconds):
        assert minimum_ttl_seconds == 60
        return changed

    monkeypatch.setattr(runtime_harness, "_verified_validation_session", verify)

    with pytest.raises(RuntimeError, match="delegated_identity_authority_changed"):
        await _renew_delegated_validation_session(
            expected_authority=expected, minimum_ttl_seconds=60
        )


@pytest.mark.asyncio
async def test_mcp_wire_error_is_never_decoded_as_success() -> None:
    class Client:
        async def call_tool(self, _name, _arguments):
            return SimpleNamespace(isError=True, content=[])

    with pytest.raises(ValidationChildToolError, match="mcp_tool_error"):
        await runtime_harness._call_tool(Client(), "synthetic", {}, 1.0)


@pytest.mark.asyncio
async def test_bounded_sdk_call_preserves_verified_tenant_context() -> None:
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import (
        ActorContext,
        current_actor,
        use_actor,
    )

    actor = ActorContext(
        actor_id="subject:opaque:synthetic",
        actor_type=ActorType.SYSTEM,
        roles=("kg:admin",),
        tenant_id=_TENANT_ID,
        authenticated=True,
    )
    with use_actor(actor):
        observed_tenant = await runtime_harness._bounded_sync_call(
            lambda: current_actor().tenant_id,
            1.0,
        )

    assert observed_tenant == _TENANT_ID
    assert runtime_harness._expected_trace_name(_RUN_ID, observed_tenant) == (
        _TRACE_NAME
    )


@pytest.mark.asyncio
async def test_bounded_sdk_call_poisoned_after_cancellation(monkeypatch) -> None:
    import threading

    active = threading.Lock()
    poisoned = threading.Event()
    release_worker = threading.Event()
    monkeypatch.setattr(runtime_harness, "_SYNC_CALL_ACTIVE", active)
    monkeypatch.setattr(runtime_harness, "_SYNC_CALL_POISONED", poisoned)

    task = asyncio.create_task(
        runtime_harness._bounded_sync_call(release_worker.wait, 5.0)
    )
    await asyncio.sleep(0)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert poisoned.is_set()
        with pytest.raises(RuntimeError, match="blocking_sdk_worker_abandoned"):
            await runtime_harness._bounded_sync_call(lambda: True, 1.0)
    finally:
        release_worker.set()
    for _ in range(100):
        if active.acquire(blocking=False):
            active.release()
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("cancelled SDK worker did not leave its guarded section")


@pytest.mark.parametrize(
    "payload",
    [
        "x" * (runtime_harness._MAX_TOOL_PAYLOAD + 1),
        [0] * runtime_harness._MAX_TOOL_ITEMS,
    ],
)
def test_structured_mcp_payload_budget_rejects_oversized_values(payload) -> None:
    result = SimpleNamespace(data=payload, structured_content=None, content=[])

    with pytest.raises(ValueError, match="payload_too_(large|many_items)"):
        runtime_harness._decode_tool_result(result)


@pytest.mark.parametrize("attribute", ["data", "structured_content"])
def test_structured_mcp_json_string_is_decoded_within_payload_bounds(attribute) -> None:
    result = SimpleNamespace(data=None, structured_content=None, content=[])
    setattr(result, attribute, '{"status":"completed"}')

    assert runtime_harness._decode_tool_result(result) == {"status": "completed"}


def test_structured_mcp_typed_payload_is_dumped_before_bounds_validation() -> None:
    class TypedPayload(BaseModel):
        status: str
        rows: list[dict[str, int]]

    payload = TypedPayload(status="completed", rows=[{"matched": 1}])
    result = SimpleNamespace(data=payload, structured_content=None, content=[])

    assert runtime_harness._decode_tool_result(result) == {
        "status": "completed",
        "rows": [{"matched": 1}],
    }


def test_structured_mcp_arbitrary_object_remains_rejected() -> None:
    result = SimpleNamespace(data=object(), structured_content=None, content=[])

    with pytest.raises(TypeError, match="payload_type_invalid"):
        runtime_harness._decode_tool_result(result)


def test_structured_mcp_generated_dataclass_is_normalized_after_bounds() -> None:
    payload_type = json_schema_to_type(
        {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "row": {
                    "type": "object",
                    "properties": {"matched": {"type": "integer"}},
                    "required": ["matched"],
                },
            },
            "required": ["status", "row"],
        }
    )
    payload = payload_type(status="completed", row={"matched": 1})
    result = SimpleNamespace(
        data={"result": payload}, structured_content=None, content=[]
    )

    assert runtime_harness._decode_tool_result(result) == {
        "result": {"status": "completed", "row": {"matched": 1}}
    }


def test_structured_mcp_payload_budget_rejects_depth_and_cycles() -> None:
    deep: list[Any] = []
    cursor = deep
    for _ in range(runtime_harness._MAX_TOOL_DEPTH + 2):
        child: list[Any] = []
        cursor.append(child)
        cursor = child
    cyclic: list[Any] = []
    cyclic.append(cyclic)

    with pytest.raises(ValueError, match="payload_too_deep"):
        runtime_harness._decode_tool_result(
            SimpleNamespace(data=deep, structured_content=None, content=[])
        )
    with pytest.raises(ValueError, match="payload_cycle"):
        runtime_harness._decode_tool_result(
            SimpleNamespace(data=cyclic, structured_content=None, content=[])
        )


def test_text_chunk_budget_is_enforced_before_join() -> None:
    content = [
        SimpleNamespace(text="x" * (runtime_harness._MAX_TOOL_PAYLOAD // 2 + 1)),
        SimpleNamespace(text="x" * (runtime_harness._MAX_TOOL_PAYLOAD // 2 + 1)),
    ]

    with pytest.raises(ValueError, match="payload_too_large"):
        runtime_harness._decode_tool_result(
            SimpleNamespace(data=None, structured_content=None, content=content)
        )


@pytest.mark.asyncio
async def test_langfuse_posture_requires_metadata_only_child(monkeypatch) -> None:
    async def safe_call(*_args, **_kwargs):
        return {"content_capture_enabled": False, "metadata_only": True}

    monkeypatch.setattr(runtime_harness, "_call_tool", safe_call)
    await runtime_harness._verify_langfuse_posture(object(), "langfuse", 1.0)

    async def unsafe_call(*_args, **_kwargs):
        return {"content_capture_enabled": True, "metadata_only": False}

    monkeypatch.setattr(runtime_harness, "_call_tool", unsafe_call)
    with pytest.raises(RuntimeError, match="langfuse_content_posture_invalid"):
        await runtime_harness._verify_langfuse_posture(object(), "langfuse", 1.0)


def test_expected_trace_name_is_tenant_qualified_and_rejects_raw_name() -> None:
    assert runtime_harness._expected_trace_name(_RUN_ID, _TENANT_ID) == _TRACE_NAME
    assert runtime_harness._expected_trace_name(_RUN_ID, "tenant:other") != _TRACE_NAME
    assert _TRACE_NAME != f"graph_run:{_RUN_ID}"


@pytest.mark.asyncio
async def test_trace_snapshot_filters_exact_name_and_metadata_at_provider() -> None:
    calls: list[dict[str, object]] = []

    class Client:
        async def call_tool(self, _name, arguments):
            calls.append(arguments)
            return SimpleNamespace(
                isError=False,
                structured_content={
                    "data": [
                        {
                            "id": "trace-expected",
                            "name": _TRACE_NAME,
                            "metadata": _TRACE_EVIDENCE,
                        }
                    ]
                },
                content=[],
            )

    snapshot = await runtime_harness._trace_snapshot(
        Client(),
        "langfuse_observability",
        5.0,
        from_timestamp="2030-01-01T00:00:00Z",
        expected_name=_TRACE_NAME,
    )

    assert snapshot == {"trace-expected": TraceRecord(_TRACE_NAME, _TRACE_EVIDENCE)}
    assert calls == [
        {
            "action": "trace_list",
            "page": 1,
            "limit": 20,
            "order_by": "timestamp.desc",
            "fields": "core,basic,metadata",
            "from_timestamp": "2030-01-01T00:00:00Z",
            "name": _TRACE_NAME,
        }
    ]


@pytest.mark.asyncio
async def test_trace_wait_requires_exact_cross_bound_evidence(monkeypatch) -> None:
    async def snapshot(*_args, **kwargs):
        assert kwargs["expected_name"] == _TRACE_NAME
        return {"trace-expected": TraceRecord(_TRACE_NAME, _TRACE_EVIDENCE)}

    monkeypatch.setattr(runtime_harness, "_trace_snapshot", snapshot)

    trace_id, linkage = await runtime_harness._wait_for_expected_trace(
        object(),
        "langfuse_observability",
        "2030-01-01T00:00:00Z",
        _TRACE_NAME,
        _TRACE_EVIDENCE,
        1.0,
    )

    assert (trace_id, linkage) == ("trace-expected", "run-evidence")


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [ToolError, ValidationChildToolError])
async def test_trace_wait_retries_bounded_child_tool_failure(
    monkeypatch, error_type
) -> None:
    attempts = 0

    async def snapshot(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise error_type("delegated_child_tool_failed")
        return {"trace-expected": TraceRecord(_TRACE_NAME, _TRACE_EVIDENCE)}

    monkeypatch.setattr(runtime_harness, "_trace_snapshot", snapshot)

    trace_id, linkage = await runtime_harness._wait_for_expected_trace(
        object(),
        "langfuse_observability",
        "2030-01-01T00:00:00Z",
        _TRACE_NAME,
        _TRACE_EVIDENCE,
        2.0,
    )

    assert attempts == 2
    assert (trace_id, linkage) == ("trace-expected", "run-evidence")


@pytest.mark.asyncio
async def test_trace_wait_keeps_persistent_child_tool_failure_terminal(
    monkeypatch,
) -> None:
    attempts = 0

    async def snapshot(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise ToolError("delegated_child_tool_failed")

    monkeypatch.setattr(runtime_harness, "_trace_snapshot", snapshot)

    with pytest.raises(ToolError, match="delegated_child_tool_failed"):
        await runtime_harness._wait_for_expected_trace(
            object(),
            "langfuse_observability",
            "2030-01-01T00:00:00Z",
            _TRACE_NAME,
            _TRACE_EVIDENCE,
            2.0,
        )

    assert attempts == runtime_harness._TRACE_TOOL_ERROR_RETRIES + 1


@pytest.mark.asyncio
async def test_trace_wait_rejects_wrong_model_and_duplicate_matches(
    monkeypatch,
) -> None:
    wrong = dict(_TRACE_EVIDENCE, model_ref="pref_model_" + "d" * 64)

    async def wrong_snapshot(*_args, **_kwargs):
        return {"trace-expected": TraceRecord(_TRACE_NAME, wrong)}

    monkeypatch.setattr(runtime_harness, "_trace_snapshot", wrong_snapshot)
    with pytest.raises(RuntimeError, match="trace_evidence_mismatch"):
        await runtime_harness._wait_for_expected_trace(
            object(),
            "langfuse_observability",
            "2030-01-01T00:00:00Z",
            _TRACE_NAME,
            _TRACE_EVIDENCE,
            1.0,
        )

    async def duplicate_snapshot(*_args, **_kwargs):
        record = TraceRecord(_TRACE_NAME, _TRACE_EVIDENCE)
        return {"trace-one": record, "trace-two": record}

    monkeypatch.setattr(runtime_harness, "_trace_snapshot", duplicate_snapshot)
    with pytest.raises(RuntimeError, match="trace_run_identifier_ambiguous"):
        await runtime_harness._wait_for_expected_trace(
            object(),
            "langfuse_observability",
            "2030-01-01T00:00:00Z",
            _TRACE_NAME,
            _TRACE_EVIDENCE,
            1.0,
        )


@pytest.mark.asyncio
async def test_parent_ingestion_proof_queries_one_exact_opaque_trace(
    monkeypatch,
) -> None:
    from agent_utilities.models.evidence_bundle import EvidenceBundle

    calls: list[tuple[str, dict[str, object]]] = []

    def governed_projection(
        node_ids: list[str], *, include_engine_evidence: bool = False
    ) -> dict[str, object]:
        empty = EvidenceBundle.from_engine_wire({"rows": []})
        if include_engine_evidence:
            empty.reasoning_trace.append(
                {"step": "knowledge_set_row", "object_id": "trace:opaque:evidence"}
            )
        return EvidenceBundle.from_payload(
            {
                "rows": [{"id": node_id, "name": _TRACE_NAME} for node_id in node_ids],
                "evidence_bundle": empty.model_dump(),
            },
            operation="graph_query",
        ).model_dump()

    async def fake_call(_client, name, arguments, _timeout):
        calls.append((name, arguments))
        return governed_projection(
            ["langfuse:trace:" + "a" * 32], include_engine_evidence=True
        )

    monkeypatch.setattr(runtime_harness, "_call_tool", fake_call)

    await runtime_harness._verify_parent_ingested_trace(object(), _TRACE_NAME, 1.0)

    assert calls[0][0] == "graph_query"
    assert calls[0][1]["scope"] == "local"
    assert calls[0][1]["params"] == '{"name":"' + _TRACE_NAME + '"}'
    assert calls[0][1]["cypher"].endswith("RETURN n.id AS id, n.name AS name LIMIT 2")

    async def missing_call(*_args, **_kwargs):
        return governed_projection([])

    monkeypatch.setattr(runtime_harness, "_call_tool", missing_call)
    monkeypatch.setattr(runtime_harness, "_PARENT_INGESTION_POLL_DELAY_SECONDS", 0.005)
    with pytest.raises(TimeoutError, match="trace_parent_ingestion_not_observed"):
        await runtime_harness._verify_parent_ingested_trace(object(), _TRACE_NAME, 0.02)

    async def duplicate_call(*_args, **_kwargs):
        return governed_projection(
            ["langfuse:trace:" + "a" * 32, "langfuse:trace:" + "b" * 32]
        )

    monkeypatch.setattr(runtime_harness, "_call_tool", duplicate_call)
    with pytest.raises(RuntimeError, match="trace_parent_ingestion_mismatch"):
        await runtime_harness._verify_parent_ingested_trace(object(), _TRACE_NAME, 1.0)

    async def ungoverned_aggregate(*_args, **_kwargs):
        return EvidenceBundle.from_payload(
            {"rows": [{"matched": 1}]}, operation="graph_query"
        ).model_dump()

    monkeypatch.setattr(runtime_harness, "_call_tool", ungoverned_aggregate)
    with pytest.raises(RuntimeError, match="trace_parent_ingestion_mismatch"):
        await runtime_harness._verify_parent_ingested_trace(object(), _TRACE_NAME, 1.0)

    async def duplicate_query_trace(*_args, **_kwargs):
        payload = governed_projection(["langfuse:trace:" + "a" * 32])
        payload["reasoning_trace"].append(payload["reasoning_trace"][0])
        return payload

    monkeypatch.setattr(runtime_harness, "_call_tool", duplicate_query_trace)
    with pytest.raises(RuntimeError, match="trace_parent_ingestion_mismatch"):
        await runtime_harness._verify_parent_ingested_trace(object(), _TRACE_NAME, 1.0)

    async def forged_claim(*_args, **_kwargs):
        return {"claims": [{"matched": 1}]}

    monkeypatch.setattr(runtime_harness, "_call_tool", forged_claim)
    with pytest.raises(RuntimeError, match="trace_parent_ingestion_mismatch"):
        await runtime_harness._verify_parent_ingested_trace(object(), _TRACE_NAME, 1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [ToolError, ValidationChildToolError])
async def test_parent_ingestion_retries_only_typed_transient_failures_and_zero(
    monkeypatch, error_type
) -> None:
    from agent_utilities.models.evidence_bundle import EvidenceBundle

    attempts = 0

    def projection(node_ids: list[str]) -> dict[str, object]:
        return EvidenceBundle.from_payload(
            {
                "rows": [{"id": node_id, "name": _TRACE_NAME} for node_id in node_ids],
                "evidence_bundle": EvidenceBundle.from_engine_wire(
                    {"rows": []}
                ).model_dump(),
            },
            operation="graph_query",
        ).model_dump()

    async def fake_call(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise error_type("delegated_child_tool_failed")
        if attempts == 2:
            return projection([])
        return projection(["langfuse:trace:" + "a" * 32])

    monkeypatch.setattr(runtime_harness, "_call_tool", fake_call)
    monkeypatch.setattr(runtime_harness, "_TRACE_TOOL_ERROR_RETRY_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(runtime_harness, "_PARENT_INGESTION_POLL_DELAY_SECONDS", 0.0)

    count = await runtime_harness._verify_parent_ingested_trace(
        object(), _TRACE_NAME, 1.0
    )

    assert count == 1
    assert attempts == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [ToolError, ValidationChildToolError])
async def test_parent_ingestion_keeps_persistent_typed_failure_terminal(
    monkeypatch, error_type
) -> None:
    attempts = 0

    async def fake_call(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise error_type("delegated_child_tool_failed")

    monkeypatch.setattr(runtime_harness, "_call_tool", fake_call)
    monkeypatch.setattr(runtime_harness, "_TRACE_TOOL_ERROR_RETRY_DELAY_SECONDS", 0.0)

    with pytest.raises(error_type, match="delegated_child_tool_failed"):
        await runtime_harness._verify_parent_ingested_trace(object(), _TRACE_NAME, 1.0)

    assert attempts == runtime_harness._TRACE_TOOL_ERROR_RETRIES + 1


@pytest.mark.asyncio
async def test_parent_ingestion_does_not_retry_untyped_failure(monkeypatch) -> None:
    attempts = 0

    async def fake_call(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("synthetic_untyped_failure")

    monkeypatch.setattr(runtime_harness, "_call_tool", fake_call)

    with pytest.raises(RuntimeError, match="synthetic_untyped_failure"):
        await runtime_harness._verify_parent_ingested_trace(object(), _TRACE_NAME, 1.0)

    assert attempts == 1


def test_report_publication_is_atomic_private_and_bounded(tmp_path) -> None:
    destination = tmp_path / "nested" / "matrix.md"

    runtime_harness.publish_report(destination, "controlled report\n")
    runtime_harness.publish_report(destination, "replacement report\n")

    assert destination.read_text(encoding="utf-8") == "replacement report\n"
    assert not list(destination.parent.glob(".*.tmp"))
    if os.name == "posix":
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink contract")
def test_report_publication_rejects_destination_symlink(tmp_path) -> None:
    target = tmp_path / "target.md"
    target.write_text("unchanged\n", encoding="utf-8")
    destination = tmp_path / "matrix.md"
    destination.symlink_to(target)

    with pytest.raises(RuntimeError, match="report_destination_symlink"):
        runtime_harness.publish_report(destination, "replacement\n")

    assert target.read_text(encoding="utf-8") == "unchanged\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink contract")
def test_report_publication_rejects_symlinked_directory(tmp_path) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    linked_directory = tmp_path / "linked"
    linked_directory.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(RuntimeError, match="report_directory_symlink"):
        runtime_harness.publish_report(linked_directory / "matrix.md", "report\n")

    assert not (real_directory / "matrix.md").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor traversal")
def test_report_publication_rejects_ancestor_swap(tmp_path, monkeypatch) -> None:
    ancestor = tmp_path / "trusted-report-parent-unique"
    ancestor.mkdir()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    moved = tmp_path / "moved"
    destination = ancestor / "matrix.md"
    original_open = runtime_harness.os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == ancestor.name and dir_fd is not None and not swapped:
            ancestor.rename(moved)
            ancestor.symlink_to(replacement, target_is_directory=True)
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(runtime_harness.os, "open", swapping_open)

    with pytest.raises(RuntimeError, match="report_directory_symlink"):
        runtime_harness.publish_report(destination, "report\n")

    assert swapped is True
    assert not (replacement / "matrix.md").exists()


def test_report_publication_reapplies_privacy_gate(tmp_path, monkeypatch) -> None:
    class UnsafeGuard:
        def sanitize_text(self, content):
            return content, SimpleNamespace(changed=True)

    monkeypatch.setattr(runtime_harness, "PersistencePrivacyGuard", UnsafeGuard)
    destination = tmp_path / "matrix.md"

    with pytest.raises(RuntimeError, match="report_privacy_gate_failed"):
        runtime_harness.publish_report(destination, "synthetic\n")

    assert not destination.exists()


def test_report_publication_fails_closed_without_posix_guarantees(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "matrix.md"
    monkeypatch.setattr(runtime_harness.os, "name", "nt")

    with pytest.raises(RuntimeError, match="report_platform_unsupported"):
        runtime_harness.publish_report(destination, "synthetic\n")

    assert not destination.exists()


def test_runtime_matrix_has_two_read_only_cases_per_skill() -> None:
    defaults, cases = load_matrix()
    assert defaults["sequential"] is True
    assert len(cases) == 20
    assert all(case.read_only for case in cases)
    assert all(f"skill://{case.skill}" in case.task for case in cases)
    assert all(not case.allowed_tools for case in cases if case.mode == "direct")
    assert all(
        case.allowed_tools and "graph_orchestrate" not in case.allowed_tools
        for case in cases
        if case.mode == "delegated"
    )


def test_economy_validation_omits_nonportable_reasoning_none() -> None:
    assert _validation_reasoning_effort("economy", delegated=False) is None
    assert _validation_reasoning_effort("economy", delegated=True) == ""


def test_semantic_contract_rejects_missing_routes() -> None:
    case = ValidationCase(
        case_id="synthetic-direct",
        skill="graph-query-and-explanation",
        mode="direct",
        model_class="economy",
        task="synthetic",
        expected_routes=("graph_query", "graph_search"),
        allowed_tools=(),
        read_only=True,
    )
    output = SemanticOutput(
        skill=case.skill,
        mode=case.mode,
        selected_routes=["graph_query"],
        read_only=True,
        privacy_safe=True,
        acceptance_summary="Bounded synthetic result.",
    )

    assert validate_semantic_output(case, output) == ["semantic_routes_incomplete"]


def test_semantic_contract_rejects_unexpected_routes() -> None:
    case = ValidationCase(
        case_id="synthetic-direct",
        skill="graph-query-and-explanation",
        mode="direct",
        model_class="economy",
        task="synthetic",
        expected_routes=("graph_query", "graph_search"),
        allowed_tools=(),
        read_only=True,
    )
    output = SemanticOutput(
        skill=case.skill,
        mode=case.mode,
        selected_routes=["graph_query", "graph_search", "graph_analyze"],
        read_only=True,
        privacy_safe=True,
        acceptance_summary="Bounded synthetic result.",
    )

    assert validate_semantic_output(case, output) == ["semantic_routes_unexpected"]


def test_source_tree_runtime_validation_wrapper_is_self_contained() -> None:
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "validate_prebundled_skills_runtime.py"),
            "--help",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--mode {direct,delegated,all}" in completed.stdout


def test_report_contains_only_controlled_evidence_fields() -> None:
    refs = {
        "run_ref": "pref_run_" + "1" * 64,
        "trace_ref": "pref_trace_" + "2" * 64,
        "model_ref": "pref_model_" + "3" * 64,
        "skill_ref": "pref_skill_" + "4" * 64,
        "skill_body_ref": "pref_skill_body_" + "5" * 64,
    }
    direct = CaseResult(
        case_id="synthetic-direct",
        skill="graph-query-and-explanation",
        mode="direct",
        model_class="economy",
        semantic="pass",
        model_selection="pass",
        skill_binding="pass",
        trace="pass",
        parent_ingestion="pass",
        trace_linkage="run-evidence",
        trace_name=f"graph_run:{refs['run_ref']}",
        langfuse_match_count=1,
        parent_kg_readback_count=1,
        selected_routes=("graph_query", "graph_search"),
        **refs,
    )
    delegated = CaseResult(
        case_id="synthetic-delegated",
        skill="graph-query-and-explanation",
        mode="delegated",
        model_class="standard",
        semantic="pass",
        model_selection="pass",
        skill_binding="pass",
        delegation="pass",
        trace="pass",
        parent_ingestion="pass",
        trace_linkage="run-evidence",
        trace_name=f"graph_run:{refs['run_ref']}",
        langfuse_match_count=1,
        parent_kg_readback_count=1,
        selected_routes=("graph_epistemic", "graph_orchestrate", "graph_query"),
        **refs,
    )

    report = render_report([direct, delegated], generated_at="2030-01-01T00:00:00Z")

    assert "Cases passed: 2/2" in report
    assert "Paired result" in report
    assert "configured model" in report
    assert "http://" not in report
    assert "https://" not in report
    assert "raw model output" in report.lower()


def _passing_runtime_results() -> list[CaseResult]:
    _defaults, cases = load_matrix()
    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        opaque = f"{index:064x}"
        results.append(
            CaseResult(
                case_id=case.case_id,
                skill=case.skill,
                mode=case.mode,
                model_class=case.model_class,
                model_selection="pass",
                skill_binding="pass",
                structural="pass",
                semantic="pass",
                delegation="pass" if case.mode == "delegated" else "not-applicable",
                trace="pass",
                parent_ingestion="pass",
                trace_linkage="run-evidence",
                selected_routes=case.expected_routes,
                run_ref="pref_run_" + opaque,
                trace_ref="pref_trace_" + opaque,
                model_ref="pref_model_" + opaque,
                skill_ref="pref_skill_" + opaque,
                skill_body_ref="pref_skill_body_" + opaque,
                trace_name="graph_run:pref_run_" + opaque,
                langfuse_match_count=1,
                parent_kg_readback_count=1,
            )
        )
    return results


@pytest.mark.parametrize(
    ("attribute", "invalid"),
    [
        ("error_codes", ["synthetic_error"]),
        ("trace_linkage", "none"),
        ("trace_name", ""),
        ("langfuse_match_count", 0),
        ("parent_kg_readback_count", 2),
        ("selected_routes", ()),
        ("run_ref", ""),
        ("trace_ref", "pref_trace_invalid"),
        ("model_ref", "pref_model_invalid"),
        ("skill_ref", "pref_skill_invalid"),
        ("skill_body_ref", "pref_skill_body_invalid"),
    ],
)
def test_case_pass_requires_complete_exact_evidence(attribute: str, invalid) -> None:
    result = _passing_runtime_results()[0]

    setattr(result, attribute, invalid)

    assert not result.passed


def test_result_set_requires_exact_catalog_and_unique_evidence() -> None:
    results = _passing_runtime_results()

    _validate_result_set(results, mode="all")
    _validate_result_set(
        [result for result in results if result.mode == "direct"], mode="direct"
    )

    duplicate = _passing_runtime_results()
    duplicate[1].trace_ref = duplicate[0].trace_ref
    with pytest.raises(RuntimeError, match="runtime_evidence_reference_collision"):
        _validate_result_set(duplicate, mode="all")

    with pytest.raises(RuntimeError, match="runtime_case_set_invalid"):
        _validate_result_set(results[:-1], mode="all")


def test_exact_all_mode_evidence_is_strict_schema_valid_and_content_free(
    monkeypatch,
) -> None:
    digest = "sha256:" + "1" * 64
    unsigned = build_evidence(
        _passing_runtime_results(),
        generated_at="2030-01-01T00:00:00Z",
        release_id="release-certification-v1",
        release_specification_digest=digest,
        promotion_evidence_digest="sha256:" + "2" * 64,
        graph_os_digest="sha256:" + "3" * 64,
        engine_digest="sha256:" + "4" * 64,
        runtime_config_digest="sha256:" + "2" * 64,
        runtime_profile_digest="sha256:" + "3" * 64,
        model_registry_digest="sha256:" + "5" * 64,
    )
    subject = runtime_harness._digest_bytes(runtime_harness._canonical_bytes(unsigned))
    key_id = "key:" + "4" * 64
    calls: list[str] = []

    def external(reference: str, payload: bytes) -> dict[str, Any]:
        calls.append(reference)
        if reference == "SYNTHETIC_SIGNER_COMMAND":
            assert payload == runtime_harness._canonical_bytes(unsigned)
            return {
                "algorithm": "ed25519",
                "keyId": key_id,
                "signature": "A" * 43,
                "subjectDigest": subject,
            }
        return {"verified": True, "subjectDigest": subject, "keyId": key_id}

    monkeypatch.setattr(runtime_harness, "_external_json", external)
    signed = sign_and_verify_evidence(
        unsigned,
        signer_reference="SYNTHETIC_SIGNER_COMMAND",
        verifier_reference="SYNTHETIC_VERIFIER_COMMAND",
    )
    schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "deploy/release/prebundled-skill-validation-evidence.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(signed)

    assert calls == ["SYNTHETIC_SIGNER_COMMAND", "SYNTHETIC_VERIFIER_COMMAND"]
    assert signed["catalog"]["skillCount"] == 10
    assert signed["catalog"]["testCaseCount"] == 20
    assert len(signed["cases"]) == 20
    assert all(case["langfuse"]["matchCount"] == 1 for case in signed["cases"])
    assert all(
        case["parentKnowledgeGraph"]["matchCount"] == 1 for case in signed["cases"]
    )
    serialized = json.dumps(signed, sort_keys=True)
    assert "http://" not in serialized
    assert "https://" not in serialized
    assert "/home/" not in serialized
    assert "/mnt/" not in serialized


def test_all_mode_requires_paired_markdown_and_json_evidence() -> None:
    with pytest.raises(SystemExit):
        runtime_harness._arguments(["--mode", "all"])

    digest = "sha256:" + "1" * 64
    args = runtime_harness._arguments(
        [
            "--mode",
            "all",
            "--report",
            "reports/matrix.md",
            "--evidence",
            "reports/matrix.json",
            "--release-id",
            "release-certification-v1",
            "--release-specification-digest",
            digest,
            "--promotion-evidence-digest",
            "sha256:" + "2" * 64,
            "--graph-os-digest",
            "sha256:" + "3" * 64,
            "--engine-digest",
            "sha256:" + "4" * 64,
            "--runtime-config-digest",
            "sha256:" + "2" * 64,
            "--runtime-profile-digest",
            "sha256:" + "3" * 64,
            "--model-registry-digest",
            "sha256:" + "5" * 64,
        ]
    )
    assert args.mode == "all"


@pytest.mark.parametrize(
    "wire_value",
    [
        lambda payload: payload,
        lambda payload: {"result": __import__("json").dumps(payload)},
        lambda payload: __import__("json").dumps(payload),
    ],
)
def test_delegation_accepts_only_current_128_bit_contract(wire_value) -> None:
    output, run_id = _extract_semantic_payload(
        wire_value({"output": _semantic_payload(), "run_id": _RUN_ID})
    )

    assert output.mode == "delegated"
    assert run_id == _RUN_ID


@pytest.mark.parametrize(
    "payload",
    [
        {"output": {}, "run_id": "run:0123abcd"},
        {"output": {}, "run_id": _RUN_ID, "retired_status": "ok"},
        {"result": {}, "run_id": _RUN_ID},
    ],
)
def test_delegation_rejects_short_or_removed_response_shapes(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _extract_semantic_payload(payload)


@pytest.mark.parametrize(
    "output",
    (
        '```json\n{"skill":"graph-query-and-explanation"}\n```',
        'Result: {"skill":"graph-query-and-explanation"}',
    ),
)
def test_delegation_rejects_fenced_or_prose_wrapped_json(output: str) -> None:
    with pytest.raises(ValueError, match="delegation_output_not_json"):
        _extract_semantic_payload({"output": output, "run_id": _RUN_ID})


@pytest.mark.asyncio
async def test_run_completion_polls_the_focused_job_surface(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    responses = iter(({"status": "running"}, {"status": "completed"}))

    async def fake_call(_client, name, arguments, _timeout):
        calls.append((name, arguments))
        return next(responses)

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(runtime_harness, "_call_tool", fake_call)
    monkeypatch.setattr(runtime_harness.asyncio, "sleep", no_sleep)

    status = await _wait_for_run_completion(object(), _RUN_ID, 5.0)

    assert status == {"status": "completed"}
    assert calls == [
        ("graph_jobs", {"action": "status", "job_id": _RUN_ID}),
        ("graph_jobs", {"action": "status", "job_id": _RUN_ID}),
    ]


def test_delegated_evidence_requires_exact_configured_model(monkeypatch) -> None:
    case = _delegated_case()
    digest = _skill_instruction_digest(case.skill)
    monkeypatch.setattr(
        runtime_harness,
        "_expected_delegated_model_ref",
        lambda _model_class: _MODEL_REF,
    )
    base = {
        "status": "completed",
        "model_class": "economy",
        "skill_ref": _SKILL_REF,
        "skill_instruction_digest": digest,
    }

    errors, *_ = _validate_delegated_runtime_evidence(
        case, dict(base, model_ref="pref_model_" + "f" * 64)
    )
    assert errors == ["model_reference_mismatch"]

    errors, model_ref, skill_ref, observed_digest = (
        _validate_delegated_runtime_evidence(case, dict(base, model_ref=_MODEL_REF))
    )
    assert errors == []
    assert (model_ref, skill_ref, observed_digest) == (
        _MODEL_REF,
        _SKILL_REF,
        digest,
    )


@pytest.mark.asyncio
async def test_delegated_case_binds_run_model_skill_and_trace(
    monkeypatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    case = _delegated_case()
    digest = _skill_instruction_digest(case.skill)

    async def fake_call(_client, name, arguments, _timeout):
        calls.append((name, arguments))
        if name == "graph_orchestrate":
            return {"output": _semantic_payload(), "run_id": _RUN_ID}
        return {
            "status": "completed",
            "model_class": "economy",
            "model_ref": _MODEL_REF,
            "skill_ref": _SKILL_REF,
            "skill_instruction_digest": digest,
        }

    async def fake_trace(
        _client,
        _tool,
        _started_at,
        expected_name,
        expected_evidence,
        _timeout,
    ):
        assert expected_name == _TRACE_NAME
        assert expected_evidence == {
            "run_ref": _RUN_REF,
            "model_ref": _MODEL_REF,
            "model_class": "economy",
            "skill_ref": _SKILL_REF,
            "skill_body_ref": persistence_reference(
                "skill_body", digest, namespace="skill-validation"
            ),
        }
        return "trace-id", "run-evidence"

    async def fake_ingestion(*_args, **_kwargs):
        return 1

    monkeypatch.setattr(runtime_harness, "_call_tool", fake_call)
    monkeypatch.setattr(runtime_harness, "_wait_for_expected_trace", fake_trace)
    monkeypatch.setattr(
        runtime_harness, "_verify_parent_ingested_trace", fake_ingestion
    )
    monkeypatch.setattr(
        runtime_harness,
        "_expected_delegated_model_ref",
        lambda _model_class: _MODEL_REF,
    )

    result = await _run_delegated_case(
        case,
        client=object(),
        langfuse_tool="langfuse_observability",
        tenant_id=_TENANT_ID,
        max_steps=3,
        token_budget=256,
        case_timeout=5.0,
        trace_timeout=5.0,
    )

    assert result.passed
    assert result.run_ref == _RUN_REF
    assert result.trace_name == f"graph_run:{result.run_ref}"
    orchestrate_args = next(args for name, args in calls if name == "graph_orchestrate")
    assert "action" not in orchestrate_args
    assert orchestrate_args["model_class"] == "economy"
    assert orchestrate_args["reasoning_effort"] == ""
    assert orchestrate_args["response_format"] == "json"
    assert ("graph_jobs", {"action": "status", "job_id": _RUN_ID}) in calls
