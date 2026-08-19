"""Focused contracts for the governed NE-109 data-prep surface."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pytest
from pydantic import BaseModel, ConfigDict

import agent_utilities.knowledge_graph.ingestion.envelope_ingest as envelope_ingest
from agent_utilities.data_prep import (
    CleanPipeline,
    CleanPlan,
    RowModelRegistry,
    plan_digest,
    row_model_digest,
    schema_digest,
)
from agent_utilities.knowledge_graph.core.session import GraphSession, ScopeError
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tool_specs import READ_ONLY_ACTIONS, TOOL_SPECS_BY_NAME
from agent_utilities.mcp.tools.data_prep_tools import (
    ArtifactACL,
    ArtifactAuthorityUnavailable,
    DataPrepModelAuthority,
    DataPrepRuntimeConfig,
    DataPrepService,
    DataPrepToolError,
    NativeCommitUnavailable,
    PreparedReceipt,
    PrepBudget,
    ResolvedArtifact,
    _canonical_arrow_bytes,
    _GraphNativeDataPrepPolicy,
    _json_payload,
    _native_apply_change_supported,
    _shape_digest,
    register_data_prep_tools,
    register_process_data_prep_runtime,
)
from agent_utilities.models.company_brain import ActorType, DataClassification
from agent_utilities.security.brain_context import ActorContext


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: int


class _Client:
    def __init__(self, *, supports_envelope: bool = True) -> None:
        self.supports_envelope = supports_envelope
        self.supports_calls: list[str] = []
        self.blob = type("Blob", (), {})()
        self.blob.store = lambda payload: (
            __import__("hashlib").sha256(payload).hexdigest()
        )
        self.blob.incref = lambda digest: None
        self.blob.unref = lambda digest: None

    def supports(self, capability: str) -> bool:
        self.supports_calls.append(capability)
        return self.supports_envelope and capability == "ApplyChangeEnvelope"


class _Engine:
    def __init__(self, *, supports_envelope: bool = True) -> None:
        self.client = _Client(supports_envelope=supports_envelope)
        self.graph_compute = self
        self.authority: _Authority | None = None


class _NativeNodes:
    def __init__(self, props: dict[str, Any]) -> None:
        self.props = props
        self.calls = 0

    def properties(self, node_id: str) -> dict[str, Any] | None:
        self.calls += 1
        return self.props if node_id == "occurrence:source" else None


class _NativeClient(_Client):
    def __init__(self, payload: bytes, props: dict[str, Any]) -> None:
        super().__init__()
        self.nodes = _NativeNodes(props)
        self.fetch_calls = 0
        self.blob.fetch = self._fetch
        self._payload = payload

    def _fetch(self, digest: str) -> bytes:
        self.fetch_calls += 1
        assert digest == __import__("hashlib").sha256(self._payload).hexdigest()
        return self._payload


class _NativeReadBackend:
    def __init__(self, props: dict[str, Any]) -> None:
        self.props = props
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self.point_calls: list[str] = []

    def get_node_properties(self, node_id: str) -> dict[str, Any] | None:
        """Mirror EpistemicGraphBackend's typed native point-read signature."""
        from agent_utilities.knowledge_graph.core.session import current_session

        assert current_session() is not None
        self.point_calls.append(node_id)
        return self.props if node_id == "occurrence:source" else None

    def execute_read(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        assert include_epistemic is False
        self.calls.append((query, params))
        return [{"n": self.props}]


class _NativeEngine:
    def __init__(self, payload: bytes, props: dict[str, Any], runtime: Any) -> None:
        self.client = _NativeClient(payload, props)
        self.graph_compute = self
        self.backend = _NativeReadBackend(props)
        self.data_prep_runtime_config = runtime
        self.authority = SimpleNamespace(
            envelopes=[],
            commit_result={
                "status": "success",
                "envelope_id": "envelope:native",
                "idempotency_key": "idempotency:native",
                "native_atomic": True,
            },
        )


class _Authority:
    def __init__(self, *, icv_available: bool = True) -> None:
        self.icv_available = icv_available
        self.preview_calls = 0
        self.inline_policy = True
        self.stored: list[str] = []
        self.refs: list[str] = []
        self.unrefs: list[str] = []
        self.envelopes: list[Any] = []
        self.commit_result: dict[str, Any] = {
            "status": "success",
            "envelope_id": "envelope:test",
            "idempotency_key": "idempotency:test",
            "native_atomic": True,
        }
        self.engine = _Engine()
        self.engine.authority = self
        self.table = pa.table({"value": [1, 2]})
        self.model_digest = row_model_digest(_Row)
        self.schema_digest = schema_digest(self.table)
        self.shape_digest = _shape_digest(self.table)
        self.content_digest = "sha256:" + "b" * 64
        self.registry = RowModelRegistry({"model:data:v1": _Row})
        self.source = ResolvedArtifact(
            artifact_ref="artifact:source",
            content_ref="blob:source",
            media_type="application/vnd.apache.arrow.stream",
            schema_ref="schema:data:v1",
            schema_digest=self.schema_digest,
            shape_ref="shape:data:v1",
            shape_digest=self.shape_digest,
            tenant_id="tenant-a",
            owner_id="principal-a",
            acl=ArtifactACL(
                is_public=False,
                principal_ids=("principal-a",),
                roles=("data-prep-reader",),
            ),
            content_digest=self.content_digest,
            classification=DataClassification.CONFIDENTIAL,
            retention="P90D",
            legal_hold=True,
            policy_version="test-policy",
            expires_at_ms=0,
            compressed_bytes=self.table.nbytes,
            decoded_bytes=self.table.nbytes,
            rows=self.table.num_rows,
            columns=self.table.num_columns,
            nesting_depth=0,
            table=self.table,
        )

    def artifact(
        self, artifact_ref: str, *, session: GraphSession, budget: Any
    ) -> ResolvedArtifact:
        assert artifact_ref == self.source.artifact_ref
        return self.source

    def records_artifact(
        self,
        records: list[dict[str, Any]],
        *,
        session: GraphSession,
        budget: Any,
    ) -> ResolvedArtifact:
        assert records
        return self.source

    def approved_models(self, *, session: GraphSession) -> RowModelRegistry:
        return self.registry

    def inline_records_policy_available(self, *, session: GraphSession) -> bool:
        return self.inline_policy

    def preview_ref(
        self,
        source: ResolvedArtifact,
        *,
        output_table: Any,
        evidence: Any,
        request: Any,
        session: GraphSession,
    ) -> PreparedReceipt:
        self.preview_calls += 1
        assert source is self.source
        from agent_utilities.mcp.tools.data_prep_tools import _expected_receipt_fields

        fields = _expected_receipt_fields(
            source, output_table, evidence, request, session=session
        )
        issued_at_ms = int(time.time() * 1000)
        return PreparedReceipt(
            **fields,
            native_atomic=True,
            issued_at_ms=issued_at_ms,
            actor_id=session.actor.actor_id,
            endpoint="data-prep",
            expires_at_ms=issued_at_ms + 5 * 60 * 1000,
        )

    def output_governance(
        self,
        source: ResolvedArtifact,
        *,
        output_table: Any,
        output_schema_ref: str,
        output_schema_digest: str,
        output_shape_ref: str,
        output_shape_digest: str,
        session: GraphSession,
    ) -> ResolvedArtifact:
        from agent_utilities.mcp.tools.data_prep_tools import _canonical_arrow_bytes

        output_bytes = _canonical_arrow_bytes(output_table)
        return replace(
            source,
            artifact_ref=f"artifact:prepared:{output_bytes.hex()[:16]}",
            schema_ref=output_schema_ref,
            schema_digest=output_schema_digest,
            shape_ref=output_shape_ref,
            shape_digest=output_shape_digest,
            content_digest="sha256:"
            + __import__("hashlib").sha256(output_bytes).hexdigest(),
            compressed_bytes=len(output_bytes),
            decoded_bytes=output_table.nbytes,
            rows=output_table.num_rows,
            columns=output_table.num_columns,
            table=output_table,
        )

    def native_engine(self, *, session: GraphSession) -> _Engine:
        return self.engine

    def icv_policy_available(self, *, session: GraphSession) -> bool:
        return self.icv_available

    def native_atomic_available(self, *, session: GraphSession) -> bool:
        return True

    def store_blob(
        self, payload: bytes, *, media_type: str, session: GraphSession
    ) -> str:
        digest = "sha256:" + __import__("hashlib").sha256(payload).hexdigest()
        self.stored.append(digest)
        return digest

    def incref_blob(self, digest: str, *, session: GraphSession) -> None:
        self.refs.append(digest)

    def unref_blob(self, digest: str, *, session: GraphSession) -> None:
        self.unrefs.append(digest)


@pytest.fixture(autouse=True)
def _canonical_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    def ingest(engine: _Engine, envelope: Any) -> dict[str, Any]:
        assert engine.authority is not None
        engine.authority.envelopes.append(envelope)
        return dict(engine.authority.commit_result)

    monkeypatch.setattr(envelope_ingest, "ingest_envelope", ingest)


def _session() -> GraphSession:
    actor = ActorContext(
        actor_id="principal-a",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("data-prep-reader",),
        tenant_id="tenant-a",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant="tenant-a",
        scopes=frozenset({"kg:read", "kg:write"}),
        policy_version="test-policy",
        audience="test-audience",
        trace_context="trace:test",
    )


def _fixture() -> tuple[_Authority, DataPrepService, dict[str, Any], GraphSession]:
    authority = _Authority()
    plan = CleanPlan.model_validate(
        {
            "schema_version": "1",
            "steps": [{"verb": "canonical_names"}],
            "profile": {
                "max_rows": 10,
                "max_columns": 10,
                "max_steps": 4,
                "max_bytes": 1_000_000,
            },
            "invalid_row_disposition": "fail",
            "plan_ref": "plan:data:v1",
            "policy_ref": "policy:data:v1",
            "model_ref": "model:data:v1",
            "model_digest": authority.model_digest,
        }
    )
    payload: dict[str, Any] = {
        "schema_version": "data-prep-tool.v1",
        "plan": plan.model_dump(mode="json"),
        "plan_ref": plan.plan_ref,
        "plan_digest": plan_digest(plan),
        "model_ref": plan.model_ref,
        "model_digest": authority.model_digest,
        "schema_ref": authority.source.schema_ref,
        "schema_digest": authority.schema_digest,
        "shape_ref": authority.source.shape_ref,
        "shape_digest": authority.shape_digest,
        "artifact_ref": authority.source.artifact_ref,
        "budget": {
            "max_rows": 10,
            "max_columns": 10,
            "max_compressed_bytes": 1_000_000,
            "max_decoded_bytes": 1_000_000,
            "max_depth": 1,
            "max_wall_time_ms": 10_000,
        },
    }
    return (
        authority,
        DataPrepService(authority, clock_ms=lambda: 1),
        payload,
        _session(),
    )


def _native_fixture() -> tuple[_NativeEngine, dict[str, Any], GraphSession]:
    table = pa.table({"value": [1, 2]})
    output_bytes = _canonical_arrow_bytes(table)
    output_digest = "sha256:" + __import__("hashlib").sha256(output_bytes).hexdigest()
    schema_value = schema_digest(table)
    shape_value = _shape_digest(table)
    props: dict[str, Any] = {
        "node_type": "AssetOccurrence",
        "content_digest": output_digest,
        "media_type": "application/vnd.apache.arrow.stream",
        "tenant_id": "tenant-a",
        "_owner_id": "principal-a",
        "_shared_scope": "private",
        "acl": {
            "is_public": False,
            "principal_ids": ["principal-a"],
            "principal_emails": [],
            "group_ids": [],
            "read_roles": ["data-prep-reader"],
            "markings": [],
        },
        "classification": DataClassification.CONFIDENTIAL.value,
        "retention": "P90D",
        "legal_hold": True,
        "policy_version": "test-policy",
        "expires_at_ms": 0,
        "schema_ref": "schema:native:v1",
        "schema_digest": schema_value,
        "shape_ref": "shape:native:v1",
        "shape_digest": shape_value,
        "file_size_bytes": len(output_bytes),
        "decoded_bytes": table.nbytes,
        "rows": table.num_rows,
        "columns": table.num_columns,
        "nesting_depth": 0,
    }
    model_authority = DataPrepModelAuthority(
        registry=RowModelRegistry({"model:data:v1": _Row}),
        config_digest="sha256:" + "c" * 64,
        connector_version="connector:data-prep:v1",
    )
    runtime = DataPrepRuntimeConfig(
        model_authority=model_authority,
        policy_authority=_GraphNativeDataPrepPolicy(),
        icv_policy_available=True,
    )
    engine = _NativeEngine(output_bytes, props, runtime)
    plan = CleanPlan.model_validate(
        {
            "schema_version": "1",
            "steps": [{"verb": "canonical_names"}],
            "profile": {
                "max_rows": 10,
                "max_columns": 10,
                "max_steps": 4,
                "max_bytes": 1_000_000,
            },
            "invalid_row_disposition": "fail",
            "plan_ref": "plan:native:v1",
            "policy_ref": "policy:native:v1",
            "model_ref": "model:data:v1",
            "model_digest": model_authority.registry.resolve("model:data:v1").digest,
        }
    )
    payload: dict[str, Any] = {
        "schema_version": "data-prep-tool.v1",
        "plan": plan.model_dump(mode="json"),
        "plan_ref": plan.plan_ref,
        "plan_digest": plan_digest(plan),
        "model_ref": plan.model_ref,
        "model_digest": plan.model_digest,
        "schema_ref": props["schema_ref"],
        "schema_digest": props["schema_digest"],
        "shape_ref": props["shape_ref"],
        "shape_digest": props["shape_digest"],
        "artifact_ref": "occurrence:source",
        "budget": {
            "max_rows": 10,
            "max_columns": 10,
            "max_compressed_bytes": 1_000_000,
            "max_decoded_bytes": 1_000_000,
            "max_depth": 1,
            "max_wall_time_ms": 10_000,
        },
    }
    return engine, payload, _session()


def _prepared_payload(
    payload: dict[str, Any], authority: _Authority, session: GraphSession
) -> dict[str, Any]:
    plan = CleanPlan.model_validate(payload["plan"])
    result = CleanPipeline(plan, model_registry=authority.registry).run(authority.table)
    from agent_utilities.mcp.tools.data_prep_tools import _expected_receipt_fields

    receipt = PreparedReceipt(
        **_expected_receipt_fields(
            authority.source,
            result.table,
            result.evidence,
            type("Request", (), payload)(),
            session=session,
        ),
        native_atomic=True,
        issued_at_ms=int(time.time() * 1000),
        actor_id=session.actor.actor_id,
        endpoint="data-prep",
        expires_at_ms=int(time.time() * 1000) + 5 * 60 * 1000,
    )
    prepared = dict(payload)
    prepared.update(
        {
            "prepared_ref": receipt.encode(),
            "expected_output_schema_ref": receipt.output_schema_ref,
            "expected_output_schema_digest": receipt.output_schema_digest,
            "expected_output_shape_ref": receipt.output_shape_ref,
            "expected_output_shape_digest": receipt.output_shape_digest,
        }
    )
    return prepared


def test_manifest_and_intent_authority_expose_all_actions() -> None:
    spec = TOOL_SPECS_BY_NAME["graph_data_prep"]
    assert spec.actions == (
        "clean_dataset",
        "commit_prepared",
        "profile_dataset",
        "validate_prepared",
    )
    assert READ_ONLY_ACTIONS["graph_data_prep"] == frozenset(
        {"profile_dataset", "clean_dataset", "validate_prepared"}
    )


def test_profile_is_bounded_privacy_safe_and_side_effect_free() -> None:
    authority, service, payload, session = _fixture()

    result = service.execute("profile_dataset", payload, session=session)

    assert result["side_effects"] == []
    assert result["profile"]["rows"] == 2
    assert authority.preview_calls == 0
    assert "value" not in json.dumps(result, sort_keys=True)
    assert "content_ref" not in result["artifact"]


@pytest.mark.parametrize("proof", ["tenant", "expiry", "acl", "depth"])
def test_profile_requires_tenant_acl_expiry_and_content_proofs(proof: str) -> None:
    authority, service, payload, session = _fixture()
    if proof == "tenant":
        authority.source = replace(authority.source, tenant_id="tenant-b")
    elif proof == "expiry":
        authority.source = replace(authority.source, expires_at_ms=1)
    elif proof == "acl":
        authority.source = replace(
            authority.source,
            owner_id="principal-other",
            acl=ArtifactACL(is_public=False, principal_ids=(), roles=()),
        )
    else:
        authority.source = replace(authority.source, nesting_depth=1)

    with pytest.raises((DataPrepToolError, PermissionError)):
        service.execute("profile_dataset", payload, session=session)


def test_profile_honors_verified_group_acl_membership() -> None:
    authority, service, payload, session = _fixture()
    authority.source = replace(
        authority.source,
        owner_id="principal-other",
        acl=ArtifactACL(is_public=False, group_ids=("group-data",)),
    )
    grouped_actor = replace(session.actor, groups=("group-data",))

    result = service.execute(
        "profile_dataset",
        payload,
        session=replace(session, actor=grouped_actor),
    )

    assert result["profile"]["rows"] == 2


def test_clean_and_validate_reuse_ne108_without_mutation() -> None:
    authority, service, payload, session = _fixture()

    cleaned = service.execute("clean_dataset", payload, session=session)
    validated = service.execute(
        "validate_prepared",
        _prepared_payload(payload, authority, session),
        session=session,
    )

    assert cleaned["prepared_artifact_ref"].startswith("prep:v1:")
    assert cleaned["side_effects"] == []
    assert validated["valid"] is True
    assert validated["side_effects"] == []
    assert authority.preview_calls == 1
    assert authority.stored == []


def test_commit_fails_closed_without_icv_policy() -> None:
    authority, _, payload, session = _fixture()
    authority.icv_available = False
    service = DataPrepService(authority, clock_ms=lambda: 1)

    with pytest.raises(NativeCommitUnavailable):
        service.execute(
            "commit_prepared",
            _prepared_payload(payload, authority, session),
            session=session,
        )

    assert authority.stored == []


def test_commit_requires_verified_write_scope_before_authority_access() -> None:
    authority, _, payload, session = _fixture()
    read_only = replace(session, scopes=frozenset({"kg:read"}))
    service = DataPrepService(authority, clock_ms=lambda: 1)

    with pytest.raises(ScopeError):
        service.execute(
            "commit_prepared",
            _prepared_payload(payload, authority, session),
            session=read_only,
        )

    assert authority.stored == []


def test_commit_uses_only_native_change_envelope() -> None:
    authority, _, payload, session = _fixture()
    service = DataPrepService(authority, clock_ms=lambda: 1)
    result = service.execute(
        "commit_prepared",
        _prepared_payload(payload, authority, session),
        session=session,
    )

    assert result["side_effects"] == ["native_change_envelope"]
    assert result["commit"]["native_atomic"] is True
    assert len(authority.envelopes) == 1
    envelope = authority.envelopes[0]
    assert envelope.tenant == session.tenant
    assert envelope.blob_ref == authority.stored[0]
    assert envelope.blob_digest == authority.stored[0]
    assert envelope.blob_length > 0
    assert envelope.blob_media_type == "application/vnd.apache.arrow.stream"
    assert envelope.source_acl.is_public is False
    assert envelope.classification is DataClassification.CONFIDENTIAL
    assert envelope.source_acl.user_emails == []
    assert envelope.source_acl.group_ids == []
    assert envelope.provenance["acl_principal_ids"] == ["principal-a"]
    assert envelope.structured_evidence == envelope.provenance["prep_evidence"]
    assert envelope.provenance["prep_evidence_digest"]
    assert envelope.legal_hold is True
    assert "value" not in json.dumps(result, sort_keys=True)


def test_distinct_trusted_output_governance_denies_downgrade() -> None:
    authority, service, payload, session = _fixture()
    original = authority.output_governance

    def weaker(*args: Any, **kwargs: Any) -> ResolvedArtifact:
        output = original(*args, **kwargs)
        return replace(
            output,
            classification=DataClassification.PUBLIC,
            acl=ArtifactACL(is_public=True),
            retention=None,
            legal_hold=False,
        )

    authority.output_governance = weaker  # type: ignore[method-assign]
    with pytest.raises(DataPrepToolError, match="downgrade|broader|retention|hold"):
        service.execute("clean_dataset", payload, session=session)


def test_served_path_uses_registered_process_provider_and_public_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastmcp import FastMCP

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.session import use_session

    authority, _, payload, session = _fixture()
    engine = authority.engine
    engine.data_prep_artifact_authority = authority
    assert register_process_data_prep_runtime(engine) is True
    monkeypatch.setattr(
        IntelligenceGraphEngine,
        "get_active",
        classmethod(lambda _cls: engine),
    )
    register_data_prep_tools(FastMCP("data-prep-test"))
    served = kg_server.REGISTERED_TOOLS["graph_data_prep"]
    prepared = _prepared_payload(payload, authority, session)

    with use_session(session):
        raw = served(
            action="commit_prepared",
            params_json=json.dumps(prepared),
        )

    result = json.loads(raw)
    assert result["commit"]["native_atomic"] is True
    assert len(authority.envelopes) == 1
    assert authority.envelopes[0].blob_digest == authority.envelopes[0].source_version
    assert authority.envelopes[0].structured_evidence
    assert engine.authority is authority


def test_canonical_startup_composes_native_provider_without_manual_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastmcp import FastMCP

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.session import use_session

    engine, payload, session = _native_fixture()
    assert not hasattr(engine, "data_prep_artifact_authority")
    assert register_process_data_prep_runtime(engine) is True
    assert hasattr(engine, "data_prep_artifact_authority")
    monkeypatch.setattr(
        IntelligenceGraphEngine,
        "get_active",
        classmethod(lambda _cls: engine),
    )
    register_data_prep_tools(FastMCP("native-data-prep-test"))
    served = kg_server.REGISTERED_TOOLS["graph_data_prep"]

    with use_session(session):
        clean = json.loads(
            served(action="clean_dataset", params_json=json.dumps(payload))
        )
        prepared = dict(payload)
        prepared.update(
            {
                "prepared_ref": clean["prepared_artifact_ref"],
                "expected_output_schema_ref": clean["output_schema_ref"],
                "expected_output_schema_digest": clean["output_schema_digest"],
                "expected_output_shape_ref": clean["output_shape_ref"],
                "expected_output_shape_digest": clean["output_shape_digest"],
            }
        )
        committed = json.loads(
            served(action="commit_prepared", params_json=json.dumps(prepared))
        )

    assert committed["commit"]["native_atomic"] is True
    assert len(engine.authority.envelopes) == 1
    envelope = engine.authority.envelopes[0]
    assert envelope.blob_digest == envelope.source_version
    assert envelope.structured_evidence == envelope.provenance["prep_evidence"]
    assert envelope.provenance["input_content_digest"]
    assert engine.client.fetch_calls >= 2  # clean + commit recomputation
    assert engine.backend.point_calls == [
        "occurrence:source",
        "occurrence:source",
    ]
    assert engine.backend.calls == []


@pytest.mark.parametrize(
    "case", ["tenant", "owner", "stale_policy", "expired", "malformed"]
)
def test_native_provider_denies_before_blob_fetch(case: str) -> None:
    engine, payload, session = _native_fixture()
    assert register_process_data_prep_runtime(engine) is True
    provider = engine._data_prep_runtime_provider
    budget = PrepBudget.model_validate(payload["budget"])
    if case == "tenant":
        denied_actor = replace(session.actor, tenant_id="tenant-b")
        denied = replace(session, tenant="tenant-b", actor=denied_actor)
    elif case == "owner":
        denied_actor = replace(
            session.actor, actor_id="principal-b", roles=(), groups=()
        )
        denied = replace(session, actor=denied_actor)
    else:
        denied = session
    if case == "stale_policy":
        denied = replace(session, policy_version="old-policy")
    if case == "expired":
        engine.client.nodes.props["expires_at_ms"] = 1
    if case == "malformed":
        engine.client.nodes.props["classification"] = "not-a-classification"

    with pytest.raises(
        (PermissionError, DataPrepToolError, ArtifactAuthorityUnavailable)
    ):
        provider.artifact("occurrence:source", session=denied, budget=budget)
    assert engine.client.fetch_calls == 0


def test_native_provider_routes_point_read_through_verified_graph_view() -> None:
    from agent_utilities.knowledge_graph.core.session import use_session

    engine, payload, session = _native_fixture()
    engine.graph_compute.graph_name = "__commons__"
    routed: list[str] = []
    view = SimpleNamespace(
        backend=engine.backend,
        graph_compute=engine.graph_compute,
        client=engine.client,
    )

    def for_graph(graph_name: str) -> Any:
        routed.append(graph_name)
        return view

    engine.for_graph = for_graph
    session = replace(session, graph="tenant-a-graph")
    assert register_process_data_prep_runtime(engine) is True
    provider = engine._data_prep_runtime_provider

    with use_session(session):
        provider.artifact(
            "occurrence:source",
            session=session,
            budget=PrepBudget.model_validate(payload["budget"]),
        )

    assert routed == ["tenant-a-graph"]
    assert engine.backend.point_calls == ["occurrence:source"]
    assert engine.client.fetch_calls == 1


def test_missing_authoritative_model_dependency_is_served_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastmcp import FastMCP

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.session import use_session

    engine, payload, session = _native_fixture()
    engine.data_prep_runtime_config = DataPrepRuntimeConfig(
        policy_authority=_GraphNativeDataPrepPolicy(),
        icv_policy_available=True,
    )
    assert register_process_data_prep_runtime(engine) is False
    monkeypatch.setattr(
        IntelligenceGraphEngine,
        "get_active",
        classmethod(lambda _cls: engine),
    )
    register_data_prep_tools(FastMCP("missing-model-data-prep-test"))
    served = kg_server.REGISTERED_TOOLS["graph_data_prep"]
    with use_session(session):
        result = json.loads(
            served(action="profile_dataset", params_json=json.dumps(payload))
        )
    assert result["error"]["code"] == "dependency_unavailable"


def test_process_provider_registration_cannot_be_replaced_by_runtime_metadata() -> None:
    authority, _, _, _ = _fixture()
    engine = authority.engine
    engine.data_prep_artifact_authority = authority
    assert register_process_data_prep_runtime(engine) is True

    replacement = object()
    engine.data_prep_artifact_authority = replacement
    assert register_process_data_prep_runtime(engine) is False


def test_inline_raw_bytes_and_nested_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="inline bytes"):
        _json_payload(json.dumps({"bytes": "not-an-artifact"}))

    authority, service, payload, session = _fixture()
    inline = dict(payload)
    inline.pop("artifact_ref")
    inline["records"] = [{"value": {"secret": "raw"}}]

    with pytest.raises(DataPrepToolError, match="typed plan gate"):
        service.execute("profile_dataset", inline, session=session)


def test_receipt_binds_output_content_and_rejects_tampering() -> None:
    authority, service, payload, session = _fixture()

    cleaned = service.execute("clean_dataset", payload, session=session)
    receipt = PreparedReceipt.decode(cleaned["prepared_artifact_ref"])

    assert receipt.tenant_id == session.tenant
    assert receipt.input_content_digest == authority.source.content_digest
    assert receipt.output_content_digest != receipt.input_content_digest
    assert receipt.output_schema_ref == cleaned["output_schema_ref"]
    assert receipt.output_shape_ref == cleaned["output_shape_ref"]

    tampered = cleaned["prepared_artifact_ref"][:-1] + (
        "0" if cleaned["prepared_artifact_ref"][-1] != "0" else "1"
    )
    with pytest.raises(DataPrepToolError, match="signature|malformed|encoding"):
        PreparedReceipt.decode(tampered)


def test_commit_requires_exact_schema_and_shape_approvals() -> None:
    authority, service, payload, session = _fixture()
    prepared = _prepared_payload(payload, authority, session)
    prepared["expected_output_shape_digest"] = "sha256:" + "f" * 64

    with pytest.raises(DataPrepToolError, match="schema or shape|approved"):
        service.execute("commit_prepared", prepared, session=session)
    assert authority.stored == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("media_type", "application/json"),
        ("shape_digest", "sha256:" + "f" * 64),
        ("schema_digest", "sha256:" + "f" * 64),
    ],
)
def test_source_media_schema_and_shape_fingerprints_are_authoritative(
    field: str, value: str
) -> None:
    authority, service, payload, session = _fixture()
    authority.source = replace(authority.source, **{field: value})

    with pytest.raises(DataPrepToolError, match="media|fingerprint|schema|shape"):
        service.execute("profile_dataset", payload, session=session)


def test_inline_records_require_process_governance_policy() -> None:
    authority, service, payload, session = _fixture()
    authority.inline_policy = False
    inline = dict(payload)
    inline.pop("artifact_ref")
    inline["records"] = [{"value": 1}]

    with pytest.raises(ArtifactAuthorityUnavailable, match="policy|authority"):
        service.execute("profile_dataset", inline, session=session)
    assert authority.stored == []


def test_canonical_plan_size_is_bounded_before_kernel_dispatch() -> None:
    with pytest.raises(DataPrepToolError, match="plan exceeds"):
        _json_payload(json.dumps({"plan": {"value": "x" * (128 * 1024)}}))


def test_native_failure_compensates_the_blob_reference() -> None:
    authority, service, payload, session = _fixture()
    authority.commit_result = {"status": "rejected", "native_atomic": True}

    with pytest.raises(NativeCommitUnavailable):
        service.execute(
            "commit_prepared",
            _prepared_payload(payload, authority, session),
            session=session,
        )

    assert len(authority.refs) == 1
    assert authority.unrefs == authority.refs


def test_support_probe_checks_every_native_candidate() -> None:
    class Candidate:
        def __init__(self, value: bool) -> None:
            self.calls = 0
            self.client = self
            self.value = value

        def supports(self, operation: str) -> bool:
            self.calls += 1
            assert operation == "ApplyChangeEnvelope"
            return self.value

    first = Candidate(False)
    second = Candidate(True)
    engine = type("Engine", (), {"client": first, "graph_compute": second})()

    assert _native_apply_change_supported(engine) is True
    assert first.calls == 1
    assert second.calls == 1


def test_acl_parser_keeps_principal_ids_out_of_email_fields() -> None:
    acl = ArtifactACL.from_value(
        {
            "is_public": False,
            "principal_ids": ["actor-123"],
            "principal_emails": ["reader@example.test"],
            "group_ids": ["group-1"],
            "read_roles": ["reader"],
            "markings": ["finance"],
        }
    )
    assert acl.principal_ids == ("actor-123",)
    assert acl.principal_emails == ("reader@example.test",)
    assert acl.group_ids == ("group-1",)
    assert acl.markings == ("finance",)
    with pytest.raises(DataPrepToolError, match="principal IDs"):
        ArtifactACL.from_value(
            {"is_public": False, "principal_ids": ["actor@example.test"]}
        )


def test_receipt_is_bound_to_current_actor_and_tenant() -> None:
    authority, service, payload, session = _fixture()
    prepared = _prepared_payload(payload, authority, session)

    other_actor = ActorContext(
        actor_id="principal-b",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("data-prep-reader",),
        tenant_id="tenant-a",
        authenticated=True,
    )
    authority.source = replace(
        authority.source,
        owner_id="principal-b",
        acl=ArtifactACL(
            is_public=False,
            principal_ids=("principal-b",),
            roles=("data-prep-reader",),
        ),
    )
    with pytest.raises(PermissionError, match="receipt identity"):
        service.execute(
            "commit_prepared",
            prepared,
            session=replace(session, actor=other_actor),
        )

    with pytest.raises(PermissionError, match="tenant"):
        service.execute(
            "commit_prepared",
            prepared,
            session=replace(session, tenant="tenant-b"),
        )
