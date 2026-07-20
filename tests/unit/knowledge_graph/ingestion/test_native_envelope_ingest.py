"""Production contract for engine-native ChangeEnvelope ingestion."""

from __future__ import annotations

import msgpack
import pytest

import agent_utilities.knowledge_graph.ingestion.envelope_ingest as module
from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    reset_session,
    set_session,
)
from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope
from agent_utilities.models.company_brain import ActorType, DataClassification
from agent_utilities.protocols.source_connectors.base import ExternalAccess
from agent_utilities.security.brain_context import ActorContext


class _Nodes:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, object]] = {}

    def properties(self, node_id: str):
        return self.values.get(node_id)

    def list(self):
        return list(self.values.items())


class _Changes:
    def __init__(self, nodes: _Nodes) -> None:
        self.nodes = nodes
        self.records: dict[str, dict[str, object]] = {}
        self.versions: dict[str, dict[str, object]] = {}
        self.cursors: dict[tuple[str, str], dict[str, object]] = {}
        self.applied: list[dict[str, object]] = []
        self.failures: list[Exception] = []

    def get(self, envelope_id: str):
        return self.records.get(envelope_id)

    def content_version(self, object_id: str):
        return self.versions.get(object_id)

    def cursor(self, source: str, partition: str = ""):
        return self.cursors.get((source, partition))

    def apply(self, envelope: dict[str, object]):
        self.applied.append(envelope)
        if self.failures:
            raise self.failures.pop(0)
        mutation = envelope["mutation"]
        assert isinstance(mutation, dict)
        for operation in mutation["operations"]:
            method = operation["method"]
            if method["method"] != "AddNode":
                continue
            params = method["params"]
            self.nodes.values[params["node_id"]] = msgpack.unpackb(
                params["properties_msgpack"], raw=False
            )
        version = envelope["content_version"]
        assert isinstance(version, dict)
        self.versions[str(version["object_id"])] = version
        cursor = envelope.get("cursor")
        if isinstance(cursor, dict):
            self.cursors[(str(cursor["source"]), str(cursor["partition"]))] = cursor
        envelope_id = str(envelope["envelope_id"])
        self.records[envelope_id] = envelope
        return {
            "envelope_id": envelope_id,
            "batch_id": mutation["batch_id"],
            "replayed": False,
            "projection_pending": False,
            "outbox_count": len(mutation["operations"]) + 2,
        }


class _Rdf:
    def __init__(self) -> None:
        self.reports: list[dict[str, object]] = [{"conforms": True, "results": []}]
        self.validations: list[tuple[str, str]] = []

    def validate_shacl(self, shapes: str, data_graph: str):
        self.validations.append((shapes, data_graph))
        if len(self.reports) > 1:
            return self.reports.pop(0)
        return self.reports[0]


class _Client:
    def __init__(self, *, supported: bool = True) -> None:
        self.nodes = _Nodes()
        self.changes = _Changes(self.nodes)
        self.rdf = _Rdf()
        self.supported = supported

    def supports(self, operation: str) -> bool:
        return self.supported and operation in {
            "ApplyChangeEnvelope",
            "GetChangeCursor",
        }


class _Compute:
    def __init__(self, graph: str, *, supported: bool = True) -> None:
        self.graph_name = graph
        self.catalog_epoch = 3
        self.placement_group = 8
        self.client = _Client(supported=supported)

    def for_graph(self, graph: str):
        self.graph_name = graph
        return self


@pytest.fixture(autouse=True)
def _native_profile(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APP_PROFILE", "dev")
    module._NATIVE_GRAPH_VERSIONS.clear()
    module._NATIVE_LOCKS.clear()
    actor = ActorContext(
        actor_id="fixture-service",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id="fixture-tenant",
        authenticated=True,
    )
    token = set_session(
        GraphSession(
            actor=actor,
            tenant="fixture-tenant",
            scopes=frozenset({"kg:read", "kg:write"}),
            graph="fixture-graph",
            policy_version="fixture-policy",
            audience="fixture-audience",
        )
    )
    try:
        yield
    finally:
        reset_session(token)


def _envelope(**overrides) -> ChangeEnvelope:
    values = {
        "connector": "fixture-connector",
        "source_object_id": "object-1",
        "source_version": "1",
        "checkpoint": "1",
        "typed_payload": {
            "id": "object-1",
            "type": "FixtureRecord",
            "name": "Synthetic record",
        },
    }
    values.update(overrides)
    return ChangeEnvelope(**values)


@pytest.mark.parametrize(
    "unsafe_role",
    ["operator" + "@example.invalid", "/" + "home/example/private"],
)
def test_privacy_gate_quarantines_unsafe_direct_acl_roles(unsafe_role: str) -> None:
    envelope = _envelope(
        source_acl=ExternalAccess(is_public=False, read_roles=[unsafe_role])
    )

    sanitized = module._privacy_gate(envelope)

    assert sanitized.source_acl == ExternalAccess.quarantined()
    report = sanitized.provenance["persistence_privacy"]
    assert report["redactions"] >= 1
    assert "acl_principal" in report["detected_types"]


def test_privacy_gate_preserves_valid_opaque_digest_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime deny terms cannot collide with cryptographic ID material."""

    monkeypatch.setattr(
        "agent_utilities.security.persistence_privacy._runtime_deny_terms",
        lambda: ("bad",),
    )
    digest_128 = "bad" + ("0" * 29)
    digest_256 = "bad" + ("0" * 61)
    trace_id = f"langfuse:trace:{digest_128}"
    observation_id = f"langfuse:observation:{digest_128}"
    envelope = _envelope(
        source_object_id=trace_id,
        source_version=digest_256,
        typed_payload={
            "id": trace_id,
            "type": "Trace",
            "_nodes": [{"id": observation_id, "type": "Observation"}],
            "_links": [
                {
                    "source": observation_id,
                    "target": trace_id,
                    "type": "belongsToTrace",
                }
            ],
            "valid": "bad-content",
        },
    )

    sanitized = module._privacy_gate(envelope)

    assert sanitized.source_object_id == trace_id
    assert sanitized.source_version == digest_256
    assert sanitized.typed_payload is not None
    assert sanitized.typed_payload["id"] == trace_id
    assert sanitized.typed_payload["_nodes"][0]["id"] == observation_id
    assert sanitized.typed_payload["_links"][0]["source"] == observation_id
    assert sanitized.typed_payload["_links"][0]["target"] == trace_id
    assert sanitized.typed_payload["valid"] == "[REDACTED_IDENTITY_TERM]-content"


def test_privacy_gate_still_rejects_unsafe_opaque_identifier_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent_utilities.security.persistence_privacy._runtime_deny_terms",
        lambda: ("private",),
    )
    unsafe_id = "private:trace:" + ("0" * 32)

    with pytest.raises(ValueError, match="unsafe envelope identity"):
        module._privacy_gate(
            _envelope(
                source_object_id=unsafe_id,
                typed_payload={"id": unsafe_id, "type": "Trace"},
            )
        )


def test_privacy_gate_still_rejects_nonopaque_sensitive_payload_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent_utilities.security.persistence_privacy._runtime_deny_terms",
        lambda: (),
    )

    with pytest.raises(ValueError, match="unsafe payload identity"):
        module._privacy_gate(
            _envelope(
                typed_payload={
                    "id": "operator@example.invalid",
                    "type": "Trace",
                }
            )
        )


def test_native_cursor_read_uses_the_same_hashed_source_partition() -> None:
    compute = _Compute("graph-cursor")
    partition = module._cursor_partition("instance-a")
    compute.client.changes.cursors[("fixture-connector", partition)] = {
        "source": "fixture-connector",
        "partition": partition,
        "position": {
            "kind": "opaque",
            "value": {"cursor_type": "provider_v1", "value": "page-7"},
        },
    }

    assert (
        module.read_change_cursor(
            compute, "fixture-connector", source_instance="instance-a"
        )
        == "page-7"
    )


def test_native_apply_commits_all_authority_rows_in_one_request() -> None:
    compute = _Compute("graph-native")
    envelope = _envelope(
        typed_payload={
            "id": "object-1",
            "type": "FixtureRecord",
            "_features": [{"kind": "embedding", "value": [0.1, 0.2]}],
            "_evidence": [{"modality": "structured", "locus": {"row": 1}}],
        }
    )

    result = module.ingest_envelope(compute, envelope)

    assert result["status"] == "success"
    assert result["native_atomic"] is True
    assert result["watermark_advanced"] is True
    assert len(compute.client.changes.applied) == 1
    native = compute.client.changes.applied[0]
    mutation = native["mutation"]
    assert native["schema_version"] == 1
    assert mutation["schema_version"] == 2
    assert mutation["expected_graph_version"] == 0
    assert mutation["placement_epoch"] == 3
    assert mutation["fencing_token"] == 8
    assert mutation["idempotency_key"] == envelope.idempotency_key
    assert mutation["outbox"][0]["topic"] == "kg.mutations"
    assert native["features"] and native["evidence"]
    assert native["policies"] and native["lineage"] and native["cursor"]
    assert native["privacy"]["sanitized_payload_digest"]
    assert compute.client.nodes.values["object-1"]["tenant_id"] == "fixture-tenant"


def test_native_apply_commits_auxiliary_nodes_edges_and_policy_together() -> None:
    compute = _Compute("graph-slice")
    envelope = _envelope(
        typed_payload={
            "id": "object-1",
            "type": "Document",
            "tenant_id": "source-spoof",
            "_nodes": [
                {
                    "id": "chunk-1",
                    "type": "Chunk",
                    "content": "synthetic",
                    "tenant_id": "source-spoof",
                },
                {"id": "section-1", "type": "Section", "title": "Overview"},
            ],
            "_links": [
                {"source": "object-1", "target": "chunk-1", "type": "HAS_CHUNK"},
                {
                    "source": "object-1",
                    "target": "section-1",
                    "type": "HAS_SECTION",
                },
            ],
        }
    )

    result = module.ingest_envelope(compute, envelope)

    assert result["status"] == "success"
    native = compute.client.changes.applied[0]
    methods = [item["method"]["method"] for item in native["mutation"]["operations"]]
    assert methods == ["AddNode", "AddNode", "AddNode", "AddEdge", "AddEdge"]
    assert {row["object_id"] for row in native["policies"]} == {
        "object-1",
        "chunk-1",
        "section-1",
    }
    assert result["write_result"]["nodes"] == 3
    assert result["write_result"]["edges"] == 2
    assert len(compute.client.rdf.validations) == 1
    _shapes, data_graph = compute.client.rdf.validations[0]
    assert "Document" in data_graph
    assert "Chunk" in data_graph
    assert "Section" in data_graph
    assert all(
        compute.client.nodes.values[node_id]["tenant_id"] == "fixture-tenant"
        for node_id in ("object-1", "chunk-1", "section-1")
    )


def test_public_access_and_classification_project_consistently() -> None:
    compute = _Compute("graph-public")
    envelope = _envelope(
        source_acl=ExternalAccess.public(),
        classification=DataClassification.PUBLIC,
    )

    result = module.ingest_envelope(compute, envelope)

    assert result["status"] == "success"
    native = compute.client.changes.applied[0]
    assert native["policies"][0]["classification"] == "public"
    assert compute.client.nodes.values["object-1"]["classification"] == "public"
    acl = get_company_brain().permissions.get_acl("object-1")
    assert acl is not None
    assert acl.classification.value == "public"


def test_graph_slice_helper_commits_edge_only_batches_with_governed_marker() -> None:
    compute = _Compute("graph-edge-slice")

    result = module.ingest_graph_slice(
        compute,
        "derived-links",
        [],
        [
            {
                "source": "object-1",
                "target": "object-2",
                "relationship": "RELATED_TO",
            }
        ],
    )

    assert result["status"] == "success"
    assert result["write_result"]["nodes"] == 1
    assert result["write_result"]["edges"] == 1
    marker = next(iter(compute.client.nodes.values.values()))
    assert marker["node_type"] == "SourceMaterialization"
    assert marker["relationship_count"] == 1


def test_batch_proxy_never_invokes_raw_external_batch_in_native_profile() -> None:
    compute = _Compute("graph-proxy")
    proxy = module.NativeChangeEnvelopeEngineProxy(compute)

    result = proxy.ingest_external_batch(
        "fixture-connector",
        [{"id": "proxy-object", "type": "FixtureRecord", "name": "Synthetic"}],
        [],
    )

    assert result["status"] == "success"
    assert len(compute.client.changes.applied) == 1
    assert "proxy-object" in compute.client.nodes.values


def test_graph_slice_identity_covers_auxiliary_rows_not_only_primary_version() -> None:
    compute = _Compute("graph-slice-version")
    primary = {"id": "stable", "node_type": "FixtureRecord", "updatedAt": "7"}

    module.ingest_graph_slice(
        compute,
        "fixture-connector",
        [
            primary,
            {"id": "child", "node_type": "FixtureChild", "name": "before"},
        ],
    )
    module.ingest_graph_slice(
        compute,
        "fixture-connector",
        [
            primary,
            {"id": "child", "node_type": "FixtureChild", "name": "after"},
        ],
    )

    assert len(compute.client.changes.applied) == 2
    first, second = compute.client.changes.applied
    assert first["mutation"]["idempotency_key"] != second["mutation"]["idempotency_key"]
    assert compute.client.nodes.values["child"]["name"] == "after"


def test_stale_graph_version_rebuilds_and_retries_same_idempotency_key() -> None:
    compute = _Compute("graph-stale")
    compute.client.changes.failures.append(
        RuntimeError("STALE_GRAPH_VERSION: expected 0, current 7")
    )
    envelope = _envelope()

    result = module.ingest_envelope(compute, envelope)

    assert result["status"] == "success"
    first, second = compute.client.changes.applied
    assert first["mutation"]["expected_graph_version"] == 0
    assert second["mutation"]["expected_graph_version"] == 7
    assert first["mutation"]["idempotency_key"] == second["mutation"]["idempotency_key"]


def test_occ_contention_retries_beyond_four_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compute = _Compute("graph-contended")
    compute.client.changes.failures.extend(
        RuntimeError(
            f"STALE_VERSION: graph 'fixture-graph' expected version {version - 1} "
            f"but authoritative version is {version}"
        )
        for version in range(1, 7)
    )
    backoffs: list[int] = []
    monkeypatch.setattr(module, "_native_occ_backoff", backoffs.append)

    result = module.ingest_envelope(compute, _envelope())

    assert result["status"] == "success"
    applied = compute.client.changes.applied
    assert len(applied) == 7
    assert [row["mutation"]["expected_graph_version"] for row in applied] == list(
        range(7)
    )
    assert len({row["mutation"]["idempotency_key"] for row in applied}) == 1
    assert backoffs == list(range(6))
    assert module._NATIVE_GRAPH_VERSIONS[("fixture-tenant", "fixture-graph")] == 7


def test_occ_retry_budget_is_exact_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compute = _Compute("graph-exhausted")
    compute.client.changes.failures.extend(
        RuntimeError(f"STALE_GRAPH_VERSION: expected {version}, current {version + 1}")
        for version in range(module._NATIVE_OCC_MAX_ATTEMPTS)
    )
    backoffs: list[int] = []
    monkeypatch.setattr(module, "_native_occ_backoff", backoffs.append)

    result = module.ingest_envelope(compute, _envelope())

    assert result["status"] == "failed"
    assert result["error"] == "NativeChangeEnvelopeConflictExhausted"
    assert len(compute.client.changes.applied) == module._NATIVE_OCC_MAX_ATTEMPTS
    assert backoffs == list(range(module._NATIVE_OCC_MAX_ATTEMPTS - 1))


def test_non_conflict_failure_is_never_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compute = _Compute("graph-hard-failure")
    compute.client.changes.failures.append(RuntimeError("schema admission failed"))
    backoffs: list[int] = []
    monkeypatch.setattr(module, "_native_occ_backoff", backoffs.append)

    result = module.ingest_envelope(compute, _envelope())

    assert result["status"] == "failed"
    assert len(compute.client.changes.applied) == 1
    assert backoffs == []


def test_ambiguous_projection_race_is_never_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compute = _Compute("graph-ambiguous-projection")
    compute.client.changes.failures.append(
        RuntimeError(
            "ChangeEnvelope projection raced another write: "
            "expected version 3, current 4"
        )
    )
    backoffs: list[int] = []
    monkeypatch.setattr(module, "_native_occ_backoff", backoffs.append)

    result = module.ingest_envelope(compute, _envelope())

    assert result["status"] == "failed"
    assert len(compute.client.changes.applied) == 1
    assert backoffs == []


def test_commit_then_transport_error_recovers_without_retry_or_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compute = _Compute("graph-ambiguous-transport")
    apply = compute.client.changes.apply

    def _commit_then_disconnect(envelope: dict[str, object]):
        apply(envelope)
        raise ConnectionError("connection closed after commit")

    monkeypatch.setattr(compute.client.changes, "apply", _commit_then_disconnect)
    backoffs: list[int] = []
    monkeypatch.setattr(module, "_native_occ_backoff", backoffs.append)

    result = module.ingest_envelope(compute, _envelope())

    assert result["status"] == "skipped"
    assert len(compute.client.changes.applied) == 1
    assert backoffs == []


def test_occ_backoff_uses_capped_full_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ceilings: list[tuple[float, float]] = []
    sleeps: list[float] = []

    def _upper_bound(lower: float, upper: float) -> float:
        ceilings.append((lower, upper))
        return upper

    monkeypatch.setattr(module.random, "uniform", _upper_bound)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)

    for attempt in range(8):
        module._native_occ_backoff(attempt)

    expected = [0.001, 0.002, 0.004, 0.008, 0.01, 0.01, 0.01, 0.01]
    assert ceilings == [(0.0, value) for value in expected]
    assert sleeps == expected


def test_redelivery_uses_stable_native_identity_and_never_reapplies() -> None:
    compute = _Compute("graph-replay")
    access = ExternalAccess(is_public=False, read_roles=["kg:read"])
    first = _envelope(source_acl=access)
    second = _envelope(source_acl=access)
    assert first.envelope_id != second.envelope_id
    assert first.idempotency_key == second.idempotency_key

    assert module.ingest_envelope(compute, first)["status"] == "success"
    assert get_company_brain().permissions.get_acl("object-1") is not None
    reset_company_brain()
    assert get_company_brain().permissions.get_acl("object-1") is None
    replay = module.ingest_envelope(compute, second)

    assert replay["status"] == "skipped"
    assert len(compute.client.changes.applied) == 1
    native_id = compute.client.changes.applied[0]["envelope_id"]
    assert native_id == f"envelope:{first.idempotency_key}"
    replayed_acl = get_company_brain().permissions.get_acl("object-1")
    assert replayed_acl is not None
    assert replayed_acl.read_roles == ["kg:read"]


def test_snapshot_tombstones_and_cursor_share_the_native_commit() -> None:
    compute = _Compute("graph-snapshot")
    compute.client.nodes.values.update(
        {
            "gone": {"domain": "fixture-connector", "externalToolId": "gone"},
            "kept": {"domain": "fixture-connector", "externalToolId": "kept"},
        }
    )
    marker = ChangeEnvelope.snapshot_complete(
        connector="fixture-connector", live_ids=["kept"], checkpoint="2"
    )

    result = module.ingest_envelope(compute, marker)

    assert result["status"] == "success"
    assert result["write_result"]["tombstoned"] == 1
    assert compute.client.nodes.values["gone"]["archived"] is True
    native = compute.client.changes.applied[0]
    assert len(native["mutation"]["operations"]) == 2
    assert native["cursor"]["position"] == {"kind": "sequence", "value": 2}


def test_snapshot_reconciliation_is_source_scoped_and_empty_requires_approval() -> None:
    compute = _Compute("graph-source-snapshot")
    compute.client.nodes.values.update(
        {
            "alpha-gone": {
                "domain": "external-graph",
                "source_instance": "source-alpha",
                "externalToolId": "alpha-gone",
            },
            "beta-kept": {
                "domain": "external-graph",
                "source_instance": "source-beta",
                "externalToolId": "beta-kept",
            },
        }
    )
    unapproved = ChangeEnvelope.snapshot_complete(
        connector="external-graph",
        source_instance="source-alpha",
        live_ids=[],
        checkpoint="safe-cursor-1",
    )

    assert module.ingest_envelope(compute, unapproved)["status"] == "success"
    assert "archived" not in compute.client.nodes.values["alpha-gone"]
    assert "archived" not in compute.client.nodes.values["beta-kept"]

    approved = ChangeEnvelope.snapshot_complete(
        connector="external-graph",
        source_instance="source-alpha",
        live_ids=[],
        checkpoint="safe-cursor-2",
        provenance={"authoritative_empty_approved": True},
    )
    result = module.ingest_envelope(compute, approved)

    assert result["status"] == "success"
    assert result["write_result"]["tombstoned"] == 1
    assert compute.client.nodes.values["alpha-gone"]["archived"] is True
    assert "archived" not in compute.client.nodes.values["beta-kept"]


def test_missing_native_capability_fails_closed_without_write() -> None:
    compute = _Compute("graph-old", supported=False)

    result = module.ingest_envelope(compute, _envelope())

    assert result["status"] == "failed"
    assert result["error"] == "NativeChangeEnvelopeUnavailable"
    assert compute.client.changes.applied == []
    assert compute.client.nodes.values == {}


def test_shacl_rejection_never_materializes_connector_rows() -> None:
    compute = _Compute("graph-governance")
    compute.client.rdf.reports = [{"conforms": False, "results": [{}]}]

    result = module.ingest_envelope(compute, _envelope())

    assert result["status"] == "rejected"
    assert result["error"] == "ValueError"
    assert compute.client.changes.applied == []
    assert compute.client.nodes.values == {}


def test_missing_native_shacl_capability_fails_closed_before_write() -> None:
    compute = _Compute("graph-no-shacl")
    compute.client.rdf = object()

    result = module.ingest_envelope(compute, _envelope())

    assert result["status"] == "failed"
    assert result["error"] == "NativeChangeEnvelopeUnavailable"
    assert compute.client.changes.applied == []
    assert compute.client.nodes.values == {}
