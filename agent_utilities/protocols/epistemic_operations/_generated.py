"""Generated strict Epistemic Operations Protocol client projections.

JSON Schema is authoritative. Regenerate with the protocol gate; do not edit.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_NAME = "epistemic-operations"
PROTOCOL_VERSION = "1"
CATALOG_SHA256 = "280b7c386bf71e586788bbdaeb72650c6723996f147507c56ab35f96a7c0b98a"
SCHEMA_VERSION = {
    "request_context": "2",
    "mutation_batch": "1",
    "change_envelope": "1",
    "work_item": "1",
    "artifact": "1",
    "knowledge_batch": "1",
    "analytics_job": "1",
    "trace_outcome": "1",
    "placement_route": "1",
    "claim_work_item": "1",
    "evidence_bundle": "1",
    "operation_result": "1",
}
SCHEMA_SHA256 = {
    "request_context": "310b69c8113fd441c99285053e307e84a12e373c32cf56182c5f25273133cd14",
    "mutation_batch": "90f32151cbc2df050dbb031c998e6a08468b87fdf4cbd0612c39551db25ed3ce",
    "change_envelope": "7b12ee69be0d716499c5f5819af12f9f8db9565a0a820bf6d9fd45aa40fdb3fe",
    "work_item": "53a7f6a1b435d554a76acede91e11eb869210eb75884f7d90232b95fcf5dbcb4",
    "artifact": "519c760a226ae93ebf346e29e30aeb03f4f6ce4e632bffeb9bf8302b392b6b80",
    "knowledge_batch": "ec2cf5afc5b7fa3ae0469ed394258f2e4176a310095196db993cf4cea11a9e6f",
    "analytics_job": "d109b74f0fa3013f4cc5b3c6e4df9c82b6fa6e6ce4c32af25b9a4ac0f7ee609c",
    "trace_outcome": "34740b30955cebda79ea0c2b5ba5217729ce9e544a7d7264fc6524e7636c5025",
    "placement_route": "0637a2b5ed9631212d5e3b16a6ca3010bd5993ff42336bc995372a163776d16c",
    "claim_work_item": "e2e045c0784994023ad3c6a6a91a64aa613ad4271149a88dbf5b569ae842b237",
    "evidence_bundle": "894fb20b93c3264652c390d0a778527389fe1ada5958e753c7bc80148e24b9d4",
    "operation_result": "76e3ee6c19843968d7e40d6a56aefc1a595935451c1f5eaee370adc78e9a4966",
}


class ProtocolModel(BaseModel):
    """Fail-closed base for every generated protocol DTO."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RequestContext(ProtocolModel):
    """Schema-generated strict projection."""

    schema_version: Literal["2"]
    request_id: Annotated[str, Field(min_length=1)]
    subject_id: Annotated[str, Field(min_length=1)]
    tenant_id: Annotated[str, Field(min_length=1)]
    agent_id: Annotated[str, Field(min_length=1)]
    scopes: list[Annotated[str, Field(min_length=1)]]
    audience: Annotated[str, Field(min_length=1)]
    authentication_method: Literal[
        "workload_identity", "oidc", "mutual_tls", "local_process"
    ]
    policy_version: Annotated[str, Field(min_length=1)]
    graph: Annotated[str, Field(min_length=1)]
    placement_epoch: Annotated[int, Field(ge=0)] | None
    trace_id: Annotated[str, Field(min_length=1)]
    issued_at_ms: Annotated[int, Field(ge=0)]
    expires_at_ms: Annotated[int, Field(ge=0)]


class MutationBatch(ProtocolModel):
    """Schema-generated strict projection."""

    schema_version: Literal["1"]
    batch_id: Annotated[str, Field(min_length=1)]
    context: RequestContext
    graph: Annotated[str, Field(min_length=1)]
    placement_epoch: Annotated[int, Field(ge=0)] | None
    expected_graph_version: Annotated[int, Field(ge=0)] | None
    idempotency_key: Annotated[str, Field(min_length=1)]
    operations: list[MutationOperation]
    submitted_at_ms: Annotated[int, Field(ge=0)]


class MutationOperation(ProtocolModel):
    """Schema-generated strict projection."""

    operation_id: Annotated[str, Field(min_length=1)]
    domain: Literal[
        "graph", "rdf", "vector", "timeseries", "artifact", "job", "work_item"
    ]
    action: Literal["upsert", "delete", "append", "transition", "link", "unlink"]
    target_id: Annotated[str, Field(min_length=1)]
    payload_ref: Annotated[str, Field(min_length=1)] | None
    payload_digest: Annotated[str, Field(pattern="^sha256:[0-9a-f]{64}$")] | None
    expected_version: Annotated[int, Field(ge=0)] | None


class ChangeEnvelope(ProtocolModel):
    """Schema-generated strict projection."""

    schema_version: Literal["1"]
    envelope_id: Annotated[str, Field(min_length=1)]
    context: RequestContext
    connector_kind: Annotated[str, Field(min_length=1)]
    source_instance_id: Annotated[str, Field(min_length=1)]
    source_object_id: Annotated[str, Field(min_length=1)]
    source_version: Annotated[str, Field(min_length=1)]
    operation: Literal["upsert", "delete", "snapshot_complete"]
    schema_id: Annotated[str, Field(min_length=1)]
    event_time_ms: Annotated[int, Field(ge=0)] | None
    valid_time_ms: Annotated[int, Field(ge=0)] | None
    observed_time_ms: Annotated[int, Field(ge=0)]
    artifact_refs: list[Annotated[str, Field(min_length=1)]]
    payload_ref: Annotated[str, Field(min_length=1)] | None
    payload_digest: Annotated[str, Field(pattern="^sha256:[0-9a-f]{64}$")] | None
    access: SourceAccess
    provenance_refs: list[Annotated[str, Field(min_length=1)]]
    checkpoint_ref: Annotated[str, Field(min_length=1)] | None
    idempotency_key: Annotated[str, Field(min_length=1)]


class SourceAccess(ProtocolModel):
    """Schema-generated strict projection."""

    classification: Literal[
        "public", "internal", "confidential", "restricted", "regulated"
    ]
    read_scopes: list[Annotated[str, Field(min_length=1)]]
    purpose_tags: list[Annotated[str, Field(min_length=1)]]
    retention_policy_id: Annotated[str, Field(min_length=1)] | None
    legal_hold: bool


class WorkItem(ProtocolModel):
    """Schema-generated strict projection."""

    schema_version: Literal["1"]
    work_item_id: Annotated[str, Field(min_length=1)]
    context: RequestContext
    kind: Annotated[str, Field(min_length=1)]
    state: Literal[
        "submitted",
        "ready",
        "leased",
        "running",
        "succeeded",
        "failed",
        "cancelled",
        "dead_letter",
    ]
    priority: int
    depends_on: list[Annotated[str, Field(min_length=1)]]
    input_artifact_refs: list[Annotated[str, Field(min_length=1)]]
    output_artifact_refs: list[Annotated[str, Field(min_length=1)]]
    lease_holder_id: Annotated[str, Field(min_length=1)] | None
    lease_expires_at_ms: Annotated[int, Field(ge=0)] | None
    attempt: Annotated[int, Field(ge=0)]
    max_attempts: Annotated[int, Field(ge=1)]
    created_at_ms: Annotated[int, Field(ge=0)]
    updated_at_ms: Annotated[int, Field(ge=0)]
    idempotency_key: Annotated[str, Field(min_length=1)]


class Artifact(ProtocolModel):
    """Schema-generated strict projection."""

    schema_version: Literal["1"]
    artifact_id: Annotated[str, Field(min_length=1)]
    tenant_id: Annotated[str, Field(min_length=1)]
    media_type: Annotated[str, Field(min_length=1)]
    digest: Annotated[str, Field(pattern="^sha256:[0-9a-f]{64}$")]
    byte_length: Annotated[int, Field(ge=0)]
    content_ref: Annotated[str, Field(min_length=1)]
    classification: Literal[
        "public", "internal", "confidential", "restricted", "regulated"
    ]
    provenance_refs: list[Annotated[str, Field(min_length=1)]]
    occurrence_ids: list[Annotated[str, Field(min_length=1)]]
    rendition_ids: list[Annotated[str, Field(min_length=1)]]
    segment_ids: list[Annotated[str, Field(min_length=1)]]
    feature_ids: list[Annotated[str, Field(min_length=1)]]
    derivation_ids: list[Annotated[str, Field(min_length=1)]]
    loci: list[ArtifactLocus]
    created_at_ms: Annotated[int, Field(ge=0)]


class ArtifactLocus(ProtocolModel):
    """Schema-generated strict projection."""

    kind: Literal[
        "document_span",
        "table_cell_range",
        "image_region",
        "page_box",
        "audio_segment",
        "video_frame_range",
        "metric_window",
        "row_version",
    ]
    start: Annotated[int, Field(ge=0)] | None
    end: Annotated[int, Field(ge=0)] | None
    selector: dict[str, Any]


class KnowledgeBatch(ProtocolModel):
    """Schema-generated strict projection."""

    schema_version: Literal["1"]
    batch_id: Annotated[str, Field(min_length=1)]
    context: RequestContext
    fields: list[KnowledgeField]
    encoding: Literal["json_rows", "arrow_ipc"]
    rows: list[list[Any]]
    data_ref: Annotated[str, Field(min_length=1)] | None
    cursor: Annotated[str, Field(min_length=1)] | None
    end_of_stream: bool
    source_refs: list[Annotated[str, Field(min_length=1)]]


class KnowledgeField(ProtocolModel):
    """Schema-generated strict projection."""

    name: Annotated[str, Field(min_length=1)]
    data_type: Literal[
        "null", "boolean", "i64", "u64", "f64", "utf8", "binary", "timestamp_ms", "json"
    ]
    nullable: bool


class AnalyticsJob(ProtocolModel):
    """Schema-generated strict projection."""

    schema_version: Literal["1"]
    job_id: Annotated[str, Field(min_length=1)]
    context: RequestContext
    kind: Annotated[str, Field(min_length=1)]
    state: Literal["submitted", "running", "succeeded", "failed", "cancelled"]
    input_artifact_refs: list[Annotated[str, Field(min_length=1)]]
    parameters_digest: Annotated[str, Field(pattern="^sha256:[0-9a-f]{64}$")]
    algorithm: Annotated[str, Field(min_length=1)]
    algorithm_version: Annotated[str, Field(min_length=1)]
    checkpoint_ref: Annotated[str, Field(min_length=1)] | None
    output_artifact_refs: list[Annotated[str, Field(min_length=1)]]
    progress: Annotated[float, Field(ge=0, le=1)]
    attempt: Annotated[int, Field(ge=0)]
    max_attempts: Annotated[int, Field(ge=1)]
    created_at_ms: Annotated[int, Field(ge=0)]
    updated_at_ms: Annotated[int, Field(ge=0)]
    error: AnalyticsError | None


class AnalyticsError(ProtocolModel):
    """Schema-generated strict projection."""

    code: Annotated[str, Field(min_length=1)]
    retryable: bool
    detail_ref: Annotated[str, Field(min_length=1)] | None


class TraceOutcome(ProtocolModel):
    """Schema-generated strict projection."""

    schema_version: Literal["1"]
    outcome_id: Annotated[str, Field(min_length=1)]
    trace_id: Annotated[str, Field(min_length=1)]
    context_id: Annotated[str, Field(min_length=1)]
    operation: Annotated[str, Field(min_length=1)]
    status: Literal["succeeded", "failed", "cancelled", "denied"]
    started_at_ms: Annotated[int, Field(ge=0)]
    ended_at_ms: Annotated[int, Field(ge=0)]
    input_artifact_refs: list[Annotated[str, Field(min_length=1)]]
    output_artifact_refs: list[Annotated[str, Field(min_length=1)]]
    metrics: dict[str, float]
    policy_decision: Literal["allow", "deny", "not_applicable"]
    error_code: Annotated[str, Field(min_length=1)] | None
    error_detail_ref: Annotated[str, Field(min_length=1)] | None
    evaluation_scores: dict[str, float]


class PlacementRoute(ProtocolModel):
    """Schema-generated strict projection."""

    schema_version: Literal["1"]
    route_id: Annotated[str, Field(min_length=1)]
    tenant_ref: Annotated[str, Field(min_length=1)]
    partition_ref: Annotated[str, Field(min_length=1)]
    authoritative: Literal[True]
    placed: bool
    group: Annotated[int, Field(ge=0)]
    epoch: Annotated[int, Field(ge=0)]
    fencing_token: Annotated[int, Field(ge=0)]
    stale: bool
    leader_ref: Annotated[str, Field(min_length=1)] | None


class PlacementRouteRequest(ProtocolModel):
    """Schema-generated strict projection."""

    schema_version: Literal["1"]
    tenant_ref: Annotated[str, Field(min_length=1)]
    partition_ref: Annotated[str, Field(min_length=1)]
    client_epoch: Annotated[int, Field(ge=0)]


class ClaimWorkItemRequest(ProtocolModel):
    """Schema-generated strict projection."""

    schema_version: Literal["1"]
    tenant_ref: Annotated[str, Field(min_length=1)]
    work_item_id: Annotated[str, Field(min_length=1)] | None
    queue_ref: Annotated[str, Field(min_length=1)] | None
    resource_class: Annotated[str, Field(min_length=1)] | None
    fairness_group: Annotated[str, Field(min_length=1)] | None
    worker_ref: Annotated[str, Field(min_length=1)]
    now_ms: Annotated[int, Field(ge=0)]
    lease_ms: Annotated[int, Field(ge=1)]
    max_tenant_in_flight: Annotated[int, Field(ge=1, le=4096)]


class ClaimWorkItemResult(ProtocolModel):
    """Schema-generated strict projection."""

    schema_version: Literal["1"]
    claimed: bool
    reason: Literal["claimed", "empty", "tenant_quota"]
    work_item_id: Annotated[str, Field(min_length=1)] | None
    kind: Annotated[str, Field(min_length=1)] | None
    payload_ref: Annotated[str, Field(min_length=1)] | None
    lease_holder_ref: Annotated[str, Field(min_length=1)] | None
    lease_epoch: Annotated[int, Field(ge=0)] | None
    fencing_token: Annotated[int, Field(ge=0)] | None
    lease_expires_at_ms: Annotated[int, Field(ge=0)] | None
    attempt: Annotated[int, Field(ge=0)] | None
    max_attempts: Annotated[int, Field(ge=1)] | None
    tenant_in_flight: Annotated[int, Field(ge=0)] | None
    changed_work_item_ids: list[Annotated[str, Field(min_length=1)]]


class EvidenceBundle(ProtocolModel):
    """Schema-generated strict projection."""

    schema_version: Literal["1"]
    bundle_id: Annotated[str, Field(min_length=1)]
    resolved: bool
    answer_ref: Annotated[str, Field(min_length=1)] | None
    claims: list[EvidenceClaim]
    policy_exclusions: list[Annotated[str, Field(min_length=1)]]
    next_action_refs: list[Annotated[str, Field(min_length=1)]]


class EvidenceClaim(ProtocolModel):
    """Schema-generated strict projection."""

    claim_ref: Annotated[str, Field(min_length=1)]
    kind: str
    score: float | None
    confidence: Annotated[float, Field(ge=0, le=1)]
    valid_time: EvidenceTimeRange
    transaction_time: EvidenceTimeRange
    source_refs: list[Annotated[str, Field(min_length=1)]]
    evidence_locus_refs: list[Annotated[str, Field(min_length=1)]]
    contradiction_refs: list[Annotated[str, Field(min_length=1)]]
    proof_refs: list[Annotated[str, Field(min_length=1)]]
    policy_labels: list[Annotated[str, Field(min_length=1)]]


class EvidenceTimeRange(ProtocolModel):
    """Schema-generated strict projection."""

    start_ms: Annotated[int, Field(ge=0)] | None
    end_ms: Annotated[int, Field(ge=0)] | None


class OperationResult(ProtocolModel):
    """Schema-generated strict projection."""

    schema_version: Literal["1"]
    operation_id: Annotated[str, Field(min_length=1)]
    status: Literal["succeeded", "failed", "redirected"]
    result_kind: Annotated[str, Field(min_length=1)] | None
    result_ref: Annotated[str, Field(min_length=1)] | None
    error: OperationError | None
    redirect: OperationRedirect | None


class OperationError(ProtocolModel):
    """Schema-generated strict projection."""

    code: Annotated[str, Field(min_length=1)]
    retryable: bool
    correlation_id: Annotated[str, Field(min_length=1)]
    detail_ref: Annotated[str, Field(min_length=1)] | None


class OperationRedirect(ProtocolModel):
    """Schema-generated strict projection."""

    kind: Literal["placement"]
    target_ref: Annotated[str, Field(min_length=1)]
    group: Annotated[int, Field(ge=0)]
    epoch: Annotated[int, Field(ge=0)]
    fencing_token: Annotated[int, Field(ge=0)]
    leader_ref: Annotated[str, Field(min_length=1)] | None


BINDING_FIELDS = {
    "RequestContext": (
        "schema_version",
        "request_id",
        "subject_id",
        "tenant_id",
        "agent_id",
        "scopes",
        "audience",
        "authentication_method",
        "policy_version",
        "graph",
        "placement_epoch",
        "trace_id",
        "issued_at_ms",
        "expires_at_ms",
    ),
    "MutationBatch": (
        "schema_version",
        "batch_id",
        "context",
        "graph",
        "placement_epoch",
        "expected_graph_version",
        "idempotency_key",
        "operations",
        "submitted_at_ms",
    ),
    "MutationOperation": (
        "operation_id",
        "domain",
        "action",
        "target_id",
        "payload_ref",
        "payload_digest",
        "expected_version",
    ),
    "ChangeEnvelope": (
        "schema_version",
        "envelope_id",
        "context",
        "connector_kind",
        "source_instance_id",
        "source_object_id",
        "source_version",
        "operation",
        "schema_id",
        "event_time_ms",
        "valid_time_ms",
        "observed_time_ms",
        "artifact_refs",
        "payload_ref",
        "payload_digest",
        "access",
        "provenance_refs",
        "checkpoint_ref",
        "idempotency_key",
    ),
    "SourceAccess": (
        "classification",
        "read_scopes",
        "purpose_tags",
        "retention_policy_id",
        "legal_hold",
    ),
    "WorkItem": (
        "schema_version",
        "work_item_id",
        "context",
        "kind",
        "state",
        "priority",
        "depends_on",
        "input_artifact_refs",
        "output_artifact_refs",
        "lease_holder_id",
        "lease_expires_at_ms",
        "attempt",
        "max_attempts",
        "created_at_ms",
        "updated_at_ms",
        "idempotency_key",
    ),
    "Artifact": (
        "schema_version",
        "artifact_id",
        "tenant_id",
        "media_type",
        "digest",
        "byte_length",
        "content_ref",
        "classification",
        "provenance_refs",
        "occurrence_ids",
        "rendition_ids",
        "segment_ids",
        "feature_ids",
        "derivation_ids",
        "loci",
        "created_at_ms",
    ),
    "ArtifactLocus": (
        "kind",
        "start",
        "end",
        "selector",
    ),
    "KnowledgeBatch": (
        "schema_version",
        "batch_id",
        "context",
        "fields",
        "encoding",
        "rows",
        "data_ref",
        "cursor",
        "end_of_stream",
        "source_refs",
    ),
    "KnowledgeField": (
        "name",
        "data_type",
        "nullable",
    ),
    "AnalyticsJob": (
        "schema_version",
        "job_id",
        "context",
        "kind",
        "state",
        "input_artifact_refs",
        "parameters_digest",
        "algorithm",
        "algorithm_version",
        "checkpoint_ref",
        "output_artifact_refs",
        "progress",
        "attempt",
        "max_attempts",
        "created_at_ms",
        "updated_at_ms",
        "error",
    ),
    "AnalyticsError": (
        "code",
        "retryable",
        "detail_ref",
    ),
    "TraceOutcome": (
        "schema_version",
        "outcome_id",
        "trace_id",
        "context_id",
        "operation",
        "status",
        "started_at_ms",
        "ended_at_ms",
        "input_artifact_refs",
        "output_artifact_refs",
        "metrics",
        "policy_decision",
        "error_code",
        "error_detail_ref",
        "evaluation_scores",
    ),
    "PlacementRoute": (
        "schema_version",
        "route_id",
        "tenant_ref",
        "partition_ref",
        "authoritative",
        "placed",
        "group",
        "epoch",
        "fencing_token",
        "stale",
        "leader_ref",
    ),
    "PlacementRouteRequest": (
        "schema_version",
        "tenant_ref",
        "partition_ref",
        "client_epoch",
    ),
    "ClaimWorkItemRequest": (
        "schema_version",
        "tenant_ref",
        "work_item_id",
        "queue_ref",
        "resource_class",
        "fairness_group",
        "worker_ref",
        "now_ms",
        "lease_ms",
        "max_tenant_in_flight",
    ),
    "ClaimWorkItemResult": (
        "schema_version",
        "claimed",
        "reason",
        "work_item_id",
        "kind",
        "payload_ref",
        "lease_holder_ref",
        "lease_epoch",
        "fencing_token",
        "lease_expires_at_ms",
        "attempt",
        "max_attempts",
        "tenant_in_flight",
        "changed_work_item_ids",
    ),
    "EvidenceBundle": (
        "schema_version",
        "bundle_id",
        "resolved",
        "answer_ref",
        "claims",
        "policy_exclusions",
        "next_action_refs",
    ),
    "EvidenceClaim": (
        "claim_ref",
        "kind",
        "score",
        "confidence",
        "valid_time",
        "transaction_time",
        "source_refs",
        "evidence_locus_refs",
        "contradiction_refs",
        "proof_refs",
        "policy_labels",
    ),
    "EvidenceTimeRange": (
        "start_ms",
        "end_ms",
    ),
    "OperationResult": (
        "schema_version",
        "operation_id",
        "status",
        "result_kind",
        "result_ref",
        "error",
        "redirect",
    ),
    "OperationError": (
        "code",
        "retryable",
        "correlation_id",
        "detail_ref",
    ),
    "OperationRedirect": (
        "kind",
        "target_ref",
        "group",
        "epoch",
        "fencing_token",
        "leader_ref",
    ),
}
