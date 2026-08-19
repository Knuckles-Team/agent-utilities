"""Strict, bounded models for the Arrow data-preparation kernel.

The models are deliberately independent of Arrow.  A plan is validated before
an input table is inspected, and its allow-listed verbs are dispatched by the
kernel with explicit ``isinstance`` branches.  ``ProtocolModel`` is the
existing fail-closed protocol base: unknown fields, primitive coercion and
post-construction assignment are rejected.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from agent_utilities.protocols.epistemic_operations import ProtocolModel

SchemaVersion: TypeAlias = Literal["1"]
InvalidRowDisposition: TypeAlias = Literal["fail", "quarantine"]
ArrowTypeName: TypeAlias = Literal[
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "string",
]
ColumnName: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[^\x00]+$"),
]
OpaqueReference: TypeAlias = Annotated[
    str,
    Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    ),
]
Digest: TypeAlias = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
ScalarValue: TypeAlias = StrictBool | StrictInt | StrictFloat | StrictStr
ProfileScalar: TypeAlias = StrictBool | StrictInt | StrictFloat | StrictStr
ProfileTargetKind: TypeAlias = Literal[
    "local_arrow",
    "engine_table",
    "engine_rowset",
    "engine_subgraph",
]
ProfileWarningCode: TypeAlias = Literal[
    "empty_dataset",
    "all_null",
    "high_null_rate",
    "single_value",
    "high_cardinality",
    "cardinality_unavailable",
    "small_group_suppressed",
    "stats_suppressed",
    "topk_suppressed",
    "topk_truncated",
    "quantiles_unavailable",
    "quantiles_truncated",
    "deadline_exceeded",
    "profile_truncated",
]


class LocalProfile(ProtocolModel):
    """Explicit limits for one local, in-memory Arrow operation.

    ``max_bytes`` bounds the Arrow table footprint before any transformation.
    The other limits bound rows, columns, plan steps and redacted quarantine
    outcomes.  The effective quarantine ceiling is the lower of
    ``max_quarantine_rows`` and ``max_rows``.  The defaults are finite and
    conservative; callers still have to choose the row/column/step ceilings
    for each plan.
    """

    max_rows: int = Field(ge=1, le=1_000_000)
    max_columns: int = Field(ge=1, le=1_024)
    max_steps: int = Field(ge=1, le=64)
    max_bytes: int = Field(default=64 * 1024 * 1024, ge=1, le=512 * 1024 * 1024)
    max_quarantine_rows: int = Field(default=100_000, ge=0, le=1_000_000)
    max_outcome_rows: int = Field(default=100_000, ge=1, le=200_000)


class CanonicalNames(ProtocolModel):
    """Canonicalize every input field name to a deterministic safe identifier."""

    verb: Literal["canonical_names"]


class NullPolicy(ProtocolModel):
    """Declare how nulls in selected columns are treated."""

    verb: Literal["null_policy"]
    columns: list[ColumnName] = Field(min_length=1, max_length=1_024)
    action: Literal["allow", "reject"]

    @model_validator(mode="after")
    def columns_are_unique(self) -> NullPolicy:
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("null-policy columns must be unique")
        return self


class SafeCast(ProtocolModel):
    """Request one explicitly declared, lossless Arrow type widening."""

    verb: Literal["safe_cast"]
    column: ColumnName
    source_type: ArrowTypeName
    target_type: ArrowTypeName


class Dedupe(ProtocolModel):
    """Remove repeated rows using explicitly named key columns."""

    verb: Literal["dedupe"]
    keys: list[ColumnName] = Field(min_length=1, max_length=1_024)

    @model_validator(mode="after")
    def keys_are_unique(self) -> Dedupe:
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("dedupe keys must be unique and explicit")
        return self


class FillNulls(ProtocolModel):
    """Fill selected nulls with typed scalar values supplied by the caller."""

    verb: Literal["fill_nulls"]
    fills: dict[ColumnName, ScalarValue] = Field(min_length=1, max_length=1_024)


CleanStep: TypeAlias = Annotated[
    CanonicalNames | NullPolicy | SafeCast | Dedupe | FillNulls,
    Field(discriminator="verb"),
]

STEP_VERBS: tuple[str, ...] = (
    "canonical_names",
    "null_policy",
    "safe_cast",
    "dedupe",
    "fill_nulls",
)


class CleanPlan(ProtocolModel):
    """Versioned, immutable plan for the pure Arrow cleaning kernel."""

    schema_version: SchemaVersion
    steps: list[CleanStep] = Field(min_length=1, max_length=64)
    profile: LocalProfile
    invalid_row_disposition: InvalidRowDisposition
    plan_ref: OpaqueReference
    policy_ref: OpaqueReference
    model_ref: OpaqueReference
    model_digest: Digest | None = None
    source_ref: OpaqueReference | None = None
    artifact_ref: OpaqueReference | None = None

    @model_validator(mode="after")
    def plan_is_bounded(self) -> CleanPlan:
        if len(self.steps) > self.profile.max_steps:
            raise ValueError("clean plan exceeds the local profile step limit")
        return self


class QuarantineOutcome(ProtocolModel):
    """Privacy-safe count for one rejected-row reason; never a row payload."""

    reason_code: Literal["null_rejected", "model_rejected"]
    row_count: int = Field(ge=1, le=1_000_000)


class RowOutcome(ProtocolModel):
    """Terminal row disposition with no row payload or validation detail."""

    row_index: int = Field(ge=0, le=1_000_000)
    status: Literal["accepted", "dropped", "quarantined"]
    reason_code: Literal[
        "accepted",
        "deduplicated",
        "null_rejected",
        "model_rejected",
    ]

    @model_validator(mode="after")
    def status_matches_reason(self) -> RowOutcome:
        if self.status == "accepted" and self.reason_code != "accepted":
            raise ValueError("accepted rows must use the accepted reason code")
        if self.status == "dropped" and self.reason_code != "deduplicated":
            raise ValueError("dropped rows must use the deduplicated reason code")
        if self.status == "quarantined" and self.reason_code not in {
            "null_rejected",
            "model_rejected",
        }:
            raise ValueError("quarantined rows require a rejection reason code")
        return self


class StepEvidence(ProtocolModel):
    """Bounded, content-free accounting for one executed plan verb."""

    verb: Literal[
        "canonical_names",
        "null_policy",
        "safe_cast",
        "dedupe",
        "fill_nulls",
    ]
    rows_in: int = Field(ge=0, le=1_000_000)
    rows_out: int = Field(ge=0, le=1_000_000)
    dropped_rows: int = Field(ge=0, le=1_000_000)
    quarantined_rows: int = Field(ge=0, le=1_000_000)


class PrepEvidence(ProtocolModel):
    """Versioned, privacy-safe evidence for one completed Arrow preparation.

    The evidence intentionally contains only schema/plan/model digests, counts,
    policy/reference identifiers and allow-listed reason codes.  It cannot
    carry rows, rejected values, fill values, column names or secrets.  A
    quarantined result is never checkpoint-eligible.
    """

    evidence_version: Literal["data-prep-evidence.v1"]
    algorithm: Literal["arrow-clean-pipeline"]
    algorithm_version: Literal["1"]
    outcome: Literal["complete", "quarantined"]
    checkpoint_eligible: bool
    plan_digest: Digest
    input_schema_digest: Digest
    output_schema_digest: Digest
    rows_in: int = Field(ge=0, le=1_000_000)
    rows_out: int = Field(ge=0, le=1_000_000)
    dropped_rows: int = Field(ge=0, le=1_000_000)
    quarantined_rows: int = Field(ge=0, le=1_000_000)
    invalid_row_disposition: InvalidRowDisposition
    plan_ref: OpaqueReference
    policy_ref: OpaqueReference
    model_ref: OpaqueReference
    model_digest: Digest
    source_ref: OpaqueReference | None = None
    artifact_ref: OpaqueReference | None = None
    quarantine: list[QuarantineOutcome] = Field(default_factory=list)
    row_outcomes: list[RowOutcome] = Field(
        default_factory=list,
        max_length=200_000,
    )
    model_rejected_rows: int = Field(ge=0, le=1_000_000)
    steps: list[StepEvidence] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def checkpoint_follows_outcome(self) -> PrepEvidence:
        counted = sum(item.row_count for item in self.quarantine)
        if counted != self.quarantined_rows:
            raise ValueError("quarantine counts must equal quarantined_rows")
        if self.rows_out + self.dropped_rows + self.quarantined_rows != self.rows_in:
            raise ValueError("terminal row counts must cover every input row")
        row_indexes = [item.row_index for item in self.row_outcomes]
        if len(row_indexes) != self.rows_in or sorted(row_indexes) != list(
            range(self.rows_in)
        ):
            raise ValueError("row outcomes must cover each input ordinal exactly once")
        counted_accepted = sum(item.status == "accepted" for item in self.row_outcomes)
        if counted_accepted != self.rows_out:
            raise ValueError("accepted outcomes must equal rows_out")
        counted_dropped = sum(item.status == "dropped" for item in self.row_outcomes)
        if counted_dropped != self.dropped_rows:
            raise ValueError("dropped outcomes must equal dropped_rows")
        counted_outcomes = sum(
            item.status == "quarantined" for item in self.row_outcomes
        )
        if counted_outcomes != self.quarantined_rows:
            raise ValueError("row outcomes must account for every quarantined row")
        counted_model_rejections = sum(
            item.reason_code == "model_rejected" for item in self.row_outcomes
        )
        if counted_model_rejections != self.model_rejected_rows:
            raise ValueError("model rejection count must match row outcomes")
        if self.outcome == "quarantined":
            if self.quarantined_rows < 1 or self.checkpoint_eligible:
                raise ValueError("quarantined preparation is never checkpoint-eligible")
        elif self.quarantined_rows != 0 or not self.checkpoint_eligible:
            raise ValueError("complete preparation cannot contain quarantine outcomes")
        return self


class ProfileTarget(ProtocolModel):
    """Exact identity of the data snapshot being profiled."""

    target_kind: ProfileTargetKind
    target_ref: OpaqueReference
    schema_digest: Digest
    as_of_lsn: StrictInt | None = Field(default=None, ge=0)


class ProfileSelector(ProtocolModel):
    """Ordinal column selector pinned to one exact schema digest."""

    schema_digest: Digest
    ordinals: list[StrictInt] = Field(min_length=1, max_length=1_024)

    @model_validator(mode="after")
    def ordinals_are_unique_and_sorted(self) -> ProfileSelector:
        if any(ordinal < 0 for ordinal in self.ordinals):
            raise ValueError("profile selector ordinals must be non-negative")
        if len(set(self.ordinals)) != len(self.ordinals):
            raise ValueError("profile selector ordinals must be unique")
        if self.ordinals != sorted(self.ordinals):
            raise ValueError("profile selector ordinals must be sorted")
        return self


class ProfileLimits(ProtocolModel):
    """Finite local/remote profiling budget; no implicit unbounded work."""

    max_columns: StrictInt = Field(default=1_024, ge=1, le=1_024)
    max_cardinality: StrictInt = Field(default=100_000, ge=1, le=1_000_000)
    max_bytes: StrictInt = Field(
        default=64 * 1024 * 1024,
        ge=1,
        le=512 * 1024 * 1024,
    )
    deadline_ms: StrictInt = Field(default=30_000, ge=1, le=120_000)
    max_top_k: StrictInt = Field(default=10, ge=0, le=20)
    max_quantiles: StrictInt = Field(default=3, ge=0, le=9)
    disclosure_threshold: StrictInt = Field(default=2, ge=2, le=1_000_000)
    max_warnings: StrictInt = Field(default=32, ge=1, le=64)


class QuantilePoint(ProtocolModel):
    """A fixed probability/value pair from a numeric column."""

    probability: StrictFloat = Field(ge=0.0, le=1.0)
    value: StrictFloat

    @model_validator(mode="after")
    def values_are_finite(self) -> QuantilePoint:
        if not math.isfinite(self.probability) or not math.isfinite(self.value):
            raise ValueError("profile quantiles must be finite")
        return self


class TopKEntry(ProtocolModel):
    """A disclosed value whose frequency meets the privacy threshold."""

    value: ProfileScalar
    count: StrictInt = Field(ge=1, le=1_000_000)

    @model_validator(mode="after")
    def value_is_finite(self) -> TopKEntry:
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("profile top-k values must be finite")
        return self


class ColumnProfile(ProtocolModel):
    """Bounded, privacy-aware quality signals for one ordinal field."""

    ordinal: StrictInt = Field(ge=0, le=1_023)
    dtype: StrictStr = Field(min_length=1, max_length=128)
    null_count: StrictInt = Field(ge=0, le=1_000_000)
    null_rate: StrictFloat = Field(ge=0.0, le=1.0)
    distinct_count: StrictInt | None = Field(default=None, ge=0, le=1_000_000)
    min_value: StrictInt | StrictFloat | None = None
    max_value: StrictInt | StrictFloat | None = None
    mean: StrictFloat | None = None
    quantiles: list[QuantilePoint] = Field(default_factory=list, max_length=9)
    top_k: list[TopKEntry] = Field(default_factory=list, max_length=20)
    warning_codes: list[ProfileWarningCode] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def numeric_values_are_finite(self) -> ColumnProfile:
        numbers = (self.min_value, self.max_value, self.mean, self.null_rate)
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in numbers
        ):
            raise ValueError("profile numeric values must be finite")
        return self


class ProfileResult(ProtocolModel):
    """The one typed result shared by local Arrow and engine profiling."""

    evidence_version: Literal["data-prep-profile.v1"]
    algorithm: Literal["arrow-local-profile", "epistemic-graph-profile"]
    algorithm_version: Literal["1"]
    target: ProfileTarget
    schema_digest: Digest
    as_of_lsn: StrictInt | None = Field(default=None, ge=0)
    rows: StrictInt = Field(ge=0, le=1_000_000)
    columns: StrictInt = Field(ge=0, le=1_024)
    null_cells: StrictInt = Field(ge=0, le=1_000_000_000)
    bytes_in_memory: StrictInt = Field(ge=0, le=512 * 1024 * 1024)
    disclosure_threshold: StrictInt = Field(ge=2, le=1_000_000)
    truncated: bool = False
    warnings: list[ProfileWarningCode] = Field(default_factory=list, max_length=64)
    column_profiles: list[ColumnProfile] = Field(
        min_length=0,
        max_length=1_024,
    )

    @model_validator(mode="after")
    def identity_is_exact(self) -> ProfileResult:
        if self.target.schema_digest != self.schema_digest:
            raise ValueError("profile target and result schema digests differ")
        if self.target.as_of_lsn != self.as_of_lsn:
            raise ValueError("profile target and result LSNs differ")
        ordinals = [item.ordinal for item in self.column_profiles]
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("profile column ordinals must be unique")
        if self.columns != len(self.column_profiles):
            raise ValueError("profile column count does not match column profiles")
        if self.null_cells != sum(item.null_count for item in self.column_profiles):
            raise ValueError("profile null cells do not match column profiles")
        for column in self.column_profiles:
            if column.null_count > self.rows:
                raise ValueError("profile null count exceeds row count")
            expected_null_rate = (
                column.null_count / self.rows if self.rows else 0.0
            )
            if abs(column.null_rate - expected_null_rate) > 1e-6:
                raise ValueError("profile null rate does not match null count")
            if column.distinct_count is not None and column.distinct_count > (
                self.rows - column.null_count
            ):
                raise ValueError("profile cardinality exceeds non-null rows")
            if any(
                entry.count < self.disclosure_threshold for entry in column.top_k
            ):
                raise ValueError("profile top-k contains a below-threshold value")
            if self.rows - column.null_count < self.disclosure_threshold and (
                column.min_value is not None
                or column.max_value is not None
                or column.mean is not None
                or column.quantiles
                or column.top_k
            ):
                raise ValueError("profile discloses a small-group statistic")
        return self


class ProfileRequest(ProtocolModel):
    """Versioned, read-only request accepted by a future engine profile client."""

    schema_version: Literal["data-prep-profile.v1"]
    target: ProfileTarget
    selector: ProfileSelector | None = None
    limits: ProfileLimits = Field(default_factory=ProfileLimits)

    @model_validator(mode="after")
    def selector_matches_target(self) -> ProfileRequest:
        if self.selector is not None and (
            self.selector.schema_digest != self.target.schema_digest
        ):
            raise ValueError("profile selector schema digest is stale")
        return self


class ProfileEvidenceReference(ProtocolModel):
    """Reference returned only by a separately authorized native authority."""

    schema_version: Literal["data-prep-profile-evidence.v1"]
    authority: Literal["native_change_envelope"]
    profile_digest: Digest
    artifact_ref: OpaqueReference
    change_envelope_ref: OpaqueReference


__all__ = [
    "ArrowTypeName",
    "CanonicalNames",
    "ColumnProfile",
    "CleanPlan",
    "CleanStep",
    "ColumnName",
    "Dedupe",
    "Digest",
    "FillNulls",
    "InvalidRowDisposition",
    "LocalProfile",
    "NullPolicy",
    "OpaqueReference",
    "PrepEvidence",
    "ProfileEvidenceReference",
    "ProfileLimits",
    "ProfileRequest",
    "ProfileResult",
    "ProfileScalar",
    "ProfileSelector",
    "ProfileTarget",
    "ProfileTargetKind",
    "ProfileWarningCode",
    "QuantilePoint",
    "QuarantineOutcome",
    "RowOutcome",
    "SafeCast",
    "ScalarValue",
    "STEP_VERBS",
    "SchemaVersion",
    "StepEvidence",
    "TopKEntry",
]
