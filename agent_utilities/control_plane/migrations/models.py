"""Frozen, provenance-first contracts for controlled data migrations.

These models describe migration evidence and authority transitions only.  They
never carry source payloads, credentials, or approval authority implicitly.
Every snapshot, plan, observation, and stage record is identified by an opaque
reference, a source revision, and a digest so restart/replay can be idempotent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "MAX_COHORTS",
    "MAX_EVIDENCE_ITEMS",
    "MAX_SOURCE_FILES",
    "ApprovalState",
    "BackfillBatch",
    "BackfillObservation",
    "CanonicalRecord",
    "CohortReadCutover",
    "Count",
    "Digest",
    "GateCheck",
    "InventoryState",
    "InventoryItem",
    "LegacyRetirement",
    "MigrationInventory",
    "MigrationPlan",
    "MigrationRollback",
    "MigrationStage",
    "MigrationState",
    "OpaqueRef",
    "PrerequisiteGate",
    "ProjectionCheckpoint",
    "ShadowDelta",
    "ShadowReconciliation",
    "SourceDisposition",
    "SourceFileSnapshot",
    "SourceSnapshot",
    "Timestamp",
    "Version",
    "WriteCutoverFence",
    "canonical_digest",
]


MAX_SOURCE_FILES = 4_096
MAX_EVIDENCE_ITEMS = 10_000
MAX_COHORTS = 256

_REF_RE = r"^[A-Za-z][A-Za-z0-9:_./-]{0,255}$"
_DIGEST_RE = r"^sha256:[0-9a-f]{64}$"

OpaqueRef: TypeAlias = Annotated[str, Field(pattern=_REF_RE, min_length=1)]
Digest: TypeAlias = Annotated[str, Field(pattern=_DIGEST_RE)]
Version: TypeAlias = Annotated[int, Field(ge=1, le=2_147_483_647)]
Count: TypeAlias = Annotated[int, Field(ge=0, le=2_147_483_647)]
Timestamp: TypeAlias = Annotated[int, Field(ge=0)]

MigrationStage: TypeAlias = Literal[
    "stage_0_inventory_freeze",
    "stage_1_read_only_backfill",
    "stage_2_shadow_reconciliation",
    "stage_3_cohort_read_cutover",
    "stage_4_write_cutover",
    "stage_5_projection_checkpoint",
    "legacy_retirement",
    "rollback",
]
MigrationState: TypeAlias = Literal[
    "planned",
    "inventory_frozen",
    "backfill_observing",
    "shadow_reconciled",
    "read_cutover",
    "write_fenced",
    "checkpointed",
    "retired",
    "rolled_back",
]
SourceDisposition: TypeAlias = Literal["accepted", "rejected", "malformed"]
ApprovalState: TypeAlias = Literal["unreviewed", "approved", "rejected"]
InventoryState: TypeAlias = Literal[
    "present", "missing", "duplicate", "rejected", "malformed"
]

_FORBIDDEN_INLINE_KEYS = {
    "body",
    "bytes",
    "content",
    "credentials",
    "data",
    "password",
    "payload",
    "private_key",
    "result",
    "secret",
    "text",
    "token",
    "values",
}


def _canonical(value: object) -> object:
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def canonical_digest(value: object) -> str:
    """Return the stable digest used for replay and successor identity."""

    payload = json.dumps(
        _canonical(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=False,
        strict=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_inline_payloads(cls, value: object) -> object:
        if isinstance(value, Mapping):
            keys = {
                str(key).casefold().replace("-", "_")
                for key in value
                if str(key).casefold().replace("-", "_") in _FORBIDDEN_INLINE_KEYS
            }
            if keys:
                raise ValueError("migration_inline_payload_forbidden")
        return value


class _VersionedModel(_FrozenModel):
    version: Version
    digest: Digest

    @property
    def identity_digest(self) -> str:
        return canonical_digest(self)


class SourceFileSnapshot(_VersionedModel):
    """Repository-relative file metadata; file bytes never cross this boundary."""

    source_file_ref: OpaqueRef
    relative_path: OpaqueRef
    source_ref: OpaqueRef
    revision: OpaqueRef
    content_digest: Digest
    metadata_digest: Digest
    size_bytes: Count


class SourceSnapshot(_VersionedModel):
    """Canonical source revision used by every stage and checkpoint."""

    snapshot_ref: OpaqueRef
    source_ref: OpaqueRef
    revision: OpaqueRef
    files: tuple[SourceFileSnapshot, ...] = Field(
        min_length=1,
        max_length=MAX_SOURCE_FILES,
    )
    snapshot_digest: Digest

    @model_validator(mode="after")
    def _files_are_unique_and_bound(self) -> SourceSnapshot:
        refs = [item.source_file_ref for item in self.files]
        if len(refs) != len(set(refs)):
            raise ValueError("source_file_duplicate")
        if any(
            item.source_ref != self.source_ref or item.revision != self.revision
            for item in self.files
        ):
            raise ValueError("source_file_revision_drift")
        expected = canonical_digest(
            {
                "source_ref": self.source_ref,
                "revision": self.revision,
                "files": self.files,
            }
        )
        if self.snapshot_digest != expected:
            raise ValueError("source_snapshot_digest_mismatch")
        return self

    @classmethod
    def build(
        cls,
        *,
        snapshot_ref: str,
        source_ref: str,
        revision: str,
        files: tuple[SourceFileSnapshot, ...],
        version: int = 1,
    ) -> SourceSnapshot:
        snapshot_digest = canonical_digest(
            {
                "source_ref": source_ref,
                "revision": revision,
                "files": files,
            }
        )
        return cls(
            snapshot_ref=snapshot_ref,
            source_ref=source_ref,
            revision=revision,
            files=files,
            snapshot_digest=snapshot_digest,
            version=version,
            digest=canonical_digest(
                {
                    "snapshot_ref": snapshot_ref,
                    "snapshot_digest": snapshot_digest,
                    "version": version,
                }
            ),
        )


class MigrationPlan(_VersionedModel):
    """The only mutable migration authority; each transition is a new version."""

    migration_ref: OpaqueRef
    plan_ref: OpaqueRef
    source_snapshot_ref: OpaqueRef
    source_snapshot_digest: Digest
    target_authority_ref: OpaqueRef
    stage: MigrationStage = "stage_0_inventory_freeze"
    state: MigrationState = "planned"
    inventory_ref: OpaqueRef | None = None
    reconciliation_ref: OpaqueRef | None = None
    cohort_refs: tuple[OpaqueRef, ...] = Field(default=(), max_length=MAX_COHORTS)
    write_fence_ref: OpaqueRef | None = None
    checkpoint_ref: OpaqueRef | None = None
    retirement_ref: OpaqueRef | None = None

    @model_validator(mode="after")
    def _stage_state_is_consistent(self) -> MigrationPlan:
        expected = {
            "planned": "stage_0_inventory_freeze",
            "inventory_frozen": "stage_0_inventory_freeze",
            "backfill_observing": "stage_1_read_only_backfill",
            "shadow_reconciled": "stage_2_shadow_reconciliation",
            "read_cutover": "stage_3_cohort_read_cutover",
            "write_fenced": "stage_4_write_cutover",
            "checkpointed": "stage_5_projection_checkpoint",
            "retired": "legacy_retirement",
            "rolled_back": "rollback",
        }
        if self.stage != expected[self.state]:
            raise ValueError("migration_stage_state_mismatch")
        if len(self.cohort_refs) != len(set(self.cohort_refs)):
            raise ValueError("migration_cohort_duplicate")
        return self

    @classmethod
    def initial(
        cls,
        *,
        migration_ref: str,
        plan_ref: str,
        source_snapshot_ref: str,
        source_snapshot_digest: str,
        target_authority_ref: str,
    ) -> MigrationPlan:
        return cls(
            migration_ref=migration_ref,
            plan_ref=plan_ref,
            source_snapshot_ref=source_snapshot_ref,
            source_snapshot_digest=source_snapshot_digest,
            target_authority_ref=target_authority_ref,
            version=1,
            digest=canonical_digest(
                {
                    "migration_ref": migration_ref,
                    "plan_ref": plan_ref,
                    "source_snapshot_digest": source_snapshot_digest,
                    "target_authority_ref": target_authority_ref,
                    "version": 1,
                }
            ),
        )


class InventoryItem(_VersionedModel):
    """Stable source-to-target inventory observation, including rejects."""

    canonical_ref: OpaqueRef
    source_ref: OpaqueRef
    source_file_ref: OpaqueRef | None = None
    source_record_ref: OpaqueRef | None = None
    source_revision: OpaqueRef
    source_digest: Digest
    metadata_digest: Digest
    state: InventoryState


class MigrationInventory(_VersionedModel):
    """Frozen stage-0 inventory; a later stage cannot rewrite its source view."""

    inventory_ref: OpaqueRef
    migration_ref: OpaqueRef
    source_snapshot_ref: OpaqueRef
    source_snapshot_digest: Digest
    items: tuple[InventoryItem, ...] = Field(
        min_length=1,
        max_length=MAX_EVIDENCE_ITEMS,
    )
    frozen: Literal[True] = True
    inventory_digest: Digest

    @model_validator(mode="after")
    def _inventory_is_deterministic(self) -> MigrationInventory:
        refs = [item.canonical_ref for item in self.items]
        if len(refs) != len(set(refs)):
            raise ValueError("inventory_identity_duplicate")
        expected = canonical_digest(self.items)
        if self.inventory_digest != expected:
            raise ValueError("inventory_digest_mismatch")
        return self

    @classmethod
    def build(
        cls,
        *,
        inventory_ref: str,
        migration_ref: str,
        source_snapshot_ref: str,
        source_snapshot_digest: str,
        items: tuple[InventoryItem, ...],
        version: int = 1,
    ) -> MigrationInventory:
        inventory_digest = canonical_digest(items)
        return cls(
            inventory_ref=inventory_ref,
            migration_ref=migration_ref,
            source_snapshot_ref=source_snapshot_ref,
            source_snapshot_digest=source_snapshot_digest,
            items=items,
            inventory_digest=inventory_digest,
            version=version,
            digest=canonical_digest(
                {
                    "inventory_ref": inventory_ref,
                    "inventory_digest": inventory_digest,
                    "version": version,
                }
            ),
        )


class BackfillObservation(_VersionedModel):
    """Read-only observation that cannot imply target approval."""

    observation_ref: OpaqueRef
    migration_ref: OpaqueRef
    source_snapshot_digest: Digest
    canonical_ref: OpaqueRef
    source_ref: OpaqueRef
    source_file_ref: OpaqueRef | None = None
    source_record_ref: OpaqueRef | None = None
    source_revision: OpaqueRef
    source_digest: Digest
    metadata_digest: Digest
    disposition: SourceDisposition
    approval_state: ApprovalState = "unreviewed"
    approval_ref: OpaqueRef | None = None
    approver_ref: OpaqueRef | None = None
    observed_at: Timestamp

    @model_validator(mode="after")
    def _approval_is_explicit(self) -> BackfillObservation:
        if self.approval_state == "approved":
            if self.disposition != "accepted":
                raise ValueError("nonaccepted_backfill_cannot_approve")
            if self.approval_ref is None or self.approver_ref is None:
                raise ValueError("backfill_approval_evidence_missing")
        elif self.approval_ref is not None or self.approver_ref is not None:
            raise ValueError("backfill_approval_evidence_unexpected")
        return self


class BackfillBatch(_VersionedModel):
    """Bounded read-only evidence batch, replayable by its stable batch ref."""

    batch_ref: OpaqueRef
    migration_ref: OpaqueRef
    source_snapshot_digest: Digest
    observations: tuple[BackfillObservation, ...] = Field(
        min_length=1,
        max_length=MAX_EVIDENCE_ITEMS,
    )
    read_only: Literal[True] = True
    batch_digest: Digest

    @model_validator(mode="after")
    def _observations_are_bound(self) -> BackfillBatch:
        if any(
            item.migration_ref != self.migration_ref
            or item.source_snapshot_digest != self.source_snapshot_digest
            or item.approval_state != "unreviewed"
            for item in self.observations
        ):
            raise ValueError("backfill_observation_binding_invalid")
        refs = [item.observation_ref for item in self.observations]
        if len(refs) != len(set(refs)):
            raise ValueError("backfill_observation_duplicate")
        if self.batch_digest != canonical_digest(self.observations):
            raise ValueError("backfill_batch_digest_mismatch")
        return self

    @classmethod
    def build(
        cls,
        *,
        batch_ref: str,
        migration_ref: str,
        source_snapshot_digest: str,
        observations: tuple[BackfillObservation, ...],
        version: int = 1,
    ) -> BackfillBatch:
        batch_digest = canonical_digest(observations)
        return cls(
            batch_ref=batch_ref,
            migration_ref=migration_ref,
            source_snapshot_digest=source_snapshot_digest,
            observations=observations,
            batch_digest=batch_digest,
            version=version,
            digest=canonical_digest(
                {
                    "batch_ref": batch_ref,
                    "batch_digest": batch_digest,
                    "version": version,
                }
            ),
        )


class CanonicalRecord(_VersionedModel):
    """Digest-only record identity used by shadow reconciliation."""

    canonical_ref: OpaqueRef
    record_digest: Digest
    source_ref: OpaqueRef
    source_revision: OpaqueRef
    source_file_ref: OpaqueRef | None = None
    source_record_ref: OpaqueRef | None = None
    metadata_digest: Digest


class ShadowDelta(_VersionedModel):
    """One deterministic difference; unchanged records are deliberately omitted."""

    canonical_ref: OpaqueRef
    kind: Literal["add", "update", "delete"]
    source_digest: Digest | None = None
    target_digest: Digest | None = None
    evidence_digest: Digest

    @model_validator(mode="after")
    def _delta_has_expected_digests(self) -> ShadowDelta:
        if self.kind == "add" and self.source_digest is None:
            raise ValueError("shadow_add_source_missing")
        if self.kind == "delete" and self.target_digest is None:
            raise ValueError("shadow_delete_target_missing")
        if self.kind == "update" and (
            self.source_digest is None or self.target_digest is None
        ):
            raise ValueError("shadow_update_digest_missing")
        return self


class ShadowReconciliation(_VersionedModel):
    """Deterministic source/target comparison bound to both snapshot digests."""

    reconciliation_ref: OpaqueRef
    migration_ref: OpaqueRef
    source_snapshot_digest: Digest
    target_snapshot_digest: Digest
    deltas: tuple[ShadowDelta, ...] = Field(max_length=MAX_EVIDENCE_ITEMS)
    delta_count: Count
    delta_digest: Digest
    state: Literal["clean", "needs_review", "blocked"]
    deterministic: Literal[True] = True

    @model_validator(mode="after")
    def _delta_summary_is_exact(self) -> ShadowReconciliation:
        if self.delta_count != len(self.deltas):
            raise ValueError("shadow_delta_count_mismatch")
        if self.delta_digest != canonical_digest(self.deltas):
            raise ValueError("shadow_delta_digest_mismatch")
        if self.state == "clean" and self.deltas:
            raise ValueError("shadow_clean_with_deltas")
        if self.state == "needs_review" and not self.deltas:
            raise ValueError("shadow_nonclean_without_deltas")
        return self


class GateCheck(_FrozenModel):
    """Bounded prerequisite evidence; a missing check never passes a gate."""

    check_ref: OpaqueRef
    passed: bool
    evidence_digest: Digest
    failure_code: OpaqueRef | None = None

    @model_validator(mode="after")
    def _failure_is_consistent(self) -> GateCheck:
        if self.passed and self.failure_code is not None:
            raise ValueError("passed_gate_check_has_failure")
        if not self.passed and self.failure_code is None:
            raise ValueError("failed_gate_check_missing_reason")
        return self


class PrerequisiteGate(_VersionedModel):
    """Explicit fail-closed gate for a stage transition."""

    gate_ref: OpaqueRef
    migration_ref: OpaqueRef
    stage: MigrationStage
    source_snapshot_digest: Digest
    checks: tuple[GateCheck, ...] = Field(min_length=1, max_length=64)
    decision: Literal["passed", "blocked"]

    @model_validator(mode="after")
    def _decision_is_fail_closed(self) -> PrerequisiteGate:
        refs = [check.check_ref for check in self.checks]
        if len(refs) != len(set(refs)):
            raise ValueError("gate_check_duplicate")
        if self.decision == "passed" and not all(check.passed for check in self.checks):
            raise ValueError("gate_passed_with_failed_check")
        if self.decision == "blocked" and all(check.passed for check in self.checks):
            raise ValueError("gate_blocked_without_failure")
        return self


class CohortReadCutover(_VersionedModel):
    """One cohort-scoped registry read cutover, never a fleet-wide flip."""

    cutover_ref: OpaqueRef
    migration_ref: OpaqueRef
    cohort_ref: OpaqueRef
    registry_ref: OpaqueRef
    source_snapshot_digest: Digest
    gate_ref: OpaqueRef
    state: Literal["active", "rolled_back"] = "active"


class WriteCutoverFence(_VersionedModel):
    """Single-authority write fence with legacy writes explicitly denied."""

    fence_ref: OpaqueRef
    migration_ref: OpaqueRef
    source_snapshot_digest: Digest
    cohort_refs: tuple[OpaqueRef, ...] = Field(
        min_length=1,
        max_length=MAX_COHORTS,
    )
    authority_ref: OpaqueRef
    fence_epoch: Count
    gate_ref: OpaqueRef
    single_authority: Literal[True] = True
    legacy_writes_denied: Literal[True] = True
    state: Literal["active", "rolled_back"] = "active"

    @model_validator(mode="after")
    def _cohorts_are_unique(self) -> WriteCutoverFence:
        if len(self.cohort_refs) != len(set(self.cohort_refs)):
            raise ValueError("write_fence_cohort_duplicate")
        return self


class ProjectionCheckpoint(_VersionedModel):
    """Restart-safe projection checkpoint/rebuild evidence."""

    checkpoint_ref: OpaqueRef
    migration_ref: OpaqueRef
    source_snapshot_digest: Digest
    projection_digest: Digest
    rebuild_plan_ref: OpaqueRef
    record_count: Count
    sequence: Version
    gate_ref: OpaqueRef
    state: Literal["complete", "rebuild_required"]


class LegacyRetirement(_VersionedModel):
    """Retirement proof for temporary compatibility paths and old consumers."""

    retirement_ref: OpaqueRef
    migration_ref: OpaqueRef
    source_snapshot_digest: Digest
    write_fence_ref: OpaqueRef
    checkpoint_ref: OpaqueRef
    remaining_consumers: Count
    remaining_facades: Count
    consumer_inventory_digest: Digest
    facade_inventory_digest: Digest
    gate_ref: OpaqueRef
    state: Literal["ready", "retired", "rolled_back"]

    @model_validator(mode="after")
    def _retirement_is_proven(self) -> LegacyRetirement:
        if self.state == "retired" and (
            self.remaining_consumers != 0 or self.remaining_facades != 0
        ):
            raise ValueError("legacy_retirement_with_remaining_consumers")
        return self


class MigrationRollback(_VersionedModel):
    """Explicit, recorded rollback; no stage silently rolls back itself."""

    rollback_ref: OpaqueRef
    migration_ref: OpaqueRef
    from_stage: MigrationStage
    to_stage: MigrationStage
    source_snapshot_digest: Digest
    reason_ref: OpaqueRef
    restored_authority_ref: OpaqueRef
    fence_ref: OpaqueRef | None = None
    gate_ref: OpaqueRef
    decision: Literal["requested", "applied", "failed"]

    @model_validator(mode="after")
    def _rollback_direction_is_explicit(self) -> MigrationRollback:
        if self.from_stage == self.to_stage:
            raise ValueError("rollback_stage_unchanged")
        return self
