"""Focused fixtures for staged migration authority and replay safety."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from agent_utilities.control_plane.migrations import (
    BackfillBatch,
    BackfillObservation,
    CanonicalRecord,
    CohortReadCutover,
    GateCheck,
    InMemoryMigrationAuthority,
    InventoryItem,
    LegacyRetirement,
    MigrationCasConflictError,
    MigrationGateError,
    MigrationInventory,
    MigrationPlan,
    MigrationRollback,
    PrerequisiteGate,
    ProjectionCheckpoint,
    SourceFileSnapshot,
    SourceSnapshot,
    WriteCutoverFence,
)


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode()).hexdigest()}"


def _snapshot() -> SourceSnapshot:
    source_file = SourceFileSnapshot(
        source_file_ref="source-file:one",
        relative_path="src/one.json",
        source_ref="legacy:registry",
        revision="revision:one",
        content_digest=_digest("file-content-one"),
        metadata_digest=_digest("file-metadata-one"),
        size_bytes=12,
        version=1,
        digest=_digest("source-file-one"),
    )
    return SourceSnapshot.build(
        snapshot_ref="snapshot:one",
        source_ref="legacy:registry",
        revision="revision:one",
        files=(source_file,),
    )


def _plan(snapshot: SourceSnapshot) -> MigrationPlan:
    return MigrationPlan.initial(
        migration_ref="migration:one",
        plan_ref="plan:one",
        source_snapshot_ref=snapshot.snapshot_ref,
        source_snapshot_digest=snapshot.snapshot_digest,
        target_authority_ref="authority:target",
    )


def _inventory(snapshot: SourceSnapshot) -> MigrationInventory:
    item = InventoryItem(
        canonical_ref="record:one",
        source_ref="legacy:registry",
        source_file_ref="source-file:one",
        source_record_ref="source-record:one",
        source_revision=snapshot.revision,
        source_digest=_digest("record-one"),
        metadata_digest=_digest("record-metadata-one"),
        state="present",
        version=1,
        digest=_digest("inventory-item-one"),
    )
    return MigrationInventory.build(
        inventory_ref="inventory:one",
        migration_ref="migration:one",
        source_snapshot_ref=snapshot.snapshot_ref,
        source_snapshot_digest=snapshot.snapshot_digest,
        items=(item,),
    )


def _backfill(snapshot: SourceSnapshot) -> BackfillBatch:
    observation = BackfillObservation(
        observation_ref="observation:one",
        migration_ref="migration:one",
        source_snapshot_digest=snapshot.snapshot_digest,
        canonical_ref="record:one",
        source_ref="legacy:registry",
        source_file_ref="source-file:one",
        source_record_ref="source-record:one",
        source_revision=snapshot.revision,
        source_digest=_digest("record-one"),
        metadata_digest=_digest("record-metadata-one"),
        disposition="accepted",
        observed_at=1,
        version=1,
        digest=_digest("observation-one"),
    )
    return BackfillBatch.build(
        batch_ref="backfill:one",
        migration_ref="migration:one",
        source_snapshot_digest=snapshot.snapshot_digest,
        observations=(observation,),
    )


def _record(
    snapshot: SourceSnapshot,
    *,
    revision: str | None = None,
    label: str = "one",
) -> CanonicalRecord:
    return CanonicalRecord(
        canonical_ref="record:one",
        record_digest=_digest(f"record-{label}"),
        source_ref="legacy:registry",
        source_revision=revision or snapshot.revision,
        source_file_ref="source-file:one",
        source_record_ref="source-record:one",
        metadata_digest=_digest(f"record-metadata-{label}"),
        version=1,
        digest=_digest(f"canonical-record-{label}"),
    )


def _gate(
    snapshot: SourceSnapshot,
    *,
    gate_ref: str,
    stage: str,
) -> PrerequisiteGate:
    check = GateCheck(
        check_ref=f"check:{gate_ref.removeprefix('gate:')}",
        passed=True,
        evidence_digest=_digest(gate_ref),
    )
    return PrerequisiteGate(
        gate_ref=gate_ref,
        migration_ref="migration:one",
        stage=stage,
        source_snapshot_digest=snapshot.snapshot_digest,
        checks=(check,),
        decision="passed",
        version=1,
        digest=_digest(f"{gate_ref}:record"),
    )


def _prepare_fenced() -> tuple[
    InMemoryMigrationAuthority,
    SourceSnapshot,
    MigrationPlan,
    WriteCutoverFence,
]:
    snapshot = _snapshot()
    authority = InMemoryMigrationAuthority()
    plan = authority.create_plan(_plan(snapshot), snapshot=snapshot)
    inventory = _inventory(snapshot)
    plan = authority.freeze_inventory(
        plan.plan_ref,
        inventory,
        expected_plan_version=plan.version,
    )
    plan = authority.record_backfill(
        plan.plan_ref,
        _backfill(snapshot),
        expected_plan_version=plan.version,
    )
    authority.reconcile_shadow(
        plan.plan_ref,
        source_snapshot_digest=snapshot.snapshot_digest,
        target_snapshot_digest=snapshot.snapshot_digest,
        source_records=(_record(snapshot),),
        target_records=(_record(snapshot),),
        expected_plan_version=plan.version,
    )
    plan = authority.get_plan(plan.plan_ref)
    cutover = CohortReadCutover(
        cutover_ref="cutover:one",
        migration_ref=plan.migration_ref,
        cohort_ref="cohort:one",
        registry_ref="registry:target",
        source_snapshot_digest=snapshot.snapshot_digest,
        gate_ref="gate:read",
        version=1,
        digest=_digest("cutover-one"),
    )
    plan = authority.cutover_reads(
        plan.plan_ref,
        cutover,
        gate=_gate(
            snapshot,
            gate_ref="gate:read",
            stage="stage_3_cohort_read_cutover",
        ),
        expected_plan_version=plan.version,
    )
    fence = WriteCutoverFence(
        fence_ref="fence:one",
        migration_ref=plan.migration_ref,
        source_snapshot_digest=snapshot.snapshot_digest,
        cohort_refs=("cohort:one",),
        authority_ref=plan.target_authority_ref,
        fence_epoch=1,
        gate_ref="gate:write",
        version=1,
        digest=_digest("fence-one"),
    )
    plan = authority.fence_writes(
        plan.plan_ref,
        fence,
        gate=_gate(
            snapshot,
            gate_ref="gate:write",
            stage="stage_4_write_cutover",
        ),
        expected_plan_version=plan.version,
    )
    return authority, snapshot, plan, fence


def test_staged_authority_reaches_checkpoint_and_replays_without_delta() -> None:
    authority, snapshot, plan, fence = _prepare_fenced()
    checkpoint = ProjectionCheckpoint(
        checkpoint_ref="checkpoint:one",
        migration_ref=plan.migration_ref,
        source_snapshot_digest=snapshot.snapshot_digest,
        projection_digest=_digest("projection-one"),
        rebuild_plan_ref="rebuild:one",
        record_count=1,
        sequence=1,
        gate_ref="gate:checkpoint",
        state="complete",
        version=1,
        digest=_digest("checkpoint-one"),
    )
    gate = _gate(
        snapshot,
        gate_ref="gate:checkpoint",
        stage="stage_5_projection_checkpoint",
    )
    plan = authority.checkpoint_projection(
        plan.plan_ref,
        checkpoint,
        gate=gate,
        expected_plan_version=plan.version,
    )
    assert plan.state == "checkpointed"
    assert authority.checkpoint_projection(
        plan.plan_ref,
        checkpoint,
        gate=gate,
        expected_plan_version=1,
    ) == plan

    retirement = LegacyRetirement(
        retirement_ref="retirement:one",
        migration_ref=plan.migration_ref,
        source_snapshot_digest=snapshot.snapshot_digest,
        write_fence_ref=fence.fence_ref,
        checkpoint_ref=checkpoint.checkpoint_ref,
        remaining_consumers=0,
        remaining_facades=0,
        consumer_inventory_digest=_digest("consumer-inventory-empty"),
        facade_inventory_digest=_digest("facade-inventory-empty"),
        gate_ref="gate:retire",
        state="retired",
        version=1,
        digest=_digest("retirement-one"),
    )
    plan = authority.retire_legacy(
        plan.plan_ref,
        retirement,
        gate=_gate(
            snapshot,
            gate_ref="gate:retire",
            stage="legacy_retirement",
        ),
        expected_plan_version=plan.version,
    )
    assert plan.state == "retired"


def test_backfill_never_silently_approves_and_revision_delta_is_one() -> None:
    snapshot = _snapshot()
    authority = InMemoryMigrationAuthority()
    plan = authority.create_plan(_plan(snapshot), snapshot=snapshot)
    plan = authority.freeze_inventory(
        plan.plan_ref,
        _inventory(snapshot),
        expected_plan_version=plan.version,
    )
    approved = BackfillObservation(
        observation_ref="observation:approved",
        migration_ref=plan.migration_ref,
        source_snapshot_digest=snapshot.snapshot_digest,
        canonical_ref="record:approved",
        source_ref="legacy:registry",
        source_revision=snapshot.revision,
        source_digest=_digest("approved-record"),
        metadata_digest=_digest("approved-metadata"),
        disposition="accepted",
        approval_state="approved",
        approval_ref="approval:one",
        approver_ref="operator:one",
        observed_at=1,
        version=1,
        digest=_digest("approved-observation"),
    )
    with pytest.raises(ValidationError):
        BackfillBatch.build(
            batch_ref="backfill:approved",
            migration_ref=plan.migration_ref,
            source_snapshot_digest=snapshot.snapshot_digest,
            observations=(approved,),
        )
    plan = authority.record_backfill(
        plan.plan_ref,
        _backfill(snapshot),
        expected_plan_version=plan.version,
    )
    clean = authority.reconcile_shadow(
        plan.plan_ref,
        source_snapshot_digest=snapshot.snapshot_digest,
        target_snapshot_digest=snapshot.snapshot_digest,
        source_records=(_record(snapshot),),
        target_records=(_record(snapshot),),
        expected_plan_version=plan.version,
    )
    assert clean.delta_count == 0
    changed = authority.reconcile_shadow(
        plan.plan_ref,
        source_snapshot_digest=snapshot.snapshot_digest,
        target_snapshot_digest=_digest("target-revision-two"),
        source_records=(_record(snapshot),),
        target_records=(_record(snapshot, revision="revision:two", label="two"),),
        expected_plan_version=authority.get_plan(plan.plan_ref).version,
    )
    assert changed.delta_count == 1
    assert changed.deltas[0].kind == "update"


def test_cutover_fails_closed_on_gate_or_cas_and_rollback_is_explicit() -> None:
    authority, snapshot, plan, fence = _prepare_fenced()
    blocked_check = GateCheck(
        check_ref="check:blocked",
        passed=False,
        evidence_digest=_digest("blocked"),
        failure_code="backfill:unreviewed",
    )
    blocked_gate = PrerequisiteGate(
        gate_ref="gate:blocked",
        migration_ref=plan.migration_ref,
        stage="stage_5_projection_checkpoint",
        source_snapshot_digest=snapshot.snapshot_digest,
        checks=(blocked_check,),
        decision="blocked",
        version=1,
        digest=_digest("blocked-gate"),
    )
    checkpoint = ProjectionCheckpoint(
        checkpoint_ref="checkpoint:blocked",
        migration_ref=plan.migration_ref,
        source_snapshot_digest=snapshot.snapshot_digest,
        projection_digest=_digest("projection-blocked"),
        rebuild_plan_ref="rebuild:blocked",
        record_count=1,
        sequence=1,
        gate_ref=blocked_gate.gate_ref,
        state="complete",
        version=1,
        digest=_digest("checkpoint-blocked"),
    )
    with pytest.raises(MigrationGateError):
        authority.checkpoint_projection(
            plan.plan_ref,
            checkpoint,
            gate=blocked_gate,
            expected_plan_version=plan.version,
        )
    with pytest.raises(MigrationCasConflictError):
        authority.checkpoint_projection(
            plan.plan_ref,
            checkpoint,
            gate=_gate(
                snapshot,
                gate_ref="gate:checkpoint",
                stage="stage_5_projection_checkpoint",
            ),
            expected_plan_version=plan.version - 1,
        )

    rollback = MigrationRollback(
        rollback_ref="rollback:one",
        migration_ref=plan.migration_ref,
        from_stage="stage_4_write_cutover",
        to_stage="stage_3_cohort_read_cutover",
        source_snapshot_digest=snapshot.snapshot_digest,
        reason_ref="reason:verification-failed",
        restored_authority_ref="authority:legacy",
        fence_ref=fence.fence_ref,
        gate_ref="gate:rollback",
        decision="requested",
        version=1,
        digest=_digest("rollback-one"),
    )
    rolled_back = authority.rollback(
        plan.plan_ref,
        rollback,
        expected_plan_version=plan.version,
    )
    assert rolled_back.state == "rolled_back"


def test_inventory_and_retirement_proofs_reject_drift() -> None:
    snapshot = _snapshot()
    authority = InMemoryMigrationAuthority()
    plan = authority.create_plan(_plan(snapshot), snapshot=snapshot)
    with pytest.raises(MigrationCasConflictError):
        authority.freeze_inventory(
            plan.plan_ref,
            _inventory(snapshot),
            expected_plan_version=2,
        )
    with pytest.raises(ValidationError):
        LegacyRetirement(
            retirement_ref="retirement:unsafe",
            migration_ref=plan.migration_ref,
            source_snapshot_digest=snapshot.snapshot_digest,
            write_fence_ref="fence:missing",
            checkpoint_ref="checkpoint:missing",
            remaining_consumers=1,
            remaining_facades=0,
            consumer_inventory_digest=_digest("consumer-inventory"),
            facade_inventory_digest=_digest("facade-inventory"),
            gate_ref="gate:retire",
            state="retired",
            version=1,
            digest=_digest("unsafe-retirement"),
        )
