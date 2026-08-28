"""In-memory migration authority implementing the staged cutover contract."""

from __future__ import annotations

from collections.abc import Mapping
from threading import RLock
from typing import Literal, cast

from .errors import (
    MigrationCasConflictError,
    MigrationConflictError,
    MigrationGateError,
    MigrationNotFoundError,
    MigrationPrerequisiteError,
    MigrationReplayError,
)
from .models import (
    BackfillBatch,
    BackfillObservation,
    CanonicalRecord,
    CohortReadCutover,
    LegacyRetirement,
    MigrationInventory,
    MigrationPlan,
    MigrationRollback,
    PrerequisiteGate,
    ProjectionCheckpoint,
    ShadowDelta,
    ShadowReconciliation,
    SourceSnapshot,
    WriteCutoverFence,
    canonical_digest,
)

__all__ = ["InMemoryMigrationAuthority"]


_MAX_VERSION = 2_147_483_647


def _digest_for(data: Mapping[str, object]) -> str:
    return canonical_digest(
        {key: value for key, value in data.items() if key != "digest"}
    )


class InMemoryMigrationAuthority:
    """Deterministic contract authority for migration planning and cutover.

    The repository is intentionally small and in-memory.  A durable adapter can
    preserve the same keys and CAS rules.  Stage records are append-by-version:
    replaying the same reference and digest returns the original record, while
    a changed payload is rejected rather than silently replacing evidence.
    """

    def __init__(self) -> None:
        self._snapshots: dict[str, SourceSnapshot] = {}
        self._plans: dict[str, MigrationPlan] = {}
        self._inventories: dict[str, MigrationInventory] = {}
        self._backfills: dict[str, BackfillBatch] = {}
        self._reconciliations: dict[str, ShadowReconciliation] = {}
        self._read_cutovers: dict[str, CohortReadCutover] = {}
        self._write_fences: dict[str, WriteCutoverFence] = {}
        self._checkpoints: dict[str, ProjectionCheckpoint] = {}
        self._retirements: dict[str, LegacyRetirement] = {}
        self._rollbacks: dict[str, MigrationRollback] = {}
        self._lock = RLock()

    @staticmethod
    def _cas_version(actual: int, expected: int | None) -> None:
        if expected is None or expected != actual:
            raise MigrationCasConflictError()

    @staticmethod
    def _next_plan(
        plan: MigrationPlan,
        *,
        expected_version: int,
        state: str,
        stage: str,
        **updates: object,
    ) -> MigrationPlan:
        InMemoryMigrationAuthority._cas_version(plan.version, expected_version)
        if plan.version >= _MAX_VERSION:
            raise MigrationCasConflictError("migration_version_exhausted")
        data = plan.model_dump(mode="python")
        data.update(updates)
        data["state"] = state
        data["stage"] = stage
        data["version"] = plan.version + 1
        data["digest"] = _digest_for(data)
        return MigrationPlan(**data)

    @staticmethod
    def _same_or_conflict(
        existing: object | None,
        incoming: object,
        *,
        identity: str,
    ) -> object | None:
        if existing is None:
            return None
        if existing == incoming:
            return existing
        raise MigrationReplayError(f"{identity}_replay_drift")

    @staticmethod
    def _require_gate(
        gate: PrerequisiteGate,
        *,
        migration_ref: str,
        source_snapshot_digest: str,
        stage: str,
    ) -> None:
        if (
            gate.migration_ref != migration_ref
            or gate.source_snapshot_digest != source_snapshot_digest
            or gate.stage != stage
        ):
            raise MigrationGateError("migration_gate_binding_mismatch")
        if gate.decision != "passed":
            raise MigrationGateError("migration_gate_blocked")

    def _snapshot(self, snapshot_ref: str, digest: str) -> SourceSnapshot:
        snapshot = self._snapshots.get(snapshot_ref)
        if snapshot is None:
            raise MigrationPrerequisiteError("source_snapshot_missing")
        if snapshot.snapshot_digest != digest:
            raise MigrationConflictError("source_snapshot_digest_drift")
        return snapshot

    def _plan(self, plan_ref: str) -> MigrationPlan:
        plan = self._plans.get(plan_ref)
        if plan is None:
            raise MigrationNotFoundError()
        return plan

    def register_snapshot(self, snapshot: SourceSnapshot) -> SourceSnapshot:
        """Register one immutable source snapshot, idempotently by reference."""

        with self._lock:
            existing = self._snapshots.get(snapshot.snapshot_ref)
            same = self._same_or_conflict(
                existing,
                snapshot,
                identity="source_snapshot",
            )
            if same is not None:
                return cast(SourceSnapshot, same)
            self._snapshots[snapshot.snapshot_ref] = snapshot
            return snapshot

    def create_plan(
        self,
        plan: MigrationPlan,
        *,
        snapshot: SourceSnapshot,
    ) -> MigrationPlan:
        """Create a planned migration only against a registered source snapshot."""

        if plan.version != 1 or plan.state != "planned":
            raise MigrationConflictError("migration_plan_not_initial")
        if plan.source_snapshot_ref != snapshot.snapshot_ref:
            raise MigrationConflictError("migration_plan_snapshot_ref_mismatch")
        if plan.source_snapshot_digest != snapshot.snapshot_digest:
            raise MigrationConflictError("migration_plan_snapshot_digest_mismatch")
        with self._lock:
            self.register_snapshot(snapshot)
            existing = self._plans.get(plan.plan_ref)
            same = self._same_or_conflict(
                existing,
                plan,
                identity="migration_plan",
            )
            if same is not None:
                return cast(MigrationPlan, same)
            self._plans[plan.plan_ref] = plan
            return plan

    def get_plan(self, plan_ref: str) -> MigrationPlan:
        with self._lock:
            return self._plan(plan_ref)

    @staticmethod
    def _inventory_matches_snapshot_binding(
        inventory: MigrationInventory,
        plan: MigrationPlan,
        snapshot: SourceSnapshot,
        source_file_refs: set[str],
    ) -> bool:
        """Whether ``inventory`` is correctly bound to ``plan``'s source snapshot."""
        if (
            inventory.migration_ref != plan.migration_ref
            or inventory.source_snapshot_ref != plan.source_snapshot_ref
            or inventory.source_snapshot_digest != plan.source_snapshot_digest
        ):
            return False
        return not any(
            item.source_ref != snapshot.source_ref
            or item.source_revision != snapshot.revision
            or (
                item.source_file_ref is not None
                and item.source_file_ref not in source_file_refs
            )
            for item in inventory.items
        )

    def freeze_inventory(
        self,
        plan_ref: str,
        inventory: MigrationInventory,
        *,
        expected_plan_version: int,
    ) -> MigrationPlan:
        """Freeze stage 0 inventory without mutating source observations."""

        with self._lock:
            plan = self._plan(plan_ref)
            snapshot = self._snapshot(
                plan.source_snapshot_ref, plan.source_snapshot_digest
            )
            source_file_refs = {
                source_file.source_file_ref for source_file in snapshot.files
            }
            if not self._inventory_matches_snapshot_binding(
                inventory, plan, snapshot, source_file_refs
            ):
                raise MigrationConflictError("inventory_snapshot_binding_mismatch")
            existing_inventory = self._inventories.get(inventory.inventory_ref)
            same = self._same_or_conflict(
                existing_inventory,
                inventory,
                identity="migration_inventory",
            )
            if same is not None:
                if plan.inventory_ref == inventory.inventory_ref:
                    return plan
                raise MigrationReplayError("inventory_reference_replay_drift")
            if plan.state != "planned":
                raise MigrationPrerequisiteError("inventory_freeze_stage_invalid")
            updated = self._next_plan(
                plan,
                expected_version=expected_plan_version,
                state="inventory_frozen",
                stage="stage_0_inventory_freeze",
                inventory_ref=inventory.inventory_ref,
            )
            self._inventories[inventory.inventory_ref] = inventory
            self._plans[plan.plan_ref] = updated
            return updated

    @staticmethod
    def _backfill_observation_conflict(
        batch: BackfillBatch, inventory_refs: set[str], snapshot: SourceSnapshot
    ) -> str | None:
        """The conflict-error key if any observation in ``batch`` is invalid, else ``None``."""
        if any(
            observation.canonical_ref not in inventory_refs
            for observation in batch.observations
        ):
            return "backfill_identity_not_in_inventory"
        if any(
            observation.source_revision != snapshot.revision
            for observation in batch.observations
        ):
            return "backfill_source_revision_mismatch"
        return None

    def record_backfill(
        self,
        plan_ref: str,
        batch: BackfillBatch,
        *,
        expected_plan_version: int,
    ) -> MigrationPlan:
        """Record read-only observations; approval is never inferred."""

        with self._lock:
            plan = self._plan(plan_ref)
            snapshot = self._snapshot(
                plan.source_snapshot_ref,
                plan.source_snapshot_digest,
            )
            if (
                batch.migration_ref != plan.migration_ref
                or batch.source_snapshot_digest != plan.source_snapshot_digest
            ):
                raise MigrationConflictError("backfill_snapshot_binding_mismatch")
            if (
                plan.inventory_ref is None
                or plan.inventory_ref not in self._inventories
            ):
                raise MigrationPrerequisiteError("frozen_inventory_missing")
            inventory = self._inventories[plan.inventory_ref]
            inventory_refs = {item.canonical_ref for item in inventory.items}
            conflict = self._backfill_observation_conflict(
                batch, inventory_refs, snapshot
            )
            if conflict is not None:
                raise MigrationConflictError(conflict)
            existing = self._backfills.get(batch.batch_ref)
            same = self._same_or_conflict(
                existing,
                batch,
                identity="backfill_batch",
            )
            if same is not None:
                return plan
            if plan.state not in {"inventory_frozen", "backfill_observing"}:
                raise MigrationPrerequisiteError("backfill_stage_invalid")
            next_state = (
                "backfill_observing" if plan.state == "inventory_frozen" else plan.state
            )
            updated = self._next_plan(
                plan,
                expected_version=expected_plan_version,
                state=next_state,
                stage="stage_1_read_only_backfill",
            )
            self._backfills[batch.batch_ref] = batch
            self._plans[plan.plan_ref] = updated
            return updated

    def get_backfill(self, batch_ref: str) -> BackfillBatch:
        with self._lock:
            batch = self._backfills.get(batch_ref)
            if batch is None:
                raise MigrationNotFoundError()
            return batch

    def get_inventory(self, inventory_ref: str) -> MigrationInventory:
        with self._lock:
            inventory = self._inventories.get(inventory_ref)
            if inventory is None:
                raise MigrationNotFoundError()
            return inventory

    def _observations_for(
        self,
        plan: MigrationPlan,
    ) -> tuple[BackfillObservation, ...]:
        rows: list[BackfillObservation] = []
        for batch in self._backfills.values():
            if (
                batch.migration_ref == plan.migration_ref
                and batch.source_snapshot_digest == plan.source_snapshot_digest
            ):
                rows.extend(batch.observations)
        return tuple(sorted(rows, key=lambda item: item.canonical_ref))

    @staticmethod
    def _delta(
        canonical_ref: str,
        *,
        kind: Literal["add", "update", "delete"],
        source_digest: str | None,
        target_digest: str | None,
    ) -> ShadowDelta:
        data = {
            "canonical_ref": canonical_ref,
            "kind": kind,
            "source_digest": source_digest,
            "target_digest": target_digest,
        }
        return ShadowDelta(
            canonical_ref=canonical_ref,
            kind=kind,
            source_digest=source_digest,
            target_digest=target_digest,
            evidence_digest=canonical_digest(data),
            version=1,
            digest=_digest_for(data),
        )

    @staticmethod
    def _build_shadow_record_maps(
        source_records: tuple[CanonicalRecord, ...],
        target_records: tuple[CanonicalRecord, ...],
    ) -> tuple[dict[str, CanonicalRecord], dict[str, CanonicalRecord]]:
        """Build ``canonical_ref -> record`` maps; raises on any duplicate canonical_ref."""
        source_map = {record.canonical_ref: record for record in source_records}
        target_map = {record.canonical_ref: record for record in target_records}
        if len(source_map) != len(source_records) or len(target_map) != len(
            target_records
        ):
            raise MigrationConflictError("shadow_record_duplicate")
        return source_map, target_map

    def _load_shadow_observations(
        self, plan: MigrationPlan
    ) -> tuple[BackfillObservation, ...]:
        """Load + validate the plan's backfill observations (non-empty, no duplicate identity)."""
        observations = self._observations_for(plan)
        if not observations:
            raise MigrationPrerequisiteError("backfill_observations_missing")
        observation_refs = [item.canonical_ref for item in observations]
        if len(observation_refs) != len(set(observation_refs)):
            raise MigrationConflictError("backfill_identity_duplicate")
        return observations

    def _load_shadow_inventory(self, plan: MigrationPlan) -> MigrationInventory:
        """Load the plan's frozen inventory, or raise if it's missing."""
        inventory = self._inventories.get(plan.inventory_ref or "")
        if inventory is None:
            raise MigrationPrerequisiteError("frozen_inventory_missing")
        return inventory

    def _validate_shadow_backfill_coverage(
        self,
        plan: MigrationPlan,
        source_records: tuple[CanonicalRecord, ...],
        snapshot: SourceSnapshot,
    ) -> tuple[bool, bool]:
        """Validate the backfill/inventory prerequisites for a shadow reconciliation.

        Returns ``(has_rejected, has_inventory_reject)``.
        """
        if any(
            record.source_revision != snapshot.revision for record in source_records
        ):
            raise MigrationConflictError("shadow_source_revision_mismatch")
        observations = self._load_shadow_observations(plan)
        inventory = self._load_shadow_inventory(plan)
        accepted = {
            item.canonical_ref
            for item in observations
            if item.disposition == "accepted"
        }
        if any(record.canonical_ref not in accepted for record in source_records):
            raise MigrationPrerequisiteError("shadow_record_not_backfilled")
        has_rejected = any(item.disposition != "accepted" for item in observations)
        has_inventory_reject = any(item.state != "present" for item in inventory.items)
        return has_rejected, has_inventory_reject

    def _shadow_delta_for(
        self,
        canonical_ref: str,
        source: CanonicalRecord | None,
        target: CanonicalRecord | None,
    ) -> ShadowDelta | None:
        """Compute the delta (if any) for one canonical_ref across source/target."""
        if source is None and target is not None:
            return self._delta(
                canonical_ref,
                kind="delete",
                source_digest=None,
                target_digest=target.record_digest,
            )
        if source is not None and target is None:
            return self._delta(
                canonical_ref,
                kind="add",
                source_digest=source.record_digest,
                target_digest=None,
            )
        if (
            source is not None
            and target is not None
            and (
                source.record_digest != target.record_digest
                or source.source_revision != target.source_revision
            )
        ):
            return self._delta(
                canonical_ref,
                kind="update",
                source_digest=source.record_digest,
                target_digest=target.record_digest,
            )
        return None

    def _compute_shadow_deltas(
        self,
        source_map: dict[str, CanonicalRecord],
        target_map: dict[str, CanonicalRecord],
    ) -> list[ShadowDelta]:
        """Compute the sorted delta set between source and target canonical records."""
        deltas: list[ShadowDelta] = []
        for canonical_ref in sorted(set(source_map) | set(target_map)):
            delta = self._shadow_delta_for(
                canonical_ref,
                source_map.get(canonical_ref),
                target_map.get(canonical_ref),
            )
            if delta is not None:
                deltas.append(delta)
        return deltas

    @staticmethod
    def _shadow_reconciliation_state(
        has_rejected: bool, has_inventory_reject: bool, deltas: list[ShadowDelta]
    ) -> Literal["clean", "needs_review", "blocked"]:
        if has_rejected or has_inventory_reject:
            return "blocked"
        if not deltas:
            return "clean"
        return "needs_review"

    @staticmethod
    def _build_shadow_reconciliation_candidate(
        plan: MigrationPlan,
        *,
        source_snapshot_digest: str,
        target_snapshot_digest: str,
        deltas: list[ShadowDelta],
        state: Literal["clean", "needs_review", "blocked"],
    ) -> ShadowReconciliation:
        """Build the (not-yet-persisted) ``ShadowReconciliation`` candidate record."""
        delta_tuple = tuple(deltas)
        reconciliation_ref = (
            f"reconcile:{source_snapshot_digest[7:23]}:{target_snapshot_digest[7:23]}"
        )
        delta_digest = canonical_digest(delta_tuple)
        return ShadowReconciliation(
            reconciliation_ref=reconciliation_ref,
            migration_ref=plan.migration_ref,
            source_snapshot_digest=source_snapshot_digest,
            target_snapshot_digest=target_snapshot_digest,
            deltas=delta_tuple,
            delta_count=len(delta_tuple),
            delta_digest=delta_digest,
            state=state,
            version=1,
            digest=_digest_for(
                {
                    "reconciliation_ref": reconciliation_ref,
                    "migration_ref": plan.migration_ref,
                    "source_snapshot_digest": source_snapshot_digest,
                    "target_snapshot_digest": target_snapshot_digest,
                    "delta_digest": delta_digest,
                    "state": state,
                    "version": 1,
                }
            ),
        )

    def _commit_shadow_reconciliation(
        self,
        plan: MigrationPlan,
        candidate: ShadowReconciliation,
        *,
        expected_plan_version: int,
    ) -> ShadowReconciliation:
        """Idempotent-replay check + plan transition + persist the reconciliation."""
        existing = self._reconciliations.get(candidate.reconciliation_ref)
        if existing is not None:
            if existing == candidate:
                return existing
            raise MigrationReplayError("shadow_reconciliation_replay_drift")
        if candidate.state == "clean" and plan.state == "backfill_observing":
            updated = self._next_plan(
                plan,
                expected_version=expected_plan_version,
                state="shadow_reconciled",
                stage="stage_2_shadow_reconciliation",
                reconciliation_ref=candidate.reconciliation_ref,
            )
            self._plans[plan.plan_ref] = updated
        else:
            self._cas_version(plan.version, expected_plan_version)
        self._reconciliations[candidate.reconciliation_ref] = candidate
        return candidate

    def reconcile_shadow(
        self,
        plan_ref: str,
        *,
        source_snapshot_digest: str,
        target_snapshot_digest: str,
        source_records: tuple[CanonicalRecord, ...],
        target_records: tuple[CanonicalRecord, ...],
        expected_plan_version: int,
    ) -> ShadowReconciliation:
        """Compute one sorted delta set; no delta is an approval decision."""

        with self._lock:
            plan = self._plan(plan_ref)
            snapshot = self._snapshot(plan.source_snapshot_ref, source_snapshot_digest)
            if plan.state not in {"backfill_observing", "shadow_reconciled"}:
                raise MigrationPrerequisiteError("shadow_stage_invalid")
            source_map, target_map = self._build_shadow_record_maps(
                source_records, target_records
            )
            has_rejected, has_inventory_reject = (
                self._validate_shadow_backfill_coverage(plan, source_records, snapshot)
            )
            deltas = self._compute_shadow_deltas(source_map, target_map)
            state = self._shadow_reconciliation_state(
                has_rejected, has_inventory_reject, deltas
            )
            if target_snapshot_digest == source_snapshot_digest and deltas:
                raise MigrationConflictError("same_snapshot_nonzero_delta")
            candidate = self._build_shadow_reconciliation_candidate(
                plan,
                source_snapshot_digest=source_snapshot_digest,
                target_snapshot_digest=target_snapshot_digest,
                deltas=deltas,
                state=state,
            )
            return self._commit_shadow_reconciliation(
                plan, candidate, expected_plan_version=expected_plan_version
            )

    def get_reconciliation(self, reconciliation_ref: str) -> ShadowReconciliation:
        with self._lock:
            reconciliation = self._reconciliations.get(reconciliation_ref)
            if reconciliation is None:
                raise MigrationNotFoundError()
            return reconciliation

    def get_write_fence(self, fence_ref: str) -> WriteCutoverFence:
        with self._lock:
            fence = self._write_fences.get(fence_ref)
            if fence is None:
                raise MigrationNotFoundError()
            return fence

    def _clean_reconciliation(self, plan: MigrationPlan) -> ShadowReconciliation:
        matches = [
            reconciliation
            for reconciliation in self._reconciliations.values()
            if reconciliation.migration_ref == plan.migration_ref
            and reconciliation.source_snapshot_digest == plan.source_snapshot_digest
            and reconciliation.state == "clean"
        ]
        if not matches:
            raise MigrationPrerequisiteError("clean_shadow_missing")
        return sorted(matches, key=lambda item: item.reconciliation_ref)[-1]

    def cutover_reads(
        self,
        plan_ref: str,
        cutover: CohortReadCutover,
        *,
        gate: PrerequisiteGate,
        expected_plan_version: int,
    ) -> MigrationPlan:
        """Activate one cohort only after a passed clean-shadow gate."""

        with self._lock:
            plan = self._plan(plan_ref)
            self._require_gate(
                gate,
                migration_ref=plan.migration_ref,
                source_snapshot_digest=plan.source_snapshot_digest,
                stage="stage_3_cohort_read_cutover",
            )
            self._clean_reconciliation(plan)
            if (
                cutover.migration_ref != plan.migration_ref
                or cutover.source_snapshot_digest != plan.source_snapshot_digest
                or cutover.gate_ref != gate.gate_ref
            ):
                raise MigrationConflictError("read_cutover_binding_mismatch")
            existing = self._read_cutovers.get(cutover.cutover_ref)
            same = self._same_or_conflict(
                existing,
                cutover,
                identity="read_cutover",
            )
            if same is not None:
                return plan
            if plan.state not in {"shadow_reconciled", "read_cutover"}:
                raise MigrationPrerequisiteError("read_cutover_stage_invalid")
            cohort_refs = tuple(sorted(set(plan.cohort_refs) | {cutover.cohort_ref}))
            updated = self._next_plan(
                plan,
                expected_version=expected_plan_version,
                state="read_cutover",
                stage="stage_3_cohort_read_cutover",
                cohort_refs=cohort_refs,
            )
            self._read_cutovers[cutover.cutover_ref] = cutover
            self._plans[plan.plan_ref] = updated
            return updated

    def fence_writes(
        self,
        plan_ref: str,
        fence: WriteCutoverFence,
        *,
        gate: PrerequisiteGate,
        expected_plan_version: int,
    ) -> MigrationPlan:
        """Install one target-authority fence and deny legacy writes."""

        with self._lock:
            plan = self._plan(plan_ref)
            self._require_gate(
                gate,
                migration_ref=plan.migration_ref,
                source_snapshot_digest=plan.source_snapshot_digest,
                stage="stage_4_write_cutover",
            )
            if (
                fence.migration_ref != plan.migration_ref
                or fence.source_snapshot_digest != plan.source_snapshot_digest
                or fence.authority_ref != plan.target_authority_ref
                or fence.gate_ref != gate.gate_ref
                or set(fence.cohort_refs) != set(plan.cohort_refs)
            ):
                raise MigrationConflictError("write_fence_binding_mismatch")
            existing = self._write_fences.get(fence.fence_ref)
            same = self._same_or_conflict(existing, fence, identity="write_fence")
            if same is not None:
                return plan
            if plan.state != "read_cutover":
                raise MigrationPrerequisiteError("write_cutover_stage_invalid")
            updated = self._next_plan(
                plan,
                expected_version=expected_plan_version,
                state="write_fenced",
                stage="stage_4_write_cutover",
                write_fence_ref=fence.fence_ref,
            )
            self._write_fences[fence.fence_ref] = fence
            self._plans[plan.plan_ref] = updated
            return updated

    def checkpoint_projection(
        self,
        plan_ref: str,
        checkpoint: ProjectionCheckpoint,
        *,
        gate: PrerequisiteGate,
        expected_plan_version: int,
    ) -> MigrationPlan:
        """Persist a complete deterministic projection checkpoint/rebuild plan."""

        with self._lock:
            plan = self._plan(plan_ref)
            self._require_gate(
                gate,
                migration_ref=plan.migration_ref,
                source_snapshot_digest=plan.source_snapshot_digest,
                stage="stage_5_projection_checkpoint",
            )
            if (
                checkpoint.migration_ref != plan.migration_ref
                or checkpoint.source_snapshot_digest != plan.source_snapshot_digest
                or checkpoint.gate_ref != gate.gate_ref
                or checkpoint.state != "complete"
            ):
                raise MigrationConflictError("projection_checkpoint_binding_mismatch")
            existing = self._checkpoints.get(checkpoint.checkpoint_ref)
            same = self._same_or_conflict(
                existing,
                checkpoint,
                identity="projection_checkpoint",
            )
            if same is not None:
                return plan
            if plan.state != "write_fenced" or plan.write_fence_ref is None:
                raise MigrationPrerequisiteError("projection_checkpoint_stage_invalid")
            fence = self._write_fences.get(plan.write_fence_ref)
            if fence is None or fence.state != "active":
                raise MigrationPrerequisiteError("active_write_fence_missing")
            updated = self._next_plan(
                plan,
                expected_version=expected_plan_version,
                state="checkpointed",
                stage="stage_5_projection_checkpoint",
                checkpoint_ref=checkpoint.checkpoint_ref,
            )
            self._checkpoints[checkpoint.checkpoint_ref] = checkpoint
            self._plans[plan.plan_ref] = updated
            return updated

    def get_checkpoint(self, checkpoint_ref: str) -> ProjectionCheckpoint:
        with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_ref)
            if checkpoint is None:
                raise MigrationNotFoundError()
            return checkpoint

    @staticmethod
    def _retirement_proof_invalid(
        retirement: LegacyRetirement, plan: MigrationPlan, gate: PrerequisiteGate
    ) -> bool:
        """Whether ``retirement`` fails the explicit zero-consumer retirement-proof contract."""
        return (
            retirement.migration_ref != plan.migration_ref
            or retirement.source_snapshot_digest != plan.source_snapshot_digest
            or retirement.write_fence_ref != plan.write_fence_ref
            or retirement.checkpoint_ref != plan.checkpoint_ref
            or retirement.gate_ref != gate.gate_ref
            or retirement.state != "retired"
            or retirement.remaining_consumers != 0
            or retirement.remaining_facades != 0
        )

    def retire_legacy(
        self,
        plan_ref: str,
        retirement: LegacyRetirement,
        *,
        gate: PrerequisiteGate,
        expected_plan_version: int,
    ) -> MigrationPlan:
        """Retire compatibility facades only with an explicit zero-consumer proof."""

        with self._lock:
            plan = self._plan(plan_ref)
            self._require_gate(
                gate,
                migration_ref=plan.migration_ref,
                source_snapshot_digest=plan.source_snapshot_digest,
                stage="legacy_retirement",
            )
            if self._retirement_proof_invalid(retirement, plan, gate):
                raise MigrationConflictError("legacy_retirement_proof_invalid")
            existing = self._retirements.get(retirement.retirement_ref)
            same = self._same_or_conflict(
                existing,
                retirement,
                identity="legacy_retirement",
            )
            if same is not None:
                return plan
            if plan.state != "checkpointed":
                raise MigrationPrerequisiteError("legacy_retirement_stage_invalid")
            updated = self._next_plan(
                plan,
                expected_version=expected_plan_version,
                state="retired",
                stage="legacy_retirement",
                retirement_ref=retirement.retirement_ref,
            )
            self._retirements[retirement.retirement_ref] = retirement
            self._plans[plan.plan_ref] = updated
            return updated

    def get_retirement(self, retirement_ref: str) -> LegacyRetirement:
        with self._lock:
            retirement = self._retirements.get(retirement_ref)
            if retirement is None:
                raise MigrationNotFoundError()
            return retirement

    def _rolled_back_fence_for(
        self, plan: MigrationPlan, rollback: MigrationRollback
    ) -> WriteCutoverFence | None:
        """Build the rolled-back ``WriteCutoverFence`` record when the plan has an active fence."""
        if plan.state not in {"write_fenced", "checkpointed"}:
            return None
        if rollback.fence_ref != plan.write_fence_ref:
            raise MigrationConflictError("rollback_fence_binding_invalid")
        fence = self._write_fences.get(plan.write_fence_ref or "")
        if fence is None:
            raise MigrationPrerequisiteError("rollback_fence_missing")
        if fence.version >= _MAX_VERSION:
            raise MigrationCasConflictError("migration_version_exhausted")
        fence_data = fence.model_dump(mode="python")
        fence_data["state"] = "rolled_back"
        fence_data["version"] = fence.version + 1
        fence_data["digest"] = _digest_for(fence_data)
        return WriteCutoverFence(**fence_data)

    @staticmethod
    def _applied_rollback(rollback: MigrationRollback) -> MigrationRollback:
        """Mark a ``requested`` rollback record ``applied`` (no-op for any other decision)."""
        if rollback.decision != "requested":
            return rollback
        applied_data = rollback.model_dump(mode="python")
        applied_data["decision"] = "applied"
        applied_data["digest"] = _digest_for(applied_data)
        return MigrationRollback(**applied_data)

    def rollback(
        self,
        plan_ref: str,
        rollback: MigrationRollback,
        *,
        expected_plan_version: int,
    ) -> MigrationPlan:
        """Apply only an explicit rollback record and fence any active writes."""

        with self._lock:
            plan = self._plan(plan_ref)
            if plan.state == "retired":
                raise MigrationPrerequisiteError("retired_migration_not_rollbackable")
            if (
                rollback.migration_ref != plan.migration_ref
                or rollback.source_snapshot_digest != plan.source_snapshot_digest
                or rollback.decision not in {"requested", "applied"}
            ):
                raise MigrationConflictError("rollback_binding_invalid")
            existing = self._rollbacks.get(rollback.rollback_ref)
            if existing is not None:
                if existing == rollback:
                    return plan
                raise MigrationReplayError("rollback_replay_drift")
            rolled_back_fence = self._rolled_back_fence_for(plan, rollback)
            rollback = self._applied_rollback(rollback)
            updated = self._next_plan(
                plan,
                expected_version=expected_plan_version,
                state="rolled_back",
                stage="rollback",
            )
            if rolled_back_fence is not None:
                self._write_fences[rolled_back_fence.fence_ref] = rolled_back_fence
            self._rollbacks[rollback.rollback_ref] = rollback
            self._plans[plan.plan_ref] = updated
            return updated

    def get_rollback(self, rollback_ref: str) -> MigrationRollback:
        with self._lock:
            rollback = self._rollbacks.get(rollback_ref)
            if rollback is None:
                raise MigrationNotFoundError()
            return rollback
