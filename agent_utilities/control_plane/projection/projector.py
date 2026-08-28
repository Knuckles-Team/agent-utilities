"""Ordered, fenced GraphOS projection and observation-promotion services.

``ProjectionService`` is downstream of the authority. It reads immutable
outbox envelopes, applies them to GraphOS, and only then advances a compare-
and-set checkpoint. A GraphOS error therefore leaves authority untouched and
the event eligible for retry. Keyset cursors are per aggregate and bounded.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import Field, StrictBool

from agent_utilities.protocols.epistemic_operations import ProtocolModel

from .models import (
    MAX_BATCH_SIZE,
    AtomicCommitReceipt,
    AuthoritativeMutation,
    DriftRecord,
    GraphProjectionReceipt,
    ObservationPromotion,
    ObservationPromotionPolicy,
    OutboxEnvelope,
    ProjectionCheckpoint,
    ProjectionOutcome,
    ProjectionScope,
    SummaryValue,
    sha256_digest,
)
from .repository import (
    AtomicAuthorityRepository,
    CheckpointConflict,
    CheckpointStore,
    DriftSink,
    GraphOSProjection,
    OutboxReader,
    ProjectionContractError,
    TombstoneReader,
    _commit_validated_change,
    _validate_event_matches_mutation,
)

__all__ = [
    "ProjectionBatchResult",
    "ProjectionCleanupResult",
    "ProjectionService",
    "promote_graph_observation",
]


class ProjectionBatchResult(ProtocolModel):
    """Bounded result for one aggregate keyset page."""

    result_version: Literal["control-plane-projection-batch.v1"] = (
        "control-plane-projection-batch.v1"
    )
    scope: ProjectionScope
    status: Literal["complete", "applied", "replayed", "blocked", "failed", "drift"]
    outcomes: tuple[ProjectionOutcome, ...] = Field(
        default=(), max_length=MAX_BATCH_SIZE
    )
    checkpoint: ProjectionCheckpoint | None = None
    has_more: StrictBool = False


class ProjectionCleanupResult(ProtocolModel):
    """Bounded tombstone-cleanup result; cleanup never advances authority."""

    result_version: Literal["control-plane-projection-cleanup.v1"] = (
        "control-plane-projection-cleanup.v1"
    )
    scope: ProjectionScope
    outcomes: tuple[ProjectionOutcome, ...] = Field(
        default=(), max_length=MAX_BATCH_SIZE
    )
    checkpoint: ProjectionCheckpoint | None = None
    has_more: StrictBool = False


_DRIFT_CODES = Literal[
    "graph_apply_failed",
    "checkpoint_conflict",
    "checkpoint_fence_lost",
    "sequence_gap",
    "identity_conflict",
    "scope_mismatch",
    "tombstone_cleanup_failed",
    "rebuild_failed",
    "reverse_sync_rejected",
]


@dataclass(frozen=True)
class _EventStep:
    """Outcome of applying one outbox event to the in-flight batch state."""

    outcome: ProjectionOutcome | None
    stop: bool
    current: ProjectionCheckpoint
    persisted: ProjectionCheckpoint | None


class _BatchAborted(Exception):
    """Internal signal: a batch must return this result without applying events."""

    def __init__(self, result: ProjectionBatchResult) -> None:
        super().__init__(result.status)
        self.result = result


class _CleanupAborted(Exception):
    """Internal signal: a cleanup pass must return this result immediately."""

    def __init__(self, result: ProjectionCleanupResult) -> None:
        super().__init__(result.scope.key)
        self.result = result


class ProjectionService:
    """Apply an authority outbox to GraphOS with ordering and fencing."""

    def __init__(
        self,
        *,
        projector_id: str,
        outbox: OutboxReader,
        checkpoints: CheckpointStore,
        graph: GraphOSProjection,
        drift: DriftSink,
        tombstones: TombstoneReader | None = None,
    ) -> None:
        if not projector_id:
            raise ValueError("projector_id_must_not_be_empty")
        self.projector_id = projector_id
        self._outbox = outbox
        self._checkpoints = checkpoints
        self._graph = graph
        self._drift = drift
        self._tombstones = tombstones

    @staticmethod
    def _check_limit(limit: int) -> None:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= MAX_BATCH_SIZE
        ):
            raise ProjectionContractError("projection_batch_limit_out_of_bounds")

    @staticmethod
    def _check_fence(fence_token: int) -> None:
        if (
            not isinstance(fence_token, int)
            or isinstance(fence_token, bool)
            or fence_token < 1
        ):
            raise ProjectionContractError("projection_fence_token_invalid")

    def _read_checkpoint(
        self,
        scope: ProjectionScope,
        fence_token: int,
    ) -> ProjectionCheckpoint:
        self._check_fence(fence_token)
        current = self._checkpoints.read_checkpoint(scope)
        if current is None:
            return ProjectionCheckpoint.initial(
                projector_id=self.projector_id,
                scope=scope,
                fence_token=fence_token,
            )
        if current.projector_id != self.projector_id or current.scope != scope:
            raise ProjectionContractError("checkpoint_scope_or_projector_mismatch")
        if fence_token < current.fence_token:
            raise CheckpointConflict("projection_fence_is_stale")
        return current

    @staticmethod
    def _status(
        outcomes: tuple[ProjectionOutcome, ...],
    ) -> Literal["complete", "applied", "replayed", "blocked", "failed", "drift"]:
        if not outcomes:
            return "complete"
        if any(
            item.status in {"gap", "conflict", "rejected", "drift"} for item in outcomes
        ):
            return "drift"
        if any(item.status == "failed" for item in outcomes):
            return "failed"
        if all(item.status == "replayed" for item in outcomes):
            return "replayed"
        return "applied"

    def _record_drift(
        self,
        *,
        reason_code: _DRIFT_CODES,
        scope: ProjectionScope,
        sequence: int,
        event: OutboxEnvelope | None = None,
        expected_digest: str | None = None,
        observed_digest: str | None = None,
    ) -> None:
        self._drift.record(
            DriftRecord(
                reason_code=reason_code,
                projector_id=self.projector_id,
                aggregate_type=scope.aggregate_type,
                aggregate_id=scope.aggregate_id,
                sequence=max(sequence, 0),
                event_id=event.event_id if event is not None else None,
                expected_digest=expected_digest,
                observed_digest=observed_digest,
                repairable=True,
                recorded_at=datetime.now(UTC),
            )
        )

    @staticmethod
    def _outcome(
        status: Literal[
            "applied",
            "replayed",
            "rejected",
            "gap",
            "conflict",
            "failed",
            "drift",
            "cleaned",
        ],
        event: OutboxEnvelope,
        reason_code: str | None = None,
    ) -> ProjectionOutcome:
        return ProjectionOutcome(
            status=status,
            aggregate_type=event.aggregate_type,
            aggregate_id=event.aggregate_id,
            sequence=event.sequence,
            event_id=event.event_id,
            reason_code=reason_code,
        )

    def _failure_result(
        self,
        scope: ProjectionScope,
        *,
        reason_code: _DRIFT_CODES,
        checkpoint: ProjectionCheckpoint | None = None,
        sequence: int = 0,
        event: OutboxEnvelope | None = None,
    ) -> ProjectionBatchResult:
        self._record_drift(
            reason_code=reason_code,
            scope=scope,
            sequence=sequence,
            event=event,
        )
        return ProjectionBatchResult(
            scope=scope,
            status="failed"
            if reason_code in {"graph_apply_failed", "rebuild_failed"}
            else "drift",
            checkpoint=checkpoint,
        )

    def project_scope(
        self,
        scope: ProjectionScope,
        *,
        fence_token: int,
        limit: int = MAX_BATCH_SIZE,
    ) -> ProjectionBatchResult:
        """Project one bounded keyset page, isolating failures to this scope."""

        self._check_limit(limit)
        try:
            current, persisted, events = self._open_batch(scope, fence_token, limit)
        except _BatchAborted as aborted:
            return aborted.result

        outcomes: list[ProjectionOutcome] = []
        for event in events:
            step = self._step_event(scope, event, current, persisted, fence_token)
            if step.outcome is not None:
                outcomes.append(step.outcome)
            current, persisted = step.current, step.persisted
            if step.stop:
                break

        typed_outcomes = tuple(outcomes)
        status = self._status(typed_outcomes)
        return ProjectionBatchResult(
            scope=scope,
            status=status,
            outcomes=typed_outcomes,
            checkpoint=current,
            has_more=len(events) == limit
            and status in {"complete", "applied", "replayed"},
        )

    def _open_batch(
        self,
        scope: ProjectionScope,
        fence_token: int,
        limit: int,
    ) -> tuple[
        ProjectionCheckpoint, ProjectionCheckpoint | None, tuple[OutboxEnvelope, ...]
    ]:
        """Read the checkpoint and the next event page, or abort the batch."""

        try:
            current = self._read_checkpoint(scope, fence_token)
        except CheckpointConflict:
            raise _BatchAborted(
                self._failure_result(scope, reason_code="checkpoint_fence_lost")
            ) from None
        except Exception:
            raise _BatchAborted(
                self._failure_result(scope, reason_code="checkpoint_conflict")
            ) from None

        # ``current`` may be a synthetic, never-persisted starting point
        # (``ProjectionCheckpoint.initial``) when the store has nothing for
        # this scope yet. The CAS ``expected`` argument to ``save_checkpoint``
        # must reflect what is actually stored -- ``None`` before the first
        # event -- never that synthetic value, or the very first save always
        # loses the compare-and-swap against an empty store.
        persisted = current if current.last_sequence > 0 else None

        try:
            events = tuple(self._outbox.read_after(scope, current.last_sequence, limit))
        except Exception:
            raise _BatchAborted(
                self._failure_result(
                    scope,
                    reason_code="graph_apply_failed",
                    checkpoint=current,
                    sequence=current.last_sequence + 1,
                )
            ) from None
        if len(events) > limit:
            raise _BatchAborted(
                self._failure_result(
                    scope,
                    reason_code="sequence_gap",
                    checkpoint=current,
                    sequence=current.last_sequence + 1,
                )
            )
        return current, persisted, events

    def _step_event(
        self,
        scope: ProjectionScope,
        event: OutboxEnvelope,
        current: ProjectionCheckpoint,
        persisted: ProjectionCheckpoint | None,
        fence_token: int,
    ) -> _EventStep:
        """Advance the batch by exactly one event, or signal why it must stop."""

        step = self._validate_event_identity(scope, event, current, persisted)
        if step is not None:
            return step
        step = self._classify_event_sequence(scope, event, current, persisted)
        if step is not None:
            return step
        return self._apply_event(scope, event, current, persisted, fence_token)

    def _validate_event_identity(
        self,
        scope: ProjectionScope,
        event: OutboxEnvelope,
        current: ProjectionCheckpoint,
        persisted: ProjectionCheckpoint | None,
    ) -> _EventStep | None:
        """Reject a malformed envelope or one outside this scope; else None."""

        if not isinstance(event, OutboxEnvelope):
            self._record_drift(
                reason_code="identity_conflict",
                scope=scope,
                sequence=current.last_sequence + 1,
            )
            return _EventStep(
                outcome=None, stop=True, current=current, persisted=persisted
            )
        if (
            event.aggregate_type != scope.aggregate_type
            or event.aggregate_id != scope.aggregate_id
        ):
            outcome = self._outcome("rejected", event, "scope_mismatch")
            self._record_drift(
                reason_code="scope_mismatch",
                scope=scope,
                sequence=event.sequence,
                event=event,
            )
            return _EventStep(
                outcome=outcome, stop=True, current=current, persisted=persisted
            )
        return None

    def _classify_event_sequence(
        self,
        scope: ProjectionScope,
        event: OutboxEnvelope,
        current: ProjectionCheckpoint,
        persisted: ProjectionCheckpoint | None,
    ) -> _EventStep | None:
        """Handle a replayed, conflicting, or gapped event; else None to apply it."""

        if event.sequence <= current.last_sequence:
            if (
                event.sequence == current.last_sequence
                and event.event_id == current.last_event_id
                and event.event_digest == current.last_event_digest
            ):
                outcome = self._outcome("replayed", event)
                return _EventStep(
                    outcome=outcome, stop=False, current=current, persisted=persisted
                )
            outcome = self._outcome("conflict", event, "out_of_order_event")
            self._record_drift(
                reason_code="identity_conflict",
                scope=scope,
                sequence=event.sequence,
                event=event,
                expected_digest=current.last_event_digest,
                observed_digest=event.event_digest,
            )
            return _EventStep(
                outcome=outcome, stop=True, current=current, persisted=persisted
            )

        expected_sequence = current.last_sequence + 1
        if event.sequence != expected_sequence:
            outcome = self._outcome("gap", event, "sequence_gap")
            self._record_drift(
                reason_code="sequence_gap",
                scope=scope,
                sequence=event.sequence,
                event=event,
            )
            return _EventStep(
                outcome=outcome, stop=True, current=current, persisted=persisted
            )
        return None

    def _apply_event(
        self,
        scope: ProjectionScope,
        event: OutboxEnvelope,
        current: ProjectionCheckpoint,
        persisted: ProjectionCheckpoint | None,
        fence_token: int,
    ) -> _EventStep:
        """Apply the next expected event to GraphOS and advance the checkpoint."""

        try:
            receipt = self._graph.apply_event(event, fence_token)
            if not isinstance(receipt, GraphProjectionReceipt):
                raise ProjectionContractError("graph_projection_receipt_invalid")
            if receipt.event_id != event.event_id or not (
                receipt.applied or receipt.replayed
            ):
                raise ProjectionContractError("graph_projection_receipt_mismatch")
            next_checkpoint = ProjectionCheckpoint.after_event(
                prior=current,
                event=event,
                fence_token=fence_token,
            )
            self._checkpoints.save_checkpoint(scope, persisted, next_checkpoint)
        except CheckpointConflict:
            outcome = self._outcome("drift", event, "checkpoint_conflict")
            self._record_drift(
                reason_code="checkpoint_conflict",
                scope=scope,
                sequence=event.sequence,
                event=event,
            )
            return _EventStep(
                outcome=outcome, stop=True, current=current, persisted=persisted
            )
        except Exception:
            outcome = self._outcome("failed", event, "graph_apply_failed")
            self._record_drift(
                reason_code="graph_apply_failed",
                scope=scope,
                sequence=event.sequence,
                event=event,
            )
            return _EventStep(
                outcome=outcome, stop=True, current=current, persisted=persisted
            )

        outcome = self._outcome("applied", event)
        return _EventStep(
            outcome=outcome,
            stop=False,
            current=next_checkpoint,
            persisted=next_checkpoint,
        )

    def project(
        self,
        scopes: Iterable[ProjectionScope],
        *,
        fence_token: int,
        limit: int = MAX_BATCH_SIZE,
    ) -> tuple[ProjectionBatchResult, ...]:
        """Project independent scopes in deterministic aggregate-key order."""

        self._check_limit(limit)
        ordered = tuple(sorted(scopes, key=lambda scope: scope.key))
        return tuple(
            self.project_scope(scope, fence_token=fence_token, limit=limit)
            for scope in ordered
        )

    def rebuild_scope(
        self,
        scope: ProjectionScope,
        *,
        fence_token: int,
        limit: int = MAX_BATCH_SIZE,
    ) -> ProjectionBatchResult:
        """Reset only GraphOS and its cursor, then replay from sequence zero."""

        self._check_limit(limit)
        try:
            self._graph.reset_projection(scope, fence_token)
            self._checkpoints.reset_checkpoint(scope, fence_token)
        except Exception:
            try:
                current = self._checkpoints.read_checkpoint(scope)
            except Exception:
                current = None
            return self._failure_result(
                scope,
                reason_code="rebuild_failed",
                checkpoint=current,
            )
        return self.project_scope(scope, fence_token=fence_token, limit=limit)

    def cleanup_tombstones(
        self,
        scope: ProjectionScope,
        *,
        fence_token: int,
        before_sequence: int,
        limit: int = MAX_BATCH_SIZE,
    ) -> ProjectionCleanupResult:
        """Clean tombstones only after the durable cursor has passed them."""

        self._check_limit(limit)
        if before_sequence < 1:
            raise ProjectionContractError("tombstone_cleanup_boundary_invalid")
        if self._tombstones is None:
            raise ProjectionContractError("tombstone_reader_required")
        try:
            checkpoint, events = self._open_cleanup_batch(
                scope, self._tombstones, fence_token, before_sequence, limit
            )
        except _CleanupAborted as aborted:
            return aborted.result

        outcomes: list[ProjectionOutcome] = []
        for event in events:
            outcome = self._step_tombstone_event(scope, event, checkpoint, fence_token)
            if outcome is not None:
                outcomes.append(outcome)
        return ProjectionCleanupResult(
            scope=scope,
            outcomes=tuple(outcomes),
            checkpoint=checkpoint,
            has_more=len(events) == limit,
        )

    def _open_cleanup_batch(
        self,
        scope: ProjectionScope,
        tombstones: TombstoneReader,
        fence_token: int,
        before_sequence: int,
        limit: int,
    ) -> tuple[ProjectionCheckpoint, tuple[OutboxEnvelope, ...]]:
        """Read the checkpoint and due tombstones, or abort the cleanup pass."""

        try:
            checkpoint = self._read_checkpoint(scope, fence_token)
            events = tuple(
                tombstones.read_tombstones_before(scope, before_sequence, limit)
            )
        except CheckpointConflict:
            self._record_drift(
                reason_code="checkpoint_fence_lost",
                scope=scope,
                sequence=0,
            )
            raise _CleanupAborted(
                ProjectionCleanupResult(scope=scope, checkpoint=None)
            ) from None
        except Exception:
            self._record_drift(
                reason_code="tombstone_cleanup_failed",
                scope=scope,
                sequence=0,
            )
            raise _CleanupAborted(
                ProjectionCleanupResult(scope=scope, checkpoint=None)
            ) from None
        if len(events) > limit:
            raise ProjectionContractError("tombstone_reader_exceeded_limit")
        return checkpoint, events

    def _step_tombstone_event(
        self,
        scope: ProjectionScope,
        event: OutboxEnvelope,
        checkpoint: ProjectionCheckpoint,
        fence_token: int,
    ) -> ProjectionOutcome | None:
        """Clean up one due tombstone; return its outcome, or None to skip it."""

        if not isinstance(event, OutboxEnvelope):
            self._record_drift(
                reason_code="tombstone_cleanup_failed",
                scope=scope,
                sequence=0,
            )
            return None
        if event.operation != "tombstone" or event.sequence > checkpoint.last_sequence:
            self._record_drift(
                reason_code="tombstone_cleanup_failed",
                scope=scope,
                sequence=event.sequence,
                event=event,
            )
            return self._outcome("rejected", event, "tombstone_not_durable")
        try:
            self._graph.cleanup_tombstone(event, fence_token)
        except Exception:
            self._record_drift(
                reason_code="tombstone_cleanup_failed",
                scope=scope,
                sequence=event.sequence,
                event=event,
            )
            return self._outcome("failed", event, "tombstone_cleanup_failed")
        return self._outcome("cleaned", event)


def _validate_promotion_policy(
    observation: ObservationPromotion,
    policy: ObservationPromotionPolicy,
) -> None:
    """Reject a promotion the policy does not admit at all."""

    if not policy.enabled:
        raise ProjectionContractError("observation_promotion_disabled")
    if observation.observation_type not in policy.allowed_observation_types:
        raise ProjectionContractError("observation_type_not_allowed")
    if policy.require_evidence_ref and observation.evidence_ref is None:
        raise ProjectionContractError("observation_evidence_reference_required")


def _resolve_promotion_time(
    observation: ObservationPromotion,
    policy: ObservationPromotionPolicy,
    now: datetime | None,
) -> datetime:
    """Resolve and validate the promotion instant against the policy's max age."""

    current_time = now or datetime.now(UTC)
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise ProjectionContractError("promotion_time_requires_timezone")
    age = current_time.astimezone(UTC) - observation.observed_at.astimezone(UTC)
    if age < timedelta(0) or age.total_seconds() > policy.max_age_seconds:
        raise ProjectionContractError("observation_is_stale")
    return current_time


def _reconciled_promotion_summary(
    observation: ObservationPromotion,
    policy: ObservationPromotionPolicy,
) -> dict[str, SummaryValue]:
    """Merge the policy ref into the observation summary, rejecting a conflict."""

    summary: dict[str, SummaryValue] = dict(observation.summary)
    prior_policy = summary.get("policy_ref")
    if prior_policy is not None and prior_policy != policy.policy_ref:
        raise ProjectionContractError("observation_policy_reference_conflict")
    summary["policy_ref"] = policy.policy_ref
    return summary


def promote_graph_observation(
    repository: AtomicAuthorityRepository,
    observation: ObservationPromotion,
    policy: ObservationPromotionPolicy,
    *,
    sequence: int,
    authoritative_revision: int,
    expected_revision: int | None = None,
    now: datetime | None = None,
) -> AtomicCommitReceipt:
    """Promote one graph observation only through an explicit policy."""

    _validate_promotion_policy(observation, policy)
    current_time = _resolve_promotion_time(observation, policy, now)
    summary = _reconciled_promotion_summary(observation, policy)
    change_digest = sha256_digest(
        {
            "evidence_ref": observation.evidence_ref,
            "observation_digest": observation.observation_digest,
            "observation_id": observation.observation_id,
            "policy_ref": policy.policy_ref,
        }
    )
    mutation = AuthoritativeMutation.create_observation_promotion(
        aggregate_type=observation.aggregate_type,
        aggregate_id=observation.aggregate_id,
        sequence=sequence,
        authoritative_revision=authoritative_revision,
        expected_revision=expected_revision,
        change_digest=change_digest,
        aggregate_digest=observation.observation_digest,
        summary=summary,
    )
    event = OutboxEnvelope.create(
        aggregate_type=mutation.aggregate_type,
        aggregate_id=mutation.aggregate_id,
        sequence=mutation.sequence,
        event_type="observation.promoted",
        operation="upsert",
        authoritative_revision=mutation.authoritative_revision,
        payload_digest=mutation.change_digest,
        aggregate_digest=mutation.aggregate_digest,
        summary=mutation.summary,
        occurred_at=current_time,
    )
    _validate_event_matches_mutation(mutation, event)
    return _commit_validated_change(repository, mutation, event)
