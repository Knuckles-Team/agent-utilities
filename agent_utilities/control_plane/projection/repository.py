"""Persistence and GraphOS adapter seams for control-plane projection.

The interfaces in this module are deliberately small.  A relational adapter
implements one atomic authority-plus-outbox call; it must not expose a second
write path.  A GraphOS adapter receives only an :class:`OutboxEnvelope` and a
fencing term.  Checkpoint advancement is a separate compare-and-set operation,
so a failed projection never changes relational authority and a lost lease
cannot move a cursor.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    AtomicCommitReceipt,
    AuthoritativeMutation,
    GraphProjectionReceipt,
    OutboxEnvelope,
    ProjectionCheckpoint,
    ProjectionScope,
)

__all__ = [
    "AtomicAuthorityRepository",
    "CheckpointConflict",
    "CheckpointStore",
    "GraphOSProjection",
    "DriftSink",
    "OutboxReader",
    "ProjectionContractError",
    "ProjectionRepositoryUnavailable",
    "ReverseSyncRejected",
    "TombstoneReader",
    "commit_authoritative_change",
    "reject_reverse_sync",
]


class ProjectionContractError(ValueError):
    """A caller or adapter violated the projection contract."""


class ProjectionRepositoryUnavailable(RuntimeError):
    """An authority, outbox, checkpoint or GraphOS adapter is unavailable."""


class CheckpointConflict(ProjectionContractError):
    """A checkpoint CAS or fencing precondition failed."""


class ReverseSyncRejected(ProjectionContractError):
    """GraphOS projection data cannot become authority through reverse sync."""


@runtime_checkable
class AtomicAuthorityRepository(Protocol):
    """The sole persistence seam for an authoritative mutation.

    ``commit_authority_and_outbox`` is one transaction from the adapter's point
    of view: the authoritative row and its event are committed together, or
    neither is visible.  ``event`` is ``None`` only for a rollback, which is a
    local authority action and intentionally emits no projection event.
    """

    def commit_authority_and_outbox(
        self,
        mutation: AuthoritativeMutation,
        event: OutboxEnvelope | None,
    ) -> AtomicCommitReceipt:
        """Atomically commit authority and its optional outbox event."""


@runtime_checkable
class OutboxReader(Protocol):
    """Bounded keyset reader over immutable events for one aggregate scope."""

    def read_after(
        self,
        scope: ProjectionScope,
        after_sequence: int,
        limit: int,
    ) -> tuple[OutboxEnvelope, ...]:
        """Return at most ``limit`` events strictly after ``after_sequence``."""


@runtime_checkable
class TombstoneReader(Protocol):
    """Bounded reader for already-projected tombstones eligible for cleanup."""

    def read_tombstones_before(
        self,
        scope: ProjectionScope,
        before_sequence: int,
        limit: int,
    ) -> tuple[OutboxEnvelope, ...]:
        """Return tombstones in ascending sequence order, never raw records."""


@runtime_checkable
class CheckpointStore(Protocol):
    """Durable fenced keyset checkpoint authority."""

    def read_checkpoint(self, scope: ProjectionScope) -> ProjectionCheckpoint | None:
        """Read one checkpoint or ``None`` before the first event."""

    def save_checkpoint(
        self,
        scope: ProjectionScope,
        expected: ProjectionCheckpoint | None,
        checkpoint: ProjectionCheckpoint,
    ) -> None:
        """CAS-save a checkpoint and reject stale fence/expected state."""

    def reset_checkpoint(self, scope: ProjectionScope, fence_token: int) -> None:
        """Reset one projection cursor under the current fencing term."""


@runtime_checkable
class GraphOSProjection(Protocol):
    """GraphOS write projection with event-id idempotency."""

    def apply_event(
        self, event: OutboxEnvelope, fence_token: int
    ) -> GraphProjectionReceipt:
        """Apply one event, returning a replay receipt for an existing event ID."""

    def reset_projection(self, scope: ProjectionScope, fence_token: int) -> None:
        """Clear only this projection scope for deterministic rebuild."""

    def cleanup_tombstone(self, event: OutboxEnvelope, fence_token: int) -> None:
        """Remove one previously projected tombstone under the same fence."""


@runtime_checkable
class DriftSink(Protocol):
    """Durable/observable sink for bounded projection discrepancies."""

    def record(self, drift: object) -> None:
        """Persist one typed drift record without raw exception text."""


def reject_reverse_sync(*, source: str) -> None:
    """Reject an attempted GraphOS-to-authority write before any adapter call."""

    del source
    raise ReverseSyncRejected("reverse_sync_is_not_an_authority_path")


def _validate_event_matches_mutation(
    mutation: AuthoritativeMutation,
    event: OutboxEnvelope,
) -> None:
    expected_operation = "tombstone" if mutation.operation == "delete" else "upsert"
    if event.operation != expected_operation:
        raise ProjectionContractError("event_operation_does_not_match_mutation")
    if (
        event.aggregate_type != mutation.aggregate_type
        or event.aggregate_id != mutation.aggregate_id
        or event.sequence != mutation.sequence
        or event.authoritative_revision != mutation.authoritative_revision
    ):
        raise ProjectionContractError("event_identity_does_not_match_mutation")
    if event.payload_digest != mutation.change_digest:
        raise ProjectionContractError("event_payload_digest_does_not_match_mutation")
    if event.aggregate_digest != mutation.aggregate_digest:
        raise ProjectionContractError("event_aggregate_digest_does_not_match_mutation")
    if event.summary != mutation.summary:
        raise ProjectionContractError("event_summary_does_not_match_mutation")


def commit_authoritative_change(
    repository: AtomicAuthorityRepository,
    mutation: AuthoritativeMutation,
    event: OutboxEnvelope | None,
) -> AtomicCommitReceipt:
    """Commit authority and outbox once, preserving rollback/no-event semantics.

    This helper has no retry loop and no fallback repository.  The caller's
    adapter owns transactionality and may return ``replayed=True`` when the
    exact mutation/event was already committed after a crash.
    """

    if mutation.origin != "authoritative":
        raise ProjectionContractError("observation_promotion_requires_policy_path")
    return _commit_validated_change(repository, mutation, event)


def _commit_validated_change(
    repository: AtomicAuthorityRepository,
    mutation: AuthoritativeMutation,
    event: OutboxEnvelope | None,
) -> AtomicCommitReceipt:
    """Commit an already-policy-validated mutation through the one adapter seam."""

    if mutation.operation == "rollback":
        if event is not None:
            raise ProjectionContractError("rollback_must_not_emit_event")
    elif event is None:
        raise ProjectionContractError("forward_mutation_requires_outbox_event")
    else:
        _validate_event_matches_mutation(mutation, event)
    try:
        receipt = repository.commit_authority_and_outbox(mutation, event)
    except ReverseSyncRejected:
        raise
    except ProjectionContractError:
        raise
    except ProjectionRepositoryUnavailable:
        raise
    except Exception as exc:
        raise ProjectionRepositoryUnavailable(
            "authority_transaction_unavailable"
        ) from exc
    if receipt.mutation_id != mutation.mutation_id:
        raise ProjectionContractError("authority_receipt_mutation_mismatch")
    if (
        receipt.aggregate_type != mutation.aggregate_type
        or receipt.aggregate_id != mutation.aggregate_id
        or receipt.sequence != mutation.sequence
        or receipt.authoritative_revision != mutation.authoritative_revision
    ):
        raise ProjectionContractError("authority_receipt_identity_mismatch")
    if event is None and receipt.event_id is not None:
        raise ProjectionContractError("rollback_receipt_contains_event")
    if event is not None and receipt.event_id != event.event_id:
        raise ProjectionContractError("authority_receipt_event_mismatch")
    return receipt
