"""Focused contracts for the transactional control-plane projection seam.

These fixtures are deliberately in-memory and persistence-independent. They
exercise the protocol boundary only; the root validation track owns real
relational and GraphOS adapter acceptance.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import pytest

from agent_utilities.control_plane.projection import (
    AtomicCommitReceipt,
    AuthoritativeMutation,
    CheckpointConflict,
    DriftRecord,
    GraphProjectionReceipt,
    ObservationPromotion,
    ObservationPromotionPolicy,
    OutboxEnvelope,
    ProjectionService,
    ProjectionScope,
    Tombstone,
    commit_authoritative_change,
    promote_graph_observation,
    reject_reverse_sync,
    sha256_digest,
)


NOW = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)


def _mutation(
    sequence: int,
    *,
    operation: Literal["upsert", "delete", "rollback"] = "upsert",
    summary: dict[str, object] | None = None,
) -> AuthoritativeMutation:
    change_digest = sha256_digest({"change": sequence})
    return AuthoritativeMutation.create(
        aggregate_type="service",
        aggregate_id="service:one",
        sequence=sequence,
        operation=operation,
        authoritative_revision=sequence,
        change_digest=change_digest,
        aggregate_digest=sha256_digest({"state": sequence}),
        expected_revision=sequence - 1 if sequence > 1 else None,
        summary=summary or {"status": "ready", "revision": sequence},
    )


def _event(mutation: AuthoritativeMutation, *, tombstone: bool = False) -> OutboxEnvelope:
    event_operation: Literal["upsert", "tombstone"] = (
        "tombstone" if tombstone else "upsert"
    )
    return OutboxEnvelope.create(
        aggregate_type=mutation.aggregate_type,
        aggregate_id=mutation.aggregate_id,
        sequence=mutation.sequence,
        event_type="service.changed",
        operation=event_operation,
        authoritative_revision=mutation.authoritative_revision,
        payload_digest=mutation.change_digest,
        aggregate_digest=mutation.aggregate_digest,
        summary=mutation.summary,
        occurred_at=NOW,
        tombstone=(
            Tombstone(
                reason_code="removed",
                target_digest=mutation.aggregate_digest,
                deleted_at=NOW,
            )
            if tombstone
            else None
        ),
    )


class AuthorityFixture:
    def __init__(self) -> None:
        self.calls: list[tuple[AuthoritativeMutation, OutboxEnvelope | None]] = []

    def commit_authority_and_outbox(
        self,
        mutation: AuthoritativeMutation,
        event: OutboxEnvelope | None,
    ) -> AtomicCommitReceipt:
        self.calls.append((mutation, event))
        return AtomicCommitReceipt(
            mutation_id=mutation.mutation_id,
            aggregate_type=mutation.aggregate_type,
            aggregate_id=mutation.aggregate_id,
            authoritative_revision=mutation.authoritative_revision,
            sequence=mutation.sequence,
            event_id=event.event_id if event is not None else None,
            committed=True,
        )


class OutboxFixture:
    def __init__(self, events: tuple[OutboxEnvelope, ...]) -> None:
        self.events = events

    def read_after(
        self,
        scope: ProjectionScope,
        after_sequence: int,
        limit: int,
    ) -> tuple[OutboxEnvelope, ...]:
        return tuple(
            event
            for event in self.events
            if event.aggregate_type == scope.aggregate_type
            and event.aggregate_id == scope.aggregate_id
            and event.sequence > after_sequence
        )[:limit]

    def read_tombstones_before(
        self,
        scope: ProjectionScope,
        before_sequence: int,
        limit: int,
    ) -> tuple[OutboxEnvelope, ...]:
        return tuple(
            event
            for event in self.events
            if event.aggregate_type == scope.aggregate_type
            and event.aggregate_id == scope.aggregate_id
            and event.operation == "tombstone"
            and event.sequence < before_sequence
        )[:limit]


class CheckpointFixture:
    def __init__(self) -> None:
        self.current: dict[tuple[str, str], ProjectionCheckpoint] = {}

    def read_checkpoint(self, scope: ProjectionScope) -> ProjectionCheckpoint | None:
        return self.current.get((scope.aggregate_type, scope.aggregate_id))

    def save_checkpoint(
        self,
        scope: ProjectionScope,
        expected: ProjectionCheckpoint | None,
        checkpoint: ProjectionCheckpoint,
    ) -> None:
        key = (scope.aggregate_type, scope.aggregate_id)
        if self.current.get(key) != expected:
            raise CheckpointConflict("checkpoint_cas")
        if expected is not None and checkpoint.fence_token < expected.fence_token:
            raise CheckpointConflict("checkpoint_fence")
        self.current[key] = checkpoint

    def reset_checkpoint(self, scope: ProjectionScope, fence_token: int) -> None:
        self.current.pop((scope.aggregate_type, scope.aggregate_id), None)


class GraphFixture:
    def __init__(self) -> None:
        self.applied: list[str] = []
        self.cleaned: list[str] = []
        self.fail = False

    def apply_event(
        self,
        event: OutboxEnvelope,
        fence_token: int,
    ) -> GraphProjectionReceipt:
        del fence_token
        if self.fail:
            raise RuntimeError("fixture graph is unavailable")
        if event.event_id in self.applied:
            return GraphProjectionReceipt(event_id=event.event_id, applied=False, replayed=True)
        self.applied.append(event.event_id)
        return GraphProjectionReceipt(event_id=event.event_id, applied=True)

    def reset_projection(self, scope: ProjectionScope, fence_token: int) -> None:
        del scope, fence_token
        self.applied.clear()
        self.cleaned.clear()

    def cleanup_tombstone(self, event: OutboxEnvelope, fence_token: int) -> None:
        del fence_token
        self.cleaned.append(event.event_id)


class DriftFixture:
    def __init__(self) -> None:
        self.records: list[DriftRecord] = []

    def record(self, drift: DriftRecord) -> None:
        self.records.append(drift)


def _service(
    events: tuple[OutboxEnvelope, ...],
) -> tuple[ProjectionService, CheckpointFixture, GraphFixture, DriftFixture]:
    outbox = OutboxFixture(events)
    checkpoints = CheckpointFixture()
    graph = GraphFixture()
    drift = DriftFixture()
    return (
        ProjectionService(
            projector_id="projector:test",
            outbox=outbox,
            checkpoints=checkpoints,
            graph=graph,
            drift=drift,
            tombstones=outbox,
        ),
        checkpoints,
        graph,
        drift,
    )


def test_event_identity_is_replay_stable_and_summary_is_private() -> None:
    mutation = _mutation(1)
    first = _event(mutation)
    second = _event(mutation)

    assert first.event_id == second.event_id
    assert first.event_digest == second.event_digest
    assert tuple(first.summary) == tuple(sorted(first.summary))
    with pytest.raises(ValueError, match="summary_key_not_allowlisted"):
        _mutation(1, summary={"arguments": "not allowed"})
    with pytest.raises(ValueError, match="summary_text_contains_sensitive_marker"):
        _mutation(1, summary={"status": "token=leaked"})


def test_authority_and_outbox_are_one_call_and_rollback_has_no_event() -> None:
    authority = AuthorityFixture()
    mutation = _mutation(1)
    event = _event(mutation)
    receipt = commit_authoritative_change(authority, mutation, event)
    assert receipt.event_id == event.event_id
    assert len(authority.calls) == 1

    rollback = _mutation(2, operation="rollback")
    rollback_receipt = commit_authoritative_change(authority, rollback, None)
    assert rollback_receipt.event_id is None
    assert authority.calls[-1][1] is None

    with pytest.raises(ValueError, match="forward_mutation_requires_outbox_event"):
        commit_authoritative_change(authority, _mutation(3), None)


def test_projection_orders_events_and_rebuilds_from_authority() -> None:
    first = _mutation(1)
    second = _mutation(2)
    service, checkpoints, graph, drift = _service((_event(first), _event(second)))
    scope = ProjectionScope(aggregate_type="service", aggregate_id="service:one")

    result = service.project_scope(scope, fence_token=1, limit=2)
    assert result.status == "applied"
    assert result.checkpoint is not None
    assert result.checkpoint.last_sequence == 2
    assert len(graph.applied) == 2
    assert not drift.records

    rebuilt = service.rebuild_scope(scope, fence_token=2, limit=2)
    assert rebuilt.status == "applied"
    assert rebuilt.checkpoint is not None
    assert rebuilt.checkpoint.last_sequence == 2
    assert len(graph.applied) == 2
    assert checkpoints.read_checkpoint(scope) == rebuilt.checkpoint


def test_gap_and_projection_failure_are_visible_without_cursor_advance() -> None:
    first = _mutation(1)
    second = _mutation(2)
    service, checkpoints, graph, drift = _service((_event(second),))
    scope = ProjectionScope(aggregate_type="service", aggregate_id="service:one")
    gap = service.project_scope(scope, fence_token=1)
    assert gap.status == "drift"
    assert checkpoints.read_checkpoint(scope) is None
    assert drift.records[-1].reason_code == "sequence_gap"

    service, checkpoints, graph, drift = _service((_event(first),))
    graph.fail = True
    failed = service.project_scope(scope, fence_token=1)
    assert failed.status == "failed"
    assert checkpoints.read_checkpoint(scope) is None
    assert drift.records[-1].reason_code == "graph_apply_failed"


def test_tombstone_cleanup_requires_projected_cursor() -> None:
    first = _mutation(1)
    deleted = _mutation(2, operation="delete")
    service, _, graph, _ = _service((_event(first), _event(deleted, tombstone=True)))
    scope = ProjectionScope(aggregate_type="service", aggregate_id="service:one")
    service.project_scope(scope, fence_token=1, limit=2)
    cleaned = service.cleanup_tombstones(
        scope,
        fence_token=1,
        before_sequence=3,
        limit=1,
    )
    assert [item.status for item in cleaned.outcomes] == ["cleaned"]
    assert graph.cleaned == [cleaned.outcomes[0].event_id]


def test_observation_promotion_is_explicit_and_reverse_sync_is_rejected() -> None:
    authority = AuthorityFixture()
    observation = ObservationPromotion(
        observation_id="observation:one",
        observation_type="health",
        aggregate_type="service",
        aggregate_id="service:one",
        observation_digest=sha256_digest({"health": "ready"}),
        source_ref="probe:one",
        evidence_ref="evidence:one",
        observed_at=NOW,
        summary={"status": "ready"},
    )
    policy = ObservationPromotionPolicy(
        policy_ref="policy:health",
        enabled=True,
        allowed_observation_types=("health",),
        max_age_seconds=300,
    )
    receipt = promote_graph_observation(
        authority,
        observation,
        policy,
        sequence=1,
        authoritative_revision=1,
        now=NOW,
    )
    assert receipt.event_id is not None
    assert authority.calls[0][0].origin == "graph_observation_promotion"
    with pytest.raises(ValueError, match="reverse_sync_is_not_an_authority_path"):
        reject_reverse_sync(source="graphos")
