"""Focused NE-166 proofs for durable, idempotent scale-intent actuation.

These fixtures deliberately stop at the typed AU seam.  Runtime adapters (for
example Kubernetes or Swarm) are responsible for implementing the same
execution-key dedupe contract; they do not belong in this unit fixture.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agent_utilities.orchestration.scaling_intent_reconciler import (
    AllowScalePolicy,
    BreakGlassAuthorization,
    MemoryScaleIntentLedger,
    ScaleActuationRequest,
    ScaleActuationResult,
    ScaleControllerMode,
    ScaleFence,
    ScaleIntentConflict,
    ScaleIntentReconciler,
    ScaleIntentRecord,
    ScaleIntentState,
    ScaleLeaseUnavailable,
    ScaleObservationStatus,
    ScalePolicyDecision,
    ScaleRetryPolicy,
    ScaleTargetBinding,
)

pytestmark = pytest.mark.concept("AU-OS.scaling.reactive-replica-autoscaling")


NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


def _target(
    *, uid: str = "target-uid-1", resource_version: str = "rv-1"
) -> ScaleTargetBinding:
    return ScaleTargetBinding(
        cluster_id="cluster-1",
        runtime_id="runtime-1",
        workload_id="workload-1",
        target_uid=uid,
        target_resource_version=resource_version,
    )


def _intent(
    *,
    execution_key: str = "scale-execution-1",
    target: ScaleTargetBinding | None = None,
    max_attempts: int = 3,
) -> ScaleIntentRecord:
    return ScaleIntentRecord(
        intent_id="scale-intent-1",
        intent_revision=1,
        expected_unit_revision=7,
        desired_replicas=3,
        min_replicas=1,
        max_replicas=5,
        controller_mode=ScaleControllerMode.NATIVE,
        replica_writer_id="writer-1",
        execution_key=execution_key,
        fence=ScaleFence(
            lease_id="intent-fence-1",
            lease_epoch=4,
            fence_token=9,
            expires_at=NOW + timedelta(minutes=10),
        ),
        target=target or _target(),
        reason="bounded reconciliation fixture",
        retry_policy=ScaleRetryPolicy(max_attempts=max_attempts, backoff_s=2),
        created_at=NOW,
    )


class RecordingActuator:
    """Typed fixture actuator that records calls and returns queued outcomes."""

    name = "fixture-actuator"

    def __init__(self, events: list[str], outcomes: list[str] | None = None) -> None:
        self.events = events
        self.outcomes = list(outcomes or [])
        self.calls: list[ScaleActuationRequest] = []

    def apply(self, request: ScaleActuationRequest) -> ScaleActuationResult:
        self.events.append("actuator:call")
        self.calls.append(request)
        outcome = self.outcomes.pop(0) if self.outcomes else "success"
        if outcome == "retry":
            return ScaleActuationResult(
                execution_key=request.execution_key,
                target=request.target,
                state=ScaleIntentState.FAILED,
                scaled=False,
                retryable=True,
                error_code="temporarily_unavailable",
                detail="fixture transient failure",
            )
        if outcome == "mismatch":
            return ScaleActuationResult(
                execution_key=request.execution_key,
                target=request.target,
                state=ScaleIntentState.SUCCEEDED,
                scaled=True,
                observed_replicas=request.desired_replicas + 1,
            )
        return ScaleActuationResult(
            execution_key=request.execution_key,
            target=request.target,
            state=ScaleIntentState.SUCCEEDED,
            scaled=True,
            observed_replicas=request.desired_replicas,
        )


class DenyScalePolicy:
    def decide(self, intent: ScaleIntentRecord) -> ScalePolicyDecision:
        return ScalePolicyDecision(
            decision_id=f"deny:{intent.execution_key}",
            allowed=False,
            reason="fixture policy requires an explicit approval",
        )


def _reconciler(
    ledger: MemoryScaleIntentLedger,
    actuator: RecordingActuator,
    *,
    policy: object | None = None,
) -> ScaleIntentReconciler:
    return ScaleIntentReconciler(
        ledger,
        actuator,
        policy=policy if policy is not None else AllowScalePolicy(),
        clock=lambda: NOW,
    )


def test_persists_intent_before_typed_actuation_and_observes_success() -> None:
    ledger = MemoryScaleIntentLedger()
    actuator = RecordingActuator(ledger.events)

    result = _reconciler(ledger, actuator).reconcile(
        _intent(), controller_id="writer-1"
    )

    assert result.state == ScaleIntentState.VERIFIED.value
    assert result.observation is not None
    assert result.observation.status == ScaleObservationStatus.CONVERGED.value
    assert result.execution.state == ScaleIntentState.VERIFIED.value
    assert ledger.events.index("persist_intent") < ledger.events.index("actuator:call")
    assert len(actuator.calls) == 1


def test_duplicate_delivery_replays_without_a_second_actuator_call() -> None:
    ledger = MemoryScaleIntentLedger()
    actuator = RecordingActuator(ledger.events)
    reconciler = _reconciler(ledger, actuator)
    intent = _intent()

    first = reconciler.reconcile(intent, controller_id="writer-1")
    second = reconciler.reconcile(intent, controller_id="writer-1")

    assert first.observation == second.observation
    assert second.replayed is True
    assert second.actuator_called is False
    assert len(actuator.calls) == 1


def test_crash_after_result_repairs_observation_without_reinvoking_actuator() -> None:
    ledger = MemoryScaleIntentLedger()
    actuator = RecordingActuator(ledger.events)
    intent = _intent()
    persisted = ledger.persist_intent(intent)
    lease = ledger.acquire_controller_lease(intent, "writer-1", now=NOW)
    assert lease is not None
    started = ledger.mark_started(intent, persisted.execution, lease, now=NOW)
    ledger.record_result(
        intent,
        started,
        lease,
        ScaleActuationResult(
            execution_key=intent.execution_key,
            target=intent.target,
            state=ScaleIntentState.SUCCEEDED,
            scaled=True,
            observed_replicas=intent.desired_replicas,
        ),
        now=NOW,
    )

    result = _reconciler(ledger, actuator).reconcile(intent, controller_id="writer-1")

    assert result.replayed is True
    assert result.observation is not None
    assert result.execution.state == ScaleIntentState.VERIFIED.value
    assert actuator.calls == []


def test_crash_before_result_retries_same_execution_key_under_new_call() -> None:
    ledger = MemoryScaleIntentLedger()
    actuator = RecordingActuator(ledger.events)
    intent = _intent()
    persisted = ledger.persist_intent(intent)
    lease = ledger.acquire_controller_lease(intent, "writer-1", now=NOW)
    assert lease is not None
    ledger.mark_started(intent, persisted.execution, lease, now=NOW)

    result = _reconciler(ledger, actuator).reconcile(intent, controller_id="writer-1")

    assert result.state == ScaleIntentState.VERIFIED.value
    assert len(actuator.calls) == 1
    assert actuator.calls[0].execution_key == intent.execution_key


def test_retryable_failure_reuses_key_but_requires_a_newer_fenced_attempt() -> None:
    ledger = MemoryScaleIntentLedger()
    actuator = RecordingActuator(ledger.events, outcomes=["retry", "success"])
    reconciler = _reconciler(ledger, actuator)
    intent = _intent()

    first = reconciler.reconcile(intent, controller_id="writer-1")
    second = reconciler.reconcile(intent, controller_id="writer-1")

    assert first.state == ScaleIntentState.FAILED.value
    assert first.retry_after_s == 2
    assert second.state == ScaleIntentState.VERIFIED.value
    assert len(actuator.calls) == 2
    assert {call.execution_key for call in actuator.calls} == {intent.execution_key}
    assert (
        actuator.calls[0].lease.lease_instance_id
        != actuator.calls[1].lease.lease_instance_id
    )
    assert second.execution.result_attempt == 2


def test_same_key_changed_target_or_revision_is_a_fail_closed_conflict() -> None:
    ledger = MemoryScaleIntentLedger()
    actuator = RecordingActuator(ledger.events)
    reconciler = _reconciler(ledger, actuator)
    intent = _intent()
    reconciler.reconcile(intent, controller_id="writer-1")

    changed_target = intent.model_copy(
        update={"target": _target(uid="target-uid-2", resource_version="rv-2")}
    )
    with pytest.raises(ScaleIntentConflict):
        reconciler.reconcile(changed_target, controller_id="writer-1")

    changed_revision = intent.model_copy(
        update={"intent_revision": 2, "expected_unit_revision": 8}
    )
    with pytest.raises(ScaleIntentConflict):
        reconciler.reconcile(changed_revision, controller_id="writer-1")
    assert len(actuator.calls) == 1


def test_dry_run_is_simulated_and_never_calls_or_claims_scaled() -> None:
    ledger = MemoryScaleIntentLedger()
    actuator = RecordingActuator(ledger.events)
    reconciler = _reconciler(ledger, actuator)
    intent = _intent(execution_key="scale-dry-run-1")

    first = reconciler.reconcile(intent, controller_id="writer-1", dry_run=True)
    second = reconciler.reconcile(intent, controller_id="writer-1", dry_run=True)

    assert first.observation is not None
    assert first.observation.status == ScaleObservationStatus.SIMULATED.value
    assert first.scaled is False
    assert first.execution.dry_run is True
    assert second.replayed is True
    assert second.scaled is False
    assert actuator.calls == []


def test_missing_policy_fails_closed_before_actuation() -> None:
    ledger = MemoryScaleIntentLedger()
    actuator = RecordingActuator(ledger.events)
    reconciler = ScaleIntentReconciler(
        ledger,
        actuator,
        clock=lambda: NOW,
    )

    result = reconciler.reconcile(
        _intent(execution_key="scale-no-policy-1"), controller_id="writer-1"
    )

    assert result.state == ScaleIntentState.DENIED.value
    assert actuator.calls == []


def test_denial_requires_scoped_break_glass_audit_before_actuation() -> None:
    ledger = MemoryScaleIntentLedger()
    actuator = RecordingActuator(ledger.events)
    reconciler = _reconciler(ledger, actuator, policy=DenyScalePolicy())
    intent = _intent(execution_key="scale-break-glass-1")

    denied = reconciler.reconcile(intent, controller_id="writer-1")
    assert denied.state == ScaleIntentState.DENIED.value
    assert actuator.calls == []
    assert "break_glass:audit" not in ledger.events

    authorization = BreakGlassAuthorization(
        authorization_id="break-glass-1",
        actor_id="operator-1",
        approval_ref="approval-1",
        reason="bounded emergency recovery",
        target=intent.target,
        authorized_until=NOW + timedelta(minutes=1),
    )
    approved = reconciler.reconcile(
        intent,
        controller_id="writer-1",
        break_glass=authorization,
    )

    assert approved.break_glass_audit is not None
    assert approved.state == ScaleIntentState.VERIFIED.value
    assert ledger.events.index("break_glass:audit") < ledger.events.index(
        "actuator:call"
    )
    assert len(actuator.calls) == 1


def test_other_controller_cannot_reuse_the_declared_replica_writer() -> None:
    ledger = MemoryScaleIntentLedger()
    actuator = RecordingActuator(ledger.events)
    reconciler = _reconciler(ledger, actuator)

    with pytest.raises(ScaleLeaseUnavailable, match="declared replica writer"):
        reconciler.reconcile(_intent(), controller_id="writer-2")
    assert actuator.calls == []
