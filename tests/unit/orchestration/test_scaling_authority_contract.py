"""NE-164 scaling authority contract fixtures.

CONCEPT:AU-OS.scaling.reactive-replica-autoscaling

These tests exercise the bounded authority boundary only.  Runtime
controllers, metric collection, graph persistence, and Kubernetes adapters
remain outside this slice.
"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from agent_utilities.orchestration.scaling_authority import (
    ControllerMode,
    ControllerRegistration,
    DelegatedController,
    FailureDomain,
    FailureDomainStatus,
    LeaseFence,
    MaintenanceWindow,
    ObservedOutcome,
    OfflineRecoveryPolicy,
    QuotaPolicy,
    ResourcePool,
    RollbackPolicy,
    ScaleAuthority,
    ScaleCadence,
    ScaleDecision,
    ScaleExecution,
    ScaleIntent,
    ScaleUnit,
    SignalSummaryRef,
    WorkloadClass,
    validate_scale_intent,
    validate_scale_lifecycle,
)

pytestmark = pytest.mark.concept("AU-OS.scaling.reactive-replica-autoscaling")

NOW = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)
LATER = NOW + timedelta(minutes=5)


def _fence(*, epoch: int = 4) -> LeaseFence:
    return LeaseFence(
        lease_id="lease:scaling-authority",
        lease_epoch=epoch,
        fence_token=9,
        expires_at=NOW + timedelta(minutes=30),
    )


def _failure_domain(
    status: FailureDomainStatus = FailureDomainStatus.HEALTHY,
) -> FailureDomain:
    return FailureDomain(
        domain_id="host:small-gpu-a",
        kind="host",
        status=status,
        last_observed_at=NOW,
        authority_ref="authority:inventory",
    )


def _pool() -> ResourcePool:
    return ResourcePool(
        pool_id="pool:replica-slots",
        revision=2,
        resource_kind="replica_slots",
        capacity_units=16,
        allocated_units=2,
        reserved_headroom_units=2,
        failure_domain_id="host:small-gpu-a",
        quota=QuotaPolicy(scope="tenant:acme", max_units=12),
        lease_fence=_fence(),
        offline_recovery=OfflineRecoveryPolicy(max_staleness_s=120),
    )


def _workload() -> WorkloadClass:
    return WorkloadClass(
        workload_id="workload:gateway",
        revision=5,
        tenant_ref="tenant:acme",
        resource_pool_id="pool:replica-slots",
        cadence=ScaleCadence.SERVICE_REPLICA,
        priority=50,
        max_concurrency=8,
        quota=QuotaPolicy(scope="tenant:acme", max_units=12),
    )


def _unit() -> ScaleUnit:
    return ScaleUnit(
        unit_id="unit:gateway",
        revision=7,
        workload_class_id="workload:gateway",
        resource_pool_id="pool:replica-slots",
        cadence=ScaleCadence.SERVICE_REPLICA,
        controller_mode=ControllerMode.NATIVE,
        controller_id="controller:au",
        replica_writer_id="controller:au",
        min_replicas=1,
        max_replicas=8,
        reserved_headroom_replicas=2,
        tenant_quota=QuotaPolicy(scope="tenant:acme", max_units=8),
        failure_domain_id="host:small-gpu-a",
        offline_recovery=OfflineRecoveryPolicy(max_staleness_s=120),
        maintenance_window=MaintenanceWindow(window="02:00-04:00"),
        rollback_policy=RollbackPolicy(on_failure="automatic", max_attempts=2),
    )


def _native_writer() -> ControllerRegistration:
    return ControllerRegistration(
        unit_id="unit:gateway",
        controller_id="controller:au",
        mode=ControllerMode.NATIVE,
        writes_replicas=True,
        lease_fence=_fence(),
    )


def _authority(
    *,
    failure_domain: FailureDomain | None = None,
    unit: ScaleUnit | None = None,
    controllers: tuple[ControllerRegistration, ...] | None = None,
) -> ScaleAuthority:
    return ScaleAuthority(
        authority_revision=11,
        failure_domains=(failure_domain or _failure_domain(),),
        pools=(_pool(),),
        workloads=(_workload(),),
        units=(unit or _unit(),),
        controllers=(_native_writer(),) if controllers is None else controllers,
    )


def _summary() -> SignalSummaryRef:
    return SignalSummaryRef(
        summary_id="summary:gateway:queue-depth",
        summary_digest="a" * 64,
        source="telemetry",
        unit_id="unit:gateway",
        signal="queue_depth",
        aggregation="p95",
        window_start=NOW,
        window_end=LATER,
        sample_count=300,
    )


def _intent() -> ScaleIntent:
    return ScaleIntent(
        intent_id="intent:gateway:12",
        unit_id="unit:gateway",
        revision=12,
        expected_unit_revision=7,
        cadence="service_replica",
        desired_replicas=4,
        signal_summary_ref=_summary(),
        controller_mode=ControllerMode.NATIVE,
        proposer_id="controller:au",
        replica_writer_id="controller:au",
        lease_fence=_fence(),
        idempotency_key="idempotency:gateway:12",
        reason="p95 queue depth crossed the target",
        created_at=NOW,
    )


def test_native_authority_and_revisioned_lifecycle_are_bounded() -> None:
    authority = _authority()
    intent = _intent()
    validate_scale_intent(intent, authority)

    decision = ScaleDecision(
        decision_id="decision:gateway:12",
        intent_id=intent.intent_id,
        intent_revision=intent.revision,
        unit_id=intent.unit_id,
        verdict="accepted",
        controller_mode=ControllerMode.NATIVE,
        decisioner_id="policy:scaling",
        policy_revision=3,
        replica_writer_id="controller:au",
        lease_fence=_fence(),
        reason="within bounds and headroom",
        decided_at=NOW,
    )
    execution = ScaleExecution(
        execution_id="execution:gateway:12",
        intent_id=intent.intent_id,
        decision_id=decision.decision_id,
        intent_revision=intent.revision,
        unit_id=intent.unit_id,
        execution_revision=1,
        requested_replicas=4,
        controller_mode=ControllerMode.NATIVE,
        replica_writer_id="controller:au",
        lease_fence=_fence(),
        idempotency_key=intent.idempotency_key,
        operation="scale",
        state="succeeded",
        started_at=NOW,
        completed_at=LATER,
    )
    outcome = ObservedOutcome(
        outcome_id="outcome:gateway:12",
        execution_id=execution.execution_id,
        unit_id=intent.unit_id,
        execution_revision=execution.execution_revision,
        observed_revision=1,
        observed_replicas=4,
        status="converged",
        observed_by="observer:gateway",
        failure_domain_id="host:small-gpu-a",
        lease_fence=_fence(),
        observed_at=LATER,
    )
    validate_scale_lifecycle(intent, decision, execution, outcome)


def test_signal_reference_is_bounded_and_never_carries_samples() -> None:
    summary = _summary()
    projection = summary.graph_projection()
    assert projection["sample_count"] == 300
    assert "samples" not in projection

    with pytest.raises(ValidationError, match="extra"):
        SignalSummaryRef.model_validate(
            {**summary.model_dump(), "samples": [0.1, 0.2, 0.3]}
        )


def test_engine_local_cadence_cannot_create_graph_intent() -> None:
    payload = _intent().model_dump()
    payload["cadence"] = ScaleCadence.ENGINE_LOCAL.value
    with pytest.raises(ValidationError):
        ScaleIntent.model_validate(payload)


def test_dual_replica_writers_are_rejected() -> None:
    second_writer = ControllerRegistration(
        unit_id="unit:gateway",
        controller_id="controller:second",
        mode=ControllerMode.NATIVE,
        writes_replicas=True,
        lease_fence=_fence(),
    )
    with pytest.raises(ValidationError, match="exactly one replica writer"):
        _authority(controllers=(_native_writer(), second_writer))
    with pytest.raises(ValidationError, match="exactly one replica writer"):
        _authority(controllers=())


def test_delegated_hpa_has_one_writer_and_an_observer() -> None:
    delegated_unit = _unit().model_copy(
        update={
            "controller_mode": ControllerMode.DELEGATED,
            "delegated_controller": DelegatedController.HPA,
            "controller_id": "controller:observer",
            "replica_writer_id": "controller:hpa",
        }
    )
    hpa_writer = ControllerRegistration(
        unit_id="unit:gateway",
        controller_id="controller:hpa",
        mode=ControllerMode.DELEGATED,
        delegated_controller=DelegatedController.HPA,
        writes_replicas=True,
        lease_fence=_fence(),
    )
    au_observer = ControllerRegistration(
        unit_id="unit:gateway",
        controller_id="controller:observer",
        mode=ControllerMode.DELEGATED,
        delegated_controller=DelegatedController.HPA,
        writes_replicas=False,
        lease_fence=_fence(),
    )
    authority = _authority(
        unit=delegated_unit,
        controllers=(hpa_writer, au_observer),
    )
    assert authority.units[0].controller_mode == ControllerMode.DELEGATED.value


def test_self_and_indirect_dependency_cycles_are_rejected() -> None:
    self_dependent = _unit().model_copy(update={"depends_on": ("unit:gateway",)})
    with pytest.raises(ValidationError, match="self-dependency"):
        _authority(unit=self_dependent)

    second_unit = _unit().model_copy(
        update={
            "unit_id": "unit:worker",
            "controller_id": "controller:worker",
            "replica_writer_id": "controller:worker",
            "depends_on": ("unit:gateway",),
        }
    )
    first_unit = _unit().model_copy(update={"depends_on": ("unit:worker",)})
    worker_writer = ControllerRegistration(
        unit_id="unit:worker",
        controller_id="controller:worker",
        mode=ControllerMode.NATIVE,
        writes_replicas=True,
        lease_fence=_fence(),
    )
    with pytest.raises(ValidationError, match="dependency cycle"):
        ScaleAuthority(
            authority_revision=11,
            failure_domains=(_failure_domain(),),
            pools=(_pool(),),
            workloads=(_workload(),),
            units=(first_unit, second_unit),
            controllers=(_native_writer(), worker_writer),
        )


def test_scale_intent_enforces_bounds_quota_headroom_and_revision() -> None:
    authority = _authority()
    with pytest.raises(ValueError, match="min/max"):
        validate_scale_intent(
            _intent().model_copy(update={"desired_replicas": 9}), authority
        )
    with pytest.raises(ValueError, match="stale"):
        validate_scale_intent(
            _intent().model_copy(update={"expected_unit_revision": 6}), authority
        )
    with pytest.raises(ValidationError, match="headroom"):
        _authority(unit=_unit().model_copy(update={"reserved_headroom_replicas": 7}))


def test_offline_hold_and_unverified_execution_states_fail_closed() -> None:
    offline = _authority(failure_domain=_failure_domain(FailureDomainStatus.OFFLINE))
    with pytest.raises(ValueError, match="failure domain is offline"):
        validate_scale_intent(_intent(), offline)

    failed_payload = {
        **ScaleExecution(
            execution_id="execution:gateway:failed",
            intent_id="intent:gateway:12",
            decision_id="decision:gateway:12",
            intent_revision=12,
            unit_id="unit:gateway",
            execution_revision=1,
            requested_replicas=4,
            controller_mode=ControllerMode.NATIVE,
            replica_writer_id="controller:au",
            lease_fence=_fence(),
            idempotency_key="idempotency:gateway:12",
            operation="scale",
            state="started",
            started_at=NOW,
        ).model_dump(),
        "state": "failed",
        "completed_at": LATER,
    }
    with pytest.raises(ValidationError, match="failure_code"):
        ScaleExecution.model_validate(failed_payload)

    simulated_payload = {
        **failed_payload,
        "state": "simulated",
        "completed_at": LATER,
    }
    with pytest.raises(ValidationError, match="simulated"):
        ScaleExecution.model_validate(simulated_payload)

    with pytest.raises(ValidationError, match="UTC HH:MM-HH:MM"):
        MaintenanceWindow(window="25:00-26:00")


def test_lease_fence_cannot_change_between_lifecycle_records() -> None:
    intent = _intent()
    decision = ScaleDecision(
        decision_id="decision:gateway:12",
        intent_id=intent.intent_id,
        intent_revision=intent.revision,
        unit_id=intent.unit_id,
        verdict="accepted",
        controller_mode=ControllerMode.NATIVE,
        decisioner_id="policy:scaling",
        policy_revision=3,
        replica_writer_id="controller:au",
        lease_fence=_fence(epoch=5),
        reason="fence changed",
        decided_at=NOW,
    )
    execution = ScaleExecution(
        execution_id="execution:gateway:12",
        intent_id=intent.intent_id,
        decision_id=decision.decision_id,
        intent_revision=intent.revision,
        unit_id=intent.unit_id,
        execution_revision=1,
        requested_replicas=4,
        controller_mode=ControllerMode.NATIVE,
        replica_writer_id="controller:au",
        lease_fence=_fence(epoch=5),
        idempotency_key=intent.idempotency_key,
        operation="scale",
        state="succeeded",
        started_at=NOW,
        completed_at=LATER,
    )
    outcome = ObservedOutcome(
        outcome_id="outcome:gateway:12",
        execution_id=execution.execution_id,
        unit_id=intent.unit_id,
        execution_revision=execution.execution_revision,
        observed_revision=1,
        observed_replicas=4,
        status="converged",
        observed_by="observer:gateway",
        failure_domain_id="host:small-gpu-a",
        lease_fence=_fence(epoch=5),
        observed_at=LATER,
    )
    with pytest.raises(ValueError, match="lease/fence"):
        validate_scale_lifecycle(intent, decision, execution, outcome)
