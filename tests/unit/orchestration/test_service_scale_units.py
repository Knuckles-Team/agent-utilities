"""Focused fixtures for the shared AU-OS service-surface scale contract.

CONCEPT:AU-OS.scaling.service-scale-units

These fixtures deliberately exercise the common evaluator at the policy
boundaries.  They do not start a controller, call Kubernetes, or emulate an
engine/provider: an actuator owns those concerns after a decision is accepted.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agent_utilities.orchestration.service_scale_units import (
    CapacityAxis,
    CapacityDemand,
    CapacityObservation,
    ContinuityContract,
    ContinuityObservation,
    LoadSafetyObservation,
    PartitionLeaseContract,
    QuotaBinding,
    QuotaObservation,
    QuotaScope,
    ScalePolicy,
    ScaleUnitContract,
    SignalKind,
    SignalObservation,
    content_digest,
    evaluate_scale,
)


NOW = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)
FUTURE = NOW + timedelta(minutes=5)


def _digest(ref: str) -> str:
    return content_digest({"fixture": ref})


def _signal(
    value: int,
    *,
    signal: SignalKind = "request_rate",
    observed_at: datetime = NOW - timedelta(minutes=1),
    expires_at: datetime = FUTURE,
) -> SignalObservation:
    return SignalObservation(
        signal=signal,
        value=value,
        observed_at=observed_at,
        expires_at=expires_at,
        source_ref=f"metrics:{signal}",
        source_digest=_digest(f"metrics:{signal}"),
    )


def _capacity(
    axis: CapacityAxis,
    *,
    allocatable: int = 10_000,
    reserved: int = 1_000,
    used: int = 0,
    observed_at: datetime = NOW - timedelta(minutes=1),
    expires_at: datetime = FUTURE,
) -> CapacityObservation:
    return CapacityObservation(
        axis=axis,
        allocatable=allocatable,
        reserved=reserved,
        used=used,
        observed_at=observed_at,
        expires_at=expires_at,
        source_ref=f"capacity:{axis}",
        source_digest=_digest(f"capacity:{axis}"),
    )


def _quota(
    *,
    limit: int = 100,
    used: int = 0,
    scope: QuotaScope = "provider_global",
    observed_at: datetime = NOW - timedelta(minutes=1),
    expires_at: datetime = FUTURE,
) -> QuotaObservation:
    return QuotaObservation(
        quota_ref="provider:llm",
        scope=scope,
        limit=limit,
        used=used,
        observed_at=observed_at,
        expires_at=expires_at,
        source_ref="quota:provider-llm",
        source_digest=_digest("quota:provider-llm"),
    )


def _continuity(
    *,
    active_sessions: int = 0,
    active_streams: int = 0,
    ready: bool = True,
    observed_at: datetime = NOW - timedelta(minutes=1),
    expires_at: datetime = FUTURE,
) -> ContinuityObservation:
    return ContinuityObservation(
        ready=ready,
        active_sessions=active_sessions,
        active_streams=active_streams,
        observed_at=observed_at,
        expires_at=expires_at,
        source_ref="continuity:sessions",
        source_digest=_digest("continuity:sessions"),
    )


def _safety(*, overloaded: bool = False, noisy_neighbor: bool = False) -> LoadSafetyObservation:
    return LoadSafetyObservation(
        overloaded=overloaded,
        noisy_neighbor=noisy_neighbor,
        reason_ref="incident:capacity" if overloaded or noisy_neighbor else None,
        observed_at=NOW - timedelta(minutes=1),
        expires_at=FUTURE,
        source_ref="safety:node",
        source_digest=_digest("safety:node"),
    )


def _demands(axes: tuple[CapacityAxis, ...]) -> tuple[CapacityDemand, ...]:
    return tuple(CapacityDemand(axis=axis, per_replica=10) for axis in axes)


def _mcp_contract(
    *,
    min_replicas: int = 1,
    max_replicas: int = 8,
    drain_seconds: int = 30,
    allow_scale_to_zero: bool = False,
    allow_scale_from_zero: bool = True,
    target_signal: SignalKind = "request_rate",
    quotas: tuple[QuotaBinding, ...] | None = None,
) -> ScaleUnitContract:
    axes: tuple[CapacityAxis, ...] = (
        "cpu_milli",
        "memory_mib",
        "session_slots",
        "continuity_slots",
    )
    return ScaleUnitContract(
        unit_ref="unit:mcp-gateway",
        surface="mcp_gateway",
        policy=ScalePolicy(
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            target=100,
            target_signal=target_signal,
            scale_up_step=2,
            scale_down_step=1,
            scale_up_cooldown_s=0,
            scale_down_cooldown_s=0,
            drain_seconds=drain_seconds,
            allow_scale_to_zero=allow_scale_to_zero,
            allow_scale_from_zero=allow_scale_from_zero,
        ),
        allowed_signals=(
            "request_rate",
            "in_flight",
            "p95_latency_ms",
            "active_sessions",
            "continuity_load",
            "provider_in_flight",
            "error_rate_ppm",
        ),
        required_capacity_axes=axes,
        replica_demands=_demands(axes),
        quotas=quotas
        if quotas is not None
        else (
            QuotaBinding(
                quota_ref="provider:llm",
                axis="provider_global_quota",
                scope="provider_global",
                per_replica=10,
            ),
        ),
        continuity=ContinuityContract(
            mode="externalized",
            external_store_ref="store:sessions",
            drain_required=True,
            block_scale_down_with_active=True,
            max_drain_seconds=300,
        ),
    )


def _mcp_capacities() -> tuple[CapacityObservation, ...]:
    axes: tuple[CapacityAxis, ...] = (
        "cpu_milli",
        "memory_mib",
        "session_slots",
        "continuity_slots",
    )
    return tuple(_capacity(axis) for axis in axes)


def _api_capacities() -> tuple[CapacityObservation, ...]:
    axes: tuple[CapacityAxis, ...] = ("cpu_milli", "memory_mib", "session_slots")
    return tuple(_capacity(axis) for axis in axes)


def _api_contract(
    *,
    min_replicas: int = 1,
    allow_scale_to_zero: bool = False,
    allow_scale_from_zero: bool = True,
) -> ScaleUnitContract:
    axes: tuple[CapacityAxis, ...] = ("cpu_milli", "memory_mib", "session_slots")
    return ScaleUnitContract(
        unit_ref="unit:api-gateway",
        surface="api_gateway",
        policy=ScalePolicy(
            min_replicas=min_replicas,
            max_replicas=4,
            target=100,
            target_signal="request_rate",
            scale_up_cooldown_s=0,
            scale_down_cooldown_s=0,
            drain_seconds=0,
            allow_scale_to_zero=allow_scale_to_zero,
            allow_scale_from_zero=allow_scale_from_zero,
        ),
        allowed_signals=("request_rate", "in_flight", "p95_latency_ms", "error_rate_ppm"),
        required_capacity_axes=axes,
        replica_demands=_demands(axes),
        continuity=ContinuityContract(
            mode="stateless",
            drain_required=False,
            block_scale_down_with_active=False,
        ),
    )


def test_one_evaluator_target_tracks_mcp_and_decision_is_deterministic() -> None:
    contract = _mcp_contract()
    first = evaluate_scale(
        contract,
        current_replicas=1,
        signal=_signal(250),
        capacities=_mcp_capacities(),
        quotas=(_quota(),),
        continuity=_continuity(),
        safety=_safety(),
        now=NOW,
    )
    second = evaluate_scale(
        contract,
        current_replicas=1,
        signal=_signal(250),
        capacities=_mcp_capacities(),
        quotas=(_quota(),),
        continuity=_continuity(),
        safety=_safety(),
        now=NOW,
    )

    assert first.action == "scale_up"
    assert first.desired_replicas == 3
    assert first.reasons == ("target_tracking",)
    assert first == second
    assert first.decision_id == second.decision_id
    assert first.decision_digest == second.decision_digest


def test_missing_signal_blocks_without_optimistic_scale() -> None:
    decision = evaluate_scale(
        _mcp_contract(),
        current_replicas=1,
        signal=None,
        capacities=_mcp_capacities(),
        quotas=(_quota(),),
        continuity=_continuity(),
        safety=_safety(),
        now=NOW,
    )

    assert decision.action == "blocked"
    assert "signal_missing" in decision.reasons
    assert any(item.status == "missing" and item.kind == "signal" for item in decision.evidence)


def test_stale_provider_budget_blocks_mcp() -> None:
    stale_quota = _quota(
        observed_at=NOW - timedelta(minutes=10),
        expires_at=NOW - timedelta(minutes=5),
    )
    decision = evaluate_scale(
        _mcp_contract(),
        current_replicas=1,
        signal=_signal(250),
        capacities=_mcp_capacities(),
        quotas=(stale_quota,),
        continuity=_continuity(),
        safety=_safety(),
        now=NOW,
    )

    assert decision.action == "blocked"
    assert decision.reasons == ("provider_quota_stale",)


def test_mcp_replicas_cannot_multiply_shared_provider_budget() -> None:
    decision = evaluate_scale(
        _mcp_contract(),
        current_replicas=1,
        signal=_signal(250),
        capacities=_mcp_capacities(),
        quotas=(_quota(limit=10),),
        continuity=_continuity(),
        safety=_safety(),
        now=NOW,
    )

    assert decision.action == "blocked"
    assert "provider_quota_exhausted" in decision.reasons

    with pytest.raises(ValueError, match="cannot be multiplied"):
        QuotaBinding(
            quota_ref="provider:llm",
            axis="provider_global_quota",
            scope="provider_global",
            per_replica=10,
            multiplicative=True,
        )


def test_query_reader_refuses_to_scale_when_engine_authority_is_saturated() -> None:
    axes: tuple[CapacityAxis, ...] = (
        "cpu_milli",
        "memory_mib",
        "read_admission_slots",
        "engine_bytes",
        "fsync_iops",
    )
    contract = ScaleUnitContract(
        unit_ref="unit:query-reader",
        surface="query_reader",
        policy=ScalePolicy(
            min_replicas=1,
            max_replicas=8,
            target=100,
            target_signal="request_rate",
            scale_up_cooldown_s=0,
            scale_down_cooldown_s=0,
        ),
        allowed_signals=("request_rate", "in_flight", "p95_latency_ms", "error_rate_ppm"),
        required_capacity_axes=axes,
        replica_demands=_demands(axes),
        engine_authority_axes=("read_admission_slots", "engine_bytes", "fsync_iops"),
        continuity=ContinuityContract(
            mode="stateless",
            drain_required=False,
            block_scale_down_with_active=False,
        ),
    )
    capacities = (
        _capacity("cpu_milli"),
        _capacity("memory_mib"),
        _capacity("read_admission_slots", allocatable=100, reserved=100, used=100),
        _capacity("engine_bytes", allocatable=100, reserved=100, used=100),
        _capacity("fsync_iops", allocatable=100, reserved=100, used=100),
    )

    decision = evaluate_scale(
        contract,
        current_replicas=1,
        signal=_signal(250),
        capacities=capacities,
        quotas=(),
        continuity=None,
        safety=_safety(),
        now=NOW,
    )

    assert decision.action == "blocked"
    assert "engine_authority_saturated" in decision.reasons


def test_continuity_and_active_sessions_block_drain() -> None:
    decision = evaluate_scale(
        _mcp_contract(),
        current_replicas=3,
        signal=_signal(0),
        capacities=_mcp_capacities(),
        quotas=(_quota(),),
        continuity=_continuity(active_sessions=1),
        safety=_safety(),
        now=NOW,
    )

    assert decision.action == "blocked"
    assert "active_sessions_block_drain" in decision.reasons
    assert decision.drain_required is True

    missing = evaluate_scale(
        _mcp_contract(),
        current_replicas=3,
        signal=_signal(0),
        capacities=_mcp_capacities(),
        quotas=(_quota(),),
        continuity=None,
        safety=_safety(),
        now=NOW,
    )
    assert missing.action == "blocked"
    assert "continuity_missing" in missing.reasons


def test_overload_and_noisy_neighbor_evidence_block_actuation() -> None:
    decision = evaluate_scale(
        _mcp_contract(),
        current_replicas=1,
        signal=_signal(250),
        capacities=_mcp_capacities(),
        quotas=(_quota(),),
        continuity=_continuity(),
        safety=_safety(overloaded=True, noisy_neighbor=True),
        now=NOW,
    )

    assert decision.action == "blocked"
    assert "overload_shedding_active" in decision.reasons
    assert "noisy_neighbor" in decision.reasons


def test_scale_to_min_and_scale_from_zero_are_explicitly_bounded() -> None:
    contract = _api_contract(min_replicas=1)
    scale_to_min = evaluate_scale(
        contract,
        current_replicas=3,
        signal=_signal(0),
        capacities=_api_capacities(),
        quotas=(),
        continuity=None,
        safety=_safety(),
        now=NOW,
    )
    assert scale_to_min.action == "scale_down"
    assert scale_to_min.desired_replicas == 2

    cannot_start = evaluate_scale(
        _api_contract(min_replicas=0, allow_scale_from_zero=False),
        current_replicas=0,
        signal=_signal(100),
        capacities=_api_capacities(),
        quotas=(),
        continuity=None,
        safety=_safety(),
        now=NOW,
    )
    assert cannot_start.action == "blocked"
    assert "scale_from_zero_disabled" in cannot_start.reasons

    can_stop = evaluate_scale(
        _api_contract(min_replicas=0, allow_scale_to_zero=True),
        current_replicas=1,
        signal=_signal(0),
        capacities=_api_capacities(),
        quotas=(),
        continuity=None,
        safety=_safety(),
        now=NOW,
    )
    assert can_stop.action == "scale_down"
    assert can_stop.desired_replicas == 0


def test_surface_contract_requires_allowed_signal_and_provider_budget() -> None:
    with pytest.raises(ValueError, match="target_signal"):
        _mcp_contract(target_signal="queue_depth")

    with pytest.raises(ValueError, match="provider-global quota"):
        _mcp_contract(quotas=())


def test_partition_lease_and_drain_bounds_fail_closed() -> None:
    with pytest.raises(ValueError, match="partitions_per_replica"):
        PartitionLeaseContract(partitioned=True)
    with pytest.raises(ValueError, match="TTL and fencing"):
        PartitionLeaseContract(partitioned=True, partitions_per_replica=1, lease_required=True)

    decision = evaluate_scale(
        _mcp_contract(drain_seconds=301),
        current_replicas=3,
        signal=_signal(0),
        capacities=_mcp_capacities(),
        quotas=(_quota(),),
        continuity=_continuity(),
        safety=_safety(),
        now=NOW,
    )
    assert decision.action == "blocked"
    assert "drain_required" in decision.reasons
