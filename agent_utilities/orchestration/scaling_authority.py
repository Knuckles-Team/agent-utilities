#!/usr/bin/python
from __future__ import annotations

"""Typed authority contract for governed hierarchical scaling.

CONCEPT:AU-OS.scaling.reactive-replica-autoscaling

This module owns the *shape and invariants* of scaling authority.  It does not
collect metrics, write graph nodes, or actuate a runtime.  A metrics authority
produces a bounded :class:`SignalSummaryRef`; the graph-facing lifecycle is a
revisioned ``ScaleIntent`` → ``ScaleDecision`` → ``ScaleExecution`` →
``ObservedOutcome`` chain.  High-rate samples never cross this boundary.

The four control cadences remain distinct:

* ``engine_local`` — admission, coalescing, and back-pressure stay in-process;
* ``shard_placement`` — placement/rebalance decisions;
* ``service_replica`` — native replica control or delegated HPA/KEDA control;
* ``agent_topology`` — bounded agent/team topology changes.

Only the latter three cadences may produce graph lifecycle records.  Exactly
one controller registration may write a unit's replica field.  The other
controller mode is observation-only, so a native AU controller and HPA/KEDA
cannot oscillate over the same desired state.
"""

import re
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from typing import Any, Final, Literal, NamedTuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

CONTRACT_VERSION: Final[Literal["1"]] = "1"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_CLOCK_WINDOW_RE = re.compile(
    r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]-(?:[01][0-9]|2[0-3]):[0-5][0-9]$"
)


class ScaleCadence(StrEnum):
    """Independent control loops; they must not collapse into one tick."""

    ENGINE_LOCAL = "engine_local"
    SHARD_PLACEMENT = "shard_placement"
    SERVICE_REPLICA = "service_replica"
    AGENT_TOPOLOGY = "agent_topology"


GRAPH_CADENCES = frozenset(
    {
        ScaleCadence.SHARD_PLACEMENT.value,
        ScaleCadence.SERVICE_REPLICA.value,
        ScaleCadence.AGENT_TOPOLOGY.value,
    }
)


class ControllerMode(StrEnum):
    """The sole replica-writer mode for a scale unit."""

    NATIVE = "native"
    DELEGATED = "delegated"


class DelegatedController(StrEnum):
    """Supported external replica writers."""

    HPA = "hpa"
    KEDA = "keda"


class FailureDomainStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class _ContractModel(BaseModel):
    """Closed, immutable Pydantic boundary for authority records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        use_enum_values=True,
    )


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty identifier")
    value = value.strip()
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} contains unsupported characters")
    return value


def _digest(value: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError("digest must be 64 lowercase hexadecimal characters")
    return value


def _aware_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _nonempty_refs(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError(f"{field_name} must be a sequence")
    result = tuple(_identifier(item, field_name) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


class LeaseFence(_ContractModel):
    """Expiring lease identity carried by every lifecycle transition."""

    lease_id: str
    lease_epoch: StrictInt = Field(ge=1)
    fence_token: StrictInt = Field(ge=1)
    expires_at: datetime

    _validate_lease_id = field_validator("lease_id")(
        lambda value: _identifier(value, "lease_id")
    )
    _validate_expiry = field_validator("expires_at")(
        lambda value: _aware_datetime(value, "expires_at")
    )


class FailureDomain(_ContractModel):
    """Verified placement/failure boundary used by recovery policy."""

    domain_id: str
    kind: Literal["host", "zone", "rack", "cluster", "device"]
    status: FailureDomainStatus = FailureDomainStatus.HEALTHY
    last_observed_at: datetime
    authority_ref: str

    _validate_domain_id = field_validator("domain_id")(
        lambda value: _identifier(value, "domain_id")
    )
    _validate_authority_ref = field_validator("authority_ref")(
        lambda value: _identifier(value, "authority_ref")
    )
    _validate_last_observed = field_validator("last_observed_at")(
        lambda value: _aware_datetime(value, "last_observed_at")
    )


class OfflineRecoveryPolicy(_ContractModel):
    """What a unit may do while its failure-domain authority is unavailable."""

    mode: Literal["hold", "failover", "shed"] = "hold"
    max_staleness_s: float = Field(gt=0, le=86_400)
    allow_scale_up: StrictBool = False
    allow_scale_down: StrictBool = False

    @field_validator("max_staleness_s")
    @classmethod
    def finite_staleness(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("max_staleness_s must be finite")
        return value


class QuotaPolicy(_ContractModel):
    """Bounded quota for a tenant/workload or resource pool."""

    scope: str
    max_units: StrictInt = Field(ge=0)
    burst_units: StrictInt = Field(default=0, ge=0)

    _validate_scope = field_validator("scope")(
        lambda value: _identifier(value, "scope")
    )


class MaintenanceWindow(_ContractModel):
    """UTC maintenance window and policy for queued scale intents."""

    window: str
    outside_policy: Literal["queue", "deny"] = "queue"

    @field_validator("window")
    @classmethod
    def valid_window(cls, value: str) -> str:
        if _CLOCK_WINDOW_RE.fullmatch(value) is None:
            raise ValueError("window must be UTC HH:MM-HH:MM")
        return value


class RollbackPolicy(_ContractModel):
    """Bounded rollback behavior after failed or non-convergent execution."""

    on_failure: Literal["automatic", "hold", "manual"] = "hold"
    max_attempts: StrictInt = Field(default=1, ge=0, le=10)
    require_observed_outcome: StrictBool = True


class ResourcePool(_ContractModel):
    """Capacity authority with reserved headroom and tenant quota."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    pool_id: str
    revision: StrictInt = Field(ge=1)
    resource_kind: Literal[
        "cpu",
        "memory",
        "gpu",
        "kv_cache",
        "disk",
        "network",
        "replica_slots",
        "tokens",
    ]
    capacity_units: StrictInt = Field(ge=0)
    allocated_units: StrictInt = Field(default=0, ge=0)
    reserved_headroom_units: StrictInt = Field(default=0, ge=0)
    failure_domain_id: str
    quota: QuotaPolicy
    lease_fence: LeaseFence
    offline_recovery: OfflineRecoveryPolicy

    _validate_pool_id = field_validator("pool_id")(
        lambda value: _identifier(value, "pool_id")
    )
    _validate_domain = field_validator("failure_domain_id")(
        lambda value: _identifier(value, "failure_domain_id")
    )

    @model_validator(mode="after")
    def validate_capacity(self) -> ResourcePool:
        if self.allocated_units + self.reserved_headroom_units > self.capacity_units:
            raise ValueError(
                "allocated_units plus reserved_headroom_units exceeds capacity_units"
            )
        if self.quota.max_units + self.quota.burst_units > (
            self.capacity_units - self.reserved_headroom_units
        ):
            raise ValueError("pool quota plus burst exceeds capacity after headroom")
        return self


class WorkloadClass(_ContractModel):
    """Workload identity and quota routed into one resource pool."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    workload_id: str
    revision: StrictInt = Field(ge=1)
    tenant_ref: str
    resource_pool_id: str
    cadence: ScaleCadence
    priority: StrictInt = Field(default=0, ge=0, le=10_000)
    max_concurrency: StrictInt = Field(ge=0)
    quota: QuotaPolicy

    _validate_workload_id = field_validator("workload_id")(
        lambda value: _identifier(value, "workload_id")
    )
    _validate_tenant_ref = field_validator("tenant_ref")(
        lambda value: _identifier(value, "tenant_ref")
    )
    _validate_pool_ref = field_validator("resource_pool_id")(
        lambda value: _identifier(value, "resource_pool_id")
    )


class ScaleUnit(_ContractModel):
    """One elastic unit with explicit bounds and one declared replica writer."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    unit_id: str
    revision: StrictInt = Field(ge=1)
    workload_class_id: str
    resource_pool_id: str
    cadence: ScaleCadence
    controller_mode: ControllerMode
    delegated_controller: DelegatedController | None = None
    controller_id: str
    replica_writer_id: str
    min_replicas: StrictInt = Field(ge=0)
    max_replicas: StrictInt = Field(ge=0)
    reserved_headroom_replicas: StrictInt = Field(default=0, ge=0)
    tenant_quota: QuotaPolicy
    failure_domain_id: str
    offline_recovery: OfflineRecoveryPolicy
    maintenance_window: MaintenanceWindow | None = None
    rollback_policy: RollbackPolicy
    depends_on: tuple[str, ...] = ()

    _validate_unit_id = field_validator("unit_id")(
        lambda value: _identifier(value, "unit_id")
    )
    _validate_workload_ref = field_validator("workload_class_id")(
        lambda value: _identifier(value, "workload_class_id")
    )
    _validate_pool_ref = field_validator("resource_pool_id")(
        lambda value: _identifier(value, "resource_pool_id")
    )
    _validate_controller_id = field_validator("controller_id")(
        lambda value: _identifier(value, "controller_id")
    )
    _validate_writer_id = field_validator("replica_writer_id")(
        lambda value: _identifier(value, "replica_writer_id")
    )
    _validate_domain = field_validator("failure_domain_id")(
        lambda value: _identifier(value, "failure_domain_id")
    )

    @field_validator("depends_on", mode="before")
    @classmethod
    def validate_dependencies(cls, value: object) -> tuple[str, ...]:
        return _nonempty_refs(value, "depends_on")

    @model_validator(mode="after")
    def validate_bounds_and_mode(self) -> ScaleUnit:
        if self.max_replicas < self.min_replicas:
            raise ValueError(
                "max_replicas must be greater than or equal to min_replicas"
            )
        if self.tenant_quota.max_units < self.min_replicas:
            raise ValueError("tenant quota must cover min_replicas")
        if self.cadence == ScaleCadence.ENGINE_LOCAL.value and (
            self.controller_mode != ControllerMode.NATIVE.value
        ):
            raise ValueError("engine_local cadence cannot delegate replica control")
        if self.controller_mode == ControllerMode.NATIVE.value:
            if self.delegated_controller is not None:
                raise ValueError("native mode must not declare HPA/KEDA")
        elif self.delegated_controller is None:
            raise ValueError("delegated mode requires explicit HPA or KEDA")
        return self


class ControllerRegistration(_ContractModel):
    """One controller's observation/writer registration for a scale unit."""

    unit_id: str
    controller_id: str
    mode: ControllerMode
    delegated_controller: DelegatedController | None = None
    writes_replicas: StrictBool = False
    lease_fence: LeaseFence

    _validate_unit_id = field_validator("unit_id")(
        lambda value: _identifier(value, "unit_id")
    )
    _validate_controller_id = field_validator("controller_id")(
        lambda value: _identifier(value, "controller_id")
    )

    @model_validator(mode="after")
    def validate_registration_mode(self) -> ControllerRegistration:
        if self.mode == ControllerMode.NATIVE.value:
            if self.delegated_controller is not None:
                raise ValueError("native registration must not declare HPA/KEDA")
        elif self.delegated_controller is None:
            raise ValueError("delegated registration requires explicit HPA or KEDA")
        return self


class SignalSummaryRef(_ContractModel):
    """Bounded pointer to metrics authority output, never a sample payload."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    summary_id: str
    summary_digest: str
    source: Literal["prometheus", "telemetry", "engine"]
    unit_id: str
    signal: str
    aggregation: Literal["last", "mean", "max", "p95", "rate", "sum"]
    window_start: datetime
    window_end: datetime
    sample_count: StrictInt = Field(ge=0, le=1_000_000)

    _validate_summary_id = field_validator("summary_id")(
        lambda value: _identifier(value, "summary_id")
    )
    _validate_digest = field_validator("summary_digest")(lambda value: _digest(value))
    _validate_unit_id = field_validator("unit_id")(
        lambda value: _identifier(value, "unit_id")
    )
    _validate_signal = field_validator("signal")(
        lambda value: _identifier(value, "signal")
    )
    _validate_window_start = field_validator("window_start")(
        lambda value: _aware_datetime(value, "window_start")
    )
    _validate_window_end = field_validator("window_end")(
        lambda value: _aware_datetime(value, "window_end")
    )

    @model_validator(mode="after")
    def validate_window(self) -> SignalSummaryRef:
        if self.window_end <= self.window_start:
            raise ValueError("signal summary window_end must be after window_start")
        return self

    def graph_projection(self) -> dict[str, Any]:
        """Return the bounded graph reference; raw samples never appear."""

        return {
            "schema_version": self.schema_version,
            "summary_id": self.summary_id,
            "summary_digest": self.summary_digest,
            "source": self.source,
            "unit_id": self.unit_id,
            "signal": self.signal,
            "aggregation": self.aggregation,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "sample_count": self.sample_count,
        }


GraphCadence = Literal["shard_placement", "service_replica", "agent_topology"]


class ScaleIntent(_ContractModel):
    """Revisioned, CAS-bound desired change for a graph-persisted cadence."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    intent_id: str
    unit_id: str
    revision: StrictInt = Field(ge=1)
    expected_unit_revision: StrictInt = Field(ge=1)
    cadence: GraphCadence
    desired_replicas: StrictInt = Field(ge=0)
    signal_summary_ref: SignalSummaryRef
    controller_mode: ControllerMode
    delegated_controller: DelegatedController | None = None
    proposer_id: str
    replica_writer_id: str
    lease_fence: LeaseFence
    idempotency_key: str
    reason: str = Field(min_length=1, max_length=512)
    created_at: datetime

    _validate_intent_id = field_validator("intent_id")(
        lambda value: _identifier(value, "intent_id")
    )
    _validate_unit_id = field_validator("unit_id")(
        lambda value: _identifier(value, "unit_id")
    )
    _validate_proposer_id = field_validator("proposer_id")(
        lambda value: _identifier(value, "proposer_id")
    )
    _validate_writer_id = field_validator("replica_writer_id")(
        lambda value: _identifier(value, "replica_writer_id")
    )
    _validate_idempotency = field_validator("idempotency_key")(
        lambda value: _identifier(value, "idempotency_key")
    )
    _validate_created_at = field_validator("created_at")(
        lambda value: _aware_datetime(value, "created_at")
    )

    @model_validator(mode="after")
    def validate_graph_cadence(self) -> ScaleIntent:
        if self.cadence not in GRAPH_CADENCES:
            raise ValueError("engine_local cadence cannot create a graph ScaleIntent")
        if self.signal_summary_ref.unit_id != self.unit_id:
            raise ValueError("signal summary must reference the intent unit")
        if self.controller_mode == ControllerMode.NATIVE.value:
            if self.delegated_controller is not None:
                raise ValueError("native intent must not declare HPA/KEDA")
        elif self.delegated_controller is None:
            raise ValueError("delegated intent requires explicit HPA or KEDA")
        return self


class ScaleDecision(_ContractModel):
    """Policy decision over one revisioned intent."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    decision_id: str
    intent_id: str
    intent_revision: StrictInt = Field(ge=1)
    unit_id: str
    verdict: Literal[
        "accepted",
        "queued",
        "denied",
        "stale",
        "deferred_offline",
        "rejected",
    ]
    controller_mode: ControllerMode
    delegated_controller: DelegatedController | None = None
    decisioner_id: str
    policy_revision: StrictInt = Field(ge=1)
    replica_writer_id: str
    lease_fence: LeaseFence
    reason: str = Field(min_length=1, max_length=512)
    decided_at: datetime

    _validate_decision_id = field_validator("decision_id")(
        lambda value: _identifier(value, "decision_id")
    )
    _validate_intent_id = field_validator("intent_id")(
        lambda value: _identifier(value, "intent_id")
    )
    _validate_unit_id = field_validator("unit_id")(
        lambda value: _identifier(value, "unit_id")
    )
    _validate_decisioner_id = field_validator("decisioner_id")(
        lambda value: _identifier(value, "decisioner_id")
    )
    _validate_writer_id = field_validator("replica_writer_id")(
        lambda value: _identifier(value, "replica_writer_id")
    )
    _validate_decided_at = field_validator("decided_at")(
        lambda value: _aware_datetime(value, "decided_at")
    )

    @model_validator(mode="after")
    def validate_controller_mode(self) -> ScaleDecision:
        if self.controller_mode == ControllerMode.NATIVE.value:
            if self.delegated_controller is not None:
                raise ValueError("native decision must not declare HPA/KEDA")
        elif self.delegated_controller is None:
            raise ValueError("delegated decision requires explicit HPA or KEDA")
        return self


class ScaleExecution(_ContractModel):
    """Idempotent execution record, distinct from the policy decision."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    execution_id: str
    intent_id: str
    decision_id: str
    intent_revision: StrictInt = Field(ge=1)
    unit_id: str
    execution_revision: StrictInt = Field(ge=1)
    requested_replicas: StrictInt = Field(ge=0)
    controller_mode: ControllerMode
    replica_writer_id: str
    lease_fence: LeaseFence
    idempotency_key: str
    operation: Literal["scale", "rollback"]
    state: Literal[
        "prepared",
        "simulated",
        "started",
        "succeeded",
        "failed",
        "rolled_back",
    ]
    started_at: datetime
    completed_at: datetime | None = None
    failure_code: str | None = None
    rollback_target_replicas: StrictInt | None = Field(default=None, ge=0)

    _validate_execution_id = field_validator("execution_id")(
        lambda value: _identifier(value, "execution_id")
    )
    _validate_intent_id = field_validator("intent_id")(
        lambda value: _identifier(value, "intent_id")
    )
    _validate_decision_id = field_validator("decision_id")(
        lambda value: _identifier(value, "decision_id")
    )
    _validate_unit_id = field_validator("unit_id")(
        lambda value: _identifier(value, "unit_id")
    )
    _validate_writer_id = field_validator("replica_writer_id")(
        lambda value: _identifier(value, "replica_writer_id")
    )
    _validate_idempotency = field_validator("idempotency_key")(
        lambda value: _identifier(value, "idempotency_key")
    )
    _validate_started_at = field_validator("started_at")(
        lambda value: _aware_datetime(value, "started_at")
    )

    @field_validator("completed_at")
    @classmethod
    def _completed_at_aware(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware_datetime(value, "completed_at")

    @field_validator("failure_code")
    @classmethod
    def validate_failure_code(cls, value: str | None) -> str | None:
        return None if value is None else _identifier(value, "failure_code")

    @model_validator(mode="after")
    def validate_state(self) -> ScaleExecution:
        terminal = {"succeeded", "failed", "rolled_back"}
        if self.state in terminal and self.completed_at is None:
            raise ValueError("terminal execution state requires completed_at")
        if self.state == "failed" and self.failure_code is None:
            raise ValueError("failed execution requires failure_code")
        if self.state == "simulated" and self.completed_at is not None:
            raise ValueError("simulated execution must not claim completed actuation")
        if self.operation == "rollback" and self.rollback_target_replicas is None:
            raise ValueError("rollback execution requires rollback_target_replicas")
        return self


class ObservedOutcome(_ContractModel):
    """Observed convergence/failure after an execution, written once."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    outcome_id: str
    execution_id: str
    unit_id: str
    execution_revision: StrictInt = Field(ge=1)
    observed_revision: StrictInt = Field(ge=1)
    observed_replicas: StrictInt = Field(ge=0)
    status: Literal[
        "converged",
        "pending",
        "failed",
        "offline",
        "draining",
        "rolled_back",
    ]
    observed_by: str
    failure_domain_id: str
    lease_fence: LeaseFence
    observed_at: datetime
    reason: str = Field(default="", max_length=512)
    rollback_required: StrictBool = False

    _validate_outcome_id = field_validator("outcome_id")(
        lambda value: _identifier(value, "outcome_id")
    )
    _validate_execution_id = field_validator("execution_id")(
        lambda value: _identifier(value, "execution_id")
    )
    _validate_unit_id = field_validator("unit_id")(
        lambda value: _identifier(value, "unit_id")
    )
    _validate_observed_by = field_validator("observed_by")(
        lambda value: _identifier(value, "observed_by")
    )
    _validate_domain = field_validator("failure_domain_id")(
        lambda value: _identifier(value, "failure_domain_id")
    )
    _validate_observed_at = field_validator("observed_at")(
        lambda value: _aware_datetime(value, "observed_at")
    )

    @model_validator(mode="after")
    def validate_rollback(self) -> ObservedOutcome:
        if self.status == "rolled_back" and not self.rollback_required:
            raise ValueError(
                "rolled_back outcome must retain rollback_required evidence"
            )
        return self


class ScaleAuthority(_ContractModel):
    """Cross-record authority inventory and one-writer topology contract."""

    schema_version: Literal["1"] = CONTRACT_VERSION
    authority_revision: StrictInt = Field(ge=1)
    failure_domains: tuple[FailureDomain, ...] = ()
    pools: tuple[ResourcePool, ...] = ()
    workloads: tuple[WorkloadClass, ...] = ()
    units: tuple[ScaleUnit, ...] = ()
    controllers: tuple[ControllerRegistration, ...] = ()

    @staticmethod
    def _unique(values: Iterable[str], label: str) -> None:
        values = tuple(values)
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {label} identity")

    @model_validator(mode="after")
    def validate_inventory(self) -> ScaleAuthority:
        _validate_inventory_identities(self)
        maps = _inventory_maps(self)
        _validate_pool_links(self.pools, maps.domains)
        _validate_workload_links(self.workloads, maps.pools)
        _validate_unit_links(self.units, maps)
        _validate_controller_links(self.units, self.controllers, maps.units)
        _reject_dependency_cycles(self.units)
        return self


def _reject_dependency_cycles(units: Iterable[ScaleUnit]) -> None:
    """Reject dependency cycles so a scale unit cannot wait on itself indirectly."""

    graph = {unit.unit_id: unit.depends_on for unit in units}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(unit_id: str) -> None:
        if unit_id in visiting:
            raise ValueError("scale unit dependency cycle")
        if unit_id in visited:
            return
        visiting.add(unit_id)
        for dependency in graph.get(unit_id, ()):
            visit(dependency)
        visiting.remove(unit_id)
        visited.add(unit_id)

    for unit_id in graph:
        visit(unit_id)


class _AuthorityMaps(NamedTuple):
    """Indexes used while validating one authority inventory."""

    domains: dict[str, FailureDomain]
    pools: dict[str, ResourcePool]
    workloads: dict[str, WorkloadClass]
    units: dict[str, ScaleUnit]


def _validate_inventory_identities(authority: ScaleAuthority) -> None:
    checks = (
        ((domain.domain_id for domain in authority.failure_domains), "failure domain"),
        ((pool.pool_id for pool in authority.pools), "resource pool"),
        ((workload.workload_id for workload in authority.workloads), "workload"),
        ((unit.unit_id for unit in authority.units), "scale unit"),
        (
            (
                f"{registration.unit_id}:{registration.controller_id}"
                for registration in authority.controllers
            ),
            "controller registration",
        ),
    )
    for values, label in checks:
        authority._unique(values, label)


def _inventory_maps(authority: ScaleAuthority) -> _AuthorityMaps:
    return _AuthorityMaps(
        {domain.domain_id: domain for domain in authority.failure_domains},
        {pool.pool_id: pool for pool in authority.pools},
        {workload.workload_id: workload for workload in authority.workloads},
        {unit.unit_id: unit for unit in authority.units},
    )


def _validate_pool_links(
    pools: Iterable[ResourcePool], domains: dict[str, FailureDomain]
) -> None:
    for pool in pools:
        if pool.failure_domain_id not in domains:
            raise ValueError("resource pool references an unknown failure domain")


def _validate_workload_links(
    workloads: Iterable[WorkloadClass], pools: dict[str, ResourcePool]
) -> None:
    for workload in workloads:
        if workload.resource_pool_id not in pools:
            raise ValueError("workload references an unknown resource pool")
        if workload.quota.scope != workload.tenant_ref:
            raise ValueError("workload quota scope must match tenant_ref")
        pool = pools[workload.resource_pool_id]
        if workload.quota.max_units + workload.quota.burst_units > (
            pool.capacity_units - pool.reserved_headroom_units
        ):
            raise ValueError("workload quota plus burst exceeds pool headroom")
        if workload.max_concurrency > workload.quota.max_units:
            raise ValueError("workload concurrency exceeds workload quota")


def _validate_unit_links(units: Iterable[ScaleUnit], maps: _AuthorityMaps) -> None:
    for unit in units:
        workload = maps.workloads.get(unit.workload_class_id)
        if workload is None:
            raise ValueError("scale unit references an unknown workload class")
        _validate_unit_workload(unit, workload)
        unit_pool = maps.pools.get(unit.resource_pool_id)
        unit_pool = _validate_unit_pool(unit, unit_pool)
        _validate_unit_domain(unit, unit_pool, maps.domains)
        _validate_unit_dependencies(unit, maps.units)


def _validate_unit_workload(unit: ScaleUnit, workload: WorkloadClass) -> None:
    if unit.resource_pool_id != workload.resource_pool_id:
        raise ValueError("scale unit pool differs from workload pool")
    if unit.cadence != workload.cadence:
        raise ValueError("scale unit cadence differs from workload cadence")
    if unit.tenant_quota.scope != workload.tenant_ref:
        raise ValueError("scale unit quota scope must match workload tenant")
    if unit.max_replicas > unit.tenant_quota.max_units:
        raise ValueError("scale unit max replicas exceed tenant quota")
    if unit.tenant_quota.max_units > workload.quota.max_units:
        raise ValueError("scale unit quota exceeds workload quota")
    if unit.tenant_quota.burst_units > workload.quota.burst_units:
        raise ValueError("scale unit burst exceeds workload burst")


def _validate_unit_pool(
    unit: ScaleUnit, pool: ResourcePool | None
) -> ResourcePool:
    if pool is None:
        raise ValueError("scale unit references an unknown resource pool")
    if unit.reserved_headroom_replicas + unit.max_replicas > (
        pool.capacity_units - pool.reserved_headroom_units
    ):
        raise ValueError("scale unit max replicas breach pool headroom")
    if unit.tenant_quota.max_units < unit.min_replicas:
        raise ValueError("scale unit quota is below its replica floor")
    return pool


def _validate_unit_domain(
    unit: ScaleUnit,
    pool: ResourcePool,
    domains: dict[str, FailureDomain],
) -> None:
    if unit.failure_domain_id not in domains:
        raise ValueError("scale unit references an unknown failure domain")
    if unit.failure_domain_id != pool.failure_domain_id:
        raise ValueError("scale unit failure domain differs from resource pool")


def _validate_unit_dependencies(
    unit: ScaleUnit, units: dict[str, ScaleUnit]
) -> None:
    for dependency in unit.depends_on:
        if dependency == unit.unit_id:
            raise ValueError("scale unit contains a self-dependency")
        if dependency not in units:
            raise ValueError("scale unit depends on an unknown unit")


def _validate_controller_links(
    units: Iterable[ScaleUnit],
    controllers: Iterable[ControllerRegistration],
    unit_map: dict[str, ScaleUnit],
) -> None:
    registrations_by_unit = _group_controller_registrations(controllers, unit_map)
    for unit in units:
        registrations = registrations_by_unit.get(unit.unit_id, ())
        _validate_unit_registrations(unit, registrations)


def _group_controller_registrations(
    controllers: Iterable[ControllerRegistration], unit_map: dict[str, ScaleUnit]
) -> dict[str, list[ControllerRegistration]]:
    grouped: dict[str, list[ControllerRegistration]] = {}
    for registration in controllers:
        if registration.unit_id not in unit_map:
            raise ValueError("controller registration references an unknown unit")
        grouped.setdefault(registration.unit_id, []).append(registration)
    return grouped


def _validate_unit_registrations(
    unit: ScaleUnit, registrations: Iterable[ControllerRegistration]
) -> None:
    registrations = tuple(registrations)
    for registration in registrations:
        _validate_registration_mode(registration, unit)
    writers = [entry for entry in registrations if entry.writes_replicas]
    if len(writers) != 1:
        raise ValueError(
            f"scale unit {unit.unit_id} requires exactly one replica writer"
        )
    writer = writers[0]
    if writer.controller_id != unit.replica_writer_id:
        raise ValueError("registered replica writer does not match scale unit")
    _validate_registration_mode(writer, unit, writer=True)


def _validate_registration_mode(
    registration: ControllerRegistration, unit: ScaleUnit, *, writer: bool = False
) -> None:
    if registration.mode != unit.controller_mode:
        raise ValueError(
            "replica writer mode does not match scale unit"
            if writer
            else "controller mode does not match scale unit"
        )
    if registration.delegated_controller != unit.delegated_controller:
        raise ValueError("delegated controller does not match scale unit")


def _scale_intent_context(
    intent: ScaleIntent, authority: ScaleAuthority
) -> tuple[ScaleUnit, ResourcePool]:
    """Resolve the unit and pool an intent is authorized to change."""
    unit = next(
        (
            candidate
            for candidate in authority.units
            if candidate.unit_id == intent.unit_id
        ),
        None,
    )
    if unit is None:
        raise ValueError("scale intent references an unknown unit")
    pool = next(
        (
            candidate
            for candidate in authority.pools
            if candidate.pool_id == unit.resource_pool_id
        ),
        None,
    )
    if pool is None:  # pragma: no cover - ScaleAuthority already rejects this
        raise ValueError("scale intent references an unknown resource pool")
    return unit, pool


def _validate_scale_intent_identity(intent: ScaleIntent, unit: ScaleUnit) -> None:
    """Keep an intent on the graph-owned unit revision and writer."""
    if intent.expected_unit_revision != unit.revision:
        raise ValueError("scale intent expected_unit_revision is stale")
    if (
        intent.cadence != unit.cadence
        or intent.cadence == ScaleCadence.ENGINE_LOCAL.value
    ):
        raise ValueError("scale intent cadence does not match graph-owned unit cadence")
    if intent.replica_writer_id != unit.replica_writer_id:
        raise ValueError("scale intent replica writer is not the unit authority")


def _validate_scale_intent_capacity(
    intent: ScaleIntent, unit: ScaleUnit, pool: ResourcePool
) -> None:
    """Enforce unit, tenant, and resource-pool capacity limits."""
    if not unit.min_replicas <= intent.desired_replicas <= unit.max_replicas:
        raise ValueError("scale intent violates unit min/max replica bounds")
    if intent.desired_replicas > unit.tenant_quota.max_units:
        raise ValueError("scale intent exceeds tenant quota")
    if intent.desired_replicas + unit.reserved_headroom_replicas > (
        pool.capacity_units - pool.reserved_headroom_units
    ):
        raise ValueError("scale intent violates resource-pool headroom")


def _validate_scale_intent_failure_domain(
    intent: ScaleIntent, unit: ScaleUnit, authority: ScaleAuthority
) -> None:
    """Reject new desired state while a hold-policy domain is offline."""
    failure_domain = next(
        candidate
        for candidate in authority.failure_domains
        if candidate.domain_id == unit.failure_domain_id
    )
    if (
        failure_domain.status == FailureDomainStatus.OFFLINE.value
        and unit.offline_recovery.mode == "hold"
    ):
        # A stale/offline domain cannot safely authorize a new desired state.
        raise ValueError("scale intent is blocked while failure domain is offline")


def validate_scale_intent(intent: ScaleIntent, authority: ScaleAuthority) -> None:
    """Validate a proposed intent against the current authority revision."""

    unit, pool = _scale_intent_context(intent, authority)
    _validate_scale_intent_identity(intent, unit)
    _validate_scale_intent_capacity(intent, unit, pool)
    _validate_scale_intent_failure_domain(intent, unit, authority)


def _validate_lifecycle_decision(intent: ScaleIntent, decision: ScaleDecision) -> None:
    """Keep a decision bound to its intent and controller authority."""
    if (
        decision.intent_id != intent.intent_id
        or decision.intent_revision != intent.revision
    ):
        raise ValueError("scale decision is not bound to the intent revision")
    if decision.unit_id != intent.unit_id:
        raise ValueError("scale decision is not bound to the intent unit")
    if (
        decision.controller_mode != intent.controller_mode
        or decision.delegated_controller != intent.delegated_controller
    ):
        raise ValueError("scale decision controller mode changed from the intent")


def _validate_lifecycle_execution(
    intent: ScaleIntent,
    decision: ScaleDecision,
    execution: ScaleExecution,
) -> None:
    """Keep execution bound to the decision and intent controller."""
    if (
        execution.intent_id != intent.intent_id
        or execution.intent_revision != intent.revision
    ):
        raise ValueError("scale execution is not bound to the intent revision")
    if (
        execution.decision_id != decision.decision_id
        or execution.unit_id != intent.unit_id
    ):
        raise ValueError("scale execution is not bound to the decision/unit")
    if execution.controller_mode != intent.controller_mode:
        raise ValueError("scale execution controller mode changed from the intent")


def _validate_lifecycle_outcome(
    intent: ScaleIntent, execution: ScaleExecution, outcome: ObservedOutcome
) -> None:
    """Keep an observed outcome bound to its execution and unit."""
    if (
        outcome.execution_id != execution.execution_id
        or outcome.unit_id != intent.unit_id
    ):
        raise ValueError("observed outcome is not bound to the execution/unit")


def _validate_lifecycle_authority(
    intent: ScaleIntent,
    decision: ScaleDecision,
    execution: ScaleExecution,
    outcome: ObservedOutcome,
) -> None:
    """Keep lease fencing and replica-writer authority stable in flight."""
    fences = (
        intent.lease_fence,
        decision.lease_fence,
        execution.lease_fence,
        outcome.lease_fence,
    )
    identity = {
        (fence.lease_id, fence.lease_epoch, fence.fence_token) for fence in fences
    }
    if len(identity) != 1:
        raise ValueError("scale lifecycle lease/fence identity changed mid-flight")
    if execution.replica_writer_id != intent.replica_writer_id:
        raise ValueError("scale execution uses a different replica writer")
    if decision.replica_writer_id != intent.replica_writer_id:
        raise ValueError("scale decision uses a different replica writer")


def validate_scale_lifecycle(
    intent: ScaleIntent,
    decision: ScaleDecision,
    execution: ScaleExecution,
    outcome: ObservedOutcome,
) -> None:
    """Ensure every lifecycle record remains on one intent/unit/fence chain."""

    _validate_lifecycle_decision(intent, decision)
    _validate_lifecycle_execution(intent, decision, execution)
    _validate_lifecycle_outcome(intent, execution, outcome)
    _validate_lifecycle_authority(intent, decision, execution, outcome)


__all__ = [
    "CONTRACT_VERSION",
    "GRAPH_CADENCES",
    "ControllerMode",
    "ControllerRegistration",
    "DelegatedController",
    "FailureDomain",
    "FailureDomainStatus",
    "GraphCadence",
    "LeaseFence",
    "MaintenanceWindow",
    "OfflineRecoveryPolicy",
    "ObservedOutcome",
    "QuotaPolicy",
    "ResourcePool",
    "RollbackPolicy",
    "ScaleAuthority",
    "ScaleCadence",
    "ScaleDecision",
    "ScaleExecution",
    "ScaleIntent",
    "ScaleUnit",
    "SignalSummaryRef",
    "WorkloadClass",
    "validate_scale_intent",
    "validate_scale_lifecycle",
]
