"""Reusable service-surface scale contracts and one fail-closed evaluator.

CONCEPT:AU-OS.scaling.service-scale-units — one typed scale policy for ingestion,
dispatch, gateways, query readers, and connector/media/RLM worker pools.

The module owns no controller or actuator.  It defines the facts a controller
must present and evaluates one bounded target-tracking decision.  Surface-specific
loops are deliberately not encoded: a surface selects a contract profile, then
the same evaluator applies signal freshness, hard capacity, provider-global
quota, continuity, cooldown, drain, and noisy-neighbor rules.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

from agent_utilities.protocols.epistemic_operations import ProtocolModel

SCHEMA_VERSION: Literal["1"] = "1"
MAX_REF_LENGTH = 256
MAX_SIGNALS = 16
MAX_CAPACITY_AXES = 32
MAX_DEMANDS = 32
MAX_QUOTAS = 16
MAX_EVIDENCE = 64
MAX_REASONS = 16
MAX_REPLICAS = 10_000
MAX_STALENESS_SECONDS = 86_400
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SENSITIVE_RE = re.compile(
    r"(?:authorization|bearer|cookie|password|passwd|private[_-]?key|secret|"
    r"token(?:=|:)|api[_-]?key|raw[_-]?(?:body|span|log))",
    re.IGNORECASE,
)

SurfaceKind = Literal[
    "ingest_worker",
    "dispatch_worker",
    "graphos_gateway",
    "mcp_gateway",
    "api_gateway",
    "query_reader",
    "connector_pool",
    "media_pool",
    "rlm_pool",
]
SignalKind = Literal[
    "queue_depth",
    "consumer_lag",
    "backlog_age_ms",
    "request_rate",
    "in_flight",
    "p95_latency_ms",
    "error_rate_ppm",
    "active_sessions",
    "continuity_load",
    "provider_in_flight",
    "gpu_utilization_ppm",
]
CapacityAxis = Literal[
    "cpu_milli",
    "memory_mib",
    "gpu_count",
    "gpu_memory_mib",
    "partition_slots",
    "lease_slots",
    "engine_bytes",
    "engine_write_bytes_per_s",
    "fsync_iops",
    "read_admission_slots",
    "session_slots",
    "continuity_slots",
    "upstream_rate",
    "provider_global_quota",
]
QuotaScope = Literal["unit", "tenant", "engine_global", "provider_global"]
ContinuityMode = Literal["stateless", "session_affine", "externalized"]
ActionKind = Literal["scale_up", "scale_down", "hold", "blocked"]
EvidenceKind = Literal["signal", "capacity", "quota", "continuity", "safety"]

DecisionReason = Literal[
    "at_target",
    "cooldown",
    "signal_missing",
    "signal_stale",
    "signal_mismatch",
    "capacity_missing",
    "capacity_stale",
    "capacity_exhausted",
    "engine_authority_saturated",
    "provider_quota_missing",
    "provider_quota_stale",
    "provider_quota_exhausted",
    "provider_quota_not_multiplicative",
    "quota_missing",
    "quota_stale",
    "quota_scope_mismatch",
    "continuity_missing",
    "continuity_stale",
    "continuity_not_ready",
    "active_sessions_block_drain",
    "scale_from_zero_disabled",
    "scale_to_zero_disabled",
    "drain_required",
    "overload_shedding_active",
    "noisy_neighbor",
    "partition_lease_missing",
    "invalid_bounds",
    "target_tracking",
]


class ScaleContractError(ValueError):
    """Raised when a scale contract or observation is unsafe or ambiguous."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _ref(value: str, *, name: str) -> str:
    if len(value) > MAX_REF_LENGTH or not _REF_RE.fullmatch(value):
        raise ValueError(f"{name} must be a bounded opaque reference")
    if _SENSITIVE_RE.search(value):
        raise ValueError(f"{name} must not contain credentials or raw payloads")
    return value


def _digest(value: str, *, name: str) -> str:
    if not _DIGEST_RE.fullmatch(value):
        raise ValueError(f"{name} must be a sha256:<64 lowercase hex> digest")
    return value


def _utc(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    normalized = value.astimezone(UTC)
    if normalized.year < 2000 or normalized.year > 2300:
        raise ValueError(f"{name} is outside the supported time range")
    return normalized


def _bounded_signal(signal: SignalKind, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("scale signal values must be finite non-negative integers")
    if signal in {"error_rate_ppm", "gpu_utilization_ppm"} and value > 1_000_000:
        raise ValueError(f"{signal} must be bounded to one million ppm")
    return value


class ScalePolicy(ProtocolModel):
    """Bounded target-tracking policy shared by every surface."""

    min_replicas: Annotated[StrictInt, Field(ge=0, le=MAX_REPLICAS)] = 1
    max_replicas: Annotated[StrictInt, Field(ge=1, le=MAX_REPLICAS)] = 1
    target: Annotated[StrictInt, Field(gt=0, le=10**12)]
    target_signal: SignalKind
    scale_up_step: Annotated[StrictInt, Field(gt=0, le=MAX_REPLICAS)] = 1
    scale_down_step: Annotated[StrictInt, Field(gt=0, le=MAX_REPLICAS)] = 1
    scale_up_cooldown_s: Annotated[StrictInt, Field(ge=0, le=MAX_STALENESS_SECONDS)] = (
        30
    )
    scale_down_cooldown_s: Annotated[
        StrictInt, Field(ge=0, le=MAX_STALENESS_SECONDS)
    ] = 300
    drain_seconds: Annotated[StrictInt, Field(ge=0, le=MAX_STALENESS_SECONDS)] = 60
    allow_scale_to_zero: StrictBool = False
    allow_scale_from_zero: StrictBool = True

    @model_validator(mode="after")
    def validate_bounds(self) -> ScalePolicy:
        if self.min_replicas > self.max_replicas:
            raise ScaleContractError("min_replicas cannot exceed max_replicas")
        if self.allow_scale_to_zero and self.min_replicas != 0:
            raise ScaleContractError("scale-to-zero requires min_replicas=0")
        if self.scale_up_step > self.max_replicas:
            raise ScaleContractError("scale_up_step cannot exceed max_replicas")
        if self.scale_down_step > self.max_replicas:
            raise ScaleContractError("scale_down_step cannot exceed max_replicas")
        return self


class CapacityDemand(ProtocolModel):
    """Hard capacity consumed by one replica of a scale unit."""

    axis: CapacityAxis
    per_replica: Annotated[StrictInt, Field(ge=0, le=10**15)]


class QuotaBinding(ProtocolModel):
    """Quota dependency; provider-global budgets never multiply with replicas."""

    quota_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    axis: CapacityAxis
    scope: QuotaScope
    per_replica: Annotated[StrictInt, Field(ge=0, le=10**15)]
    multiplicative: StrictBool = False

    @field_validator("quota_ref")
    @classmethod
    def validate_ref(cls, value: str) -> str:
        return _ref(value, name="quota_ref")

    @model_validator(mode="after")
    def validate_global_quota(self) -> QuotaBinding:
        if self.scope == "provider_global" and self.multiplicative:
            raise ScaleContractError(
                "provider-global quota cannot be multiplied per replica"
            )
        if self.scope == "provider_global" and self.per_replica == 0:
            raise ScaleContractError(
                "provider-global quota requires a declared per-replica demand"
            )
        return self


class PartitionLeaseContract(ProtocolModel):
    """Partition/lease prerequisites for queue and stateful worker surfaces."""

    partitioned: StrictBool = False
    partitions_per_replica: Annotated[StrictInt, Field(ge=0, le=10**6)] = 0
    lease_required: StrictBool = False
    lease_ttl_s: Annotated[StrictInt, Field(ge=0, le=MAX_STALENESS_SECONDS)] = 0
    fencing_required: StrictBool = False

    @model_validator(mode="after")
    def validate_contract(self) -> PartitionLeaseContract:
        if self.partitioned and self.partitions_per_replica < 1:
            raise ScaleContractError(
                "partitioned surfaces require partitions_per_replica"
            )
        if self.lease_required and (self.lease_ttl_s < 1 or not self.fencing_required):
            raise ScaleContractError("leased surfaces require a TTL and fencing")
        if self.fencing_required and not self.lease_required:
            raise ScaleContractError("fencing cannot be declared without leases")
        return self


class ContinuityContract(ProtocolModel):
    """Session/stream continuity prerequisites for scaling and draining."""

    mode: ContinuityMode
    external_store_ref: Annotated[str, Field(max_length=MAX_REF_LENGTH)] | None = None
    drain_required: StrictBool = True
    block_scale_down_with_active: StrictBool = True
    max_drain_seconds: Annotated[StrictInt, Field(ge=0, le=MAX_STALENESS_SECONDS)] = 300

    @field_validator("external_store_ref")
    @classmethod
    def validate_store_ref(cls, value: str | None) -> str | None:
        return _ref(value, name="external_store_ref") if value is not None else None

    @model_validator(mode="after")
    def validate_mode(self) -> ContinuityContract:
        if self.mode == "externalized" and self.external_store_ref is None:
            raise ScaleContractError(
                "externalized continuity requires a store reference"
            )
        if self.mode != "externalized" and self.external_store_ref is not None:
            raise ScaleContractError(
                "only externalized continuity may declare a store reference"
            )
        if self.mode == "stateless" and self.block_scale_down_with_active:
            raise ScaleContractError(
                "stateless continuity cannot block scale-down on active sessions"
            )
        return self


class SignalObservation(ProtocolModel):
    """Fresh bounded signal sample used by target tracking."""

    signal: SignalKind
    value: StrictInt
    observed_at: datetime
    expires_at: datetime
    source_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    source_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: int, info: ValidationInfo) -> int:
        signal = info.data.get("signal")
        return _bounded_signal(signal, value) if signal is not None else value

    @field_validator("observed_at", "expires_at")
    @classmethod
    def validate_times(cls, value: datetime, info: object) -> datetime:
        return _utc(value, name=str(getattr(info, "field_name", "signal_time")))

    @field_validator("source_ref")
    @classmethod
    def validate_source_ref(cls, value: str) -> str:
        return _ref(value, name="signal_source_ref")

    @field_validator("source_digest")
    @classmethod
    def validate_source_digest(cls, value: str) -> str:
        return _digest(value, name="signal_source_digest")

    @model_validator(mode="after")
    def validate_window(self) -> SignalObservation:
        if self.expires_at <= self.observed_at:
            raise ValueError("signal expires_at must follow observed_at")
        return self


class CapacityObservation(ProtocolModel):
    """Fresh allocatable/reserved/used hard-capacity observation."""

    axis: CapacityAxis
    allocatable: Annotated[StrictInt, Field(ge=0, le=10**15)]
    reserved: Annotated[StrictInt, Field(ge=0, le=10**15)]
    used: Annotated[StrictInt, Field(ge=0, le=10**15)]
    observed_at: datetime
    expires_at: datetime
    source_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    source_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]

    @field_validator("observed_at", "expires_at")
    @classmethod
    def validate_times(cls, value: datetime, info: object) -> datetime:
        return _utc(value, name=str(getattr(info, "field_name", "capacity_time")))

    @field_validator("source_ref")
    @classmethod
    def validate_source_ref(cls, value: str) -> str:
        return _ref(value, name="capacity_source_ref")

    @field_validator("source_digest")
    @classmethod
    def validate_source_digest(cls, value: str) -> str:
        return _digest(value, name="capacity_source_digest")

    @model_validator(mode="after")
    def validate_capacity(self) -> CapacityObservation:
        if not self.used <= self.reserved <= self.allocatable:
            raise ScaleContractError(
                "capacity must satisfy used <= reserved <= allocatable"
            )
        if self.expires_at <= self.observed_at:
            raise ValueError("capacity expires_at must follow observed_at")
        return self

    @property
    def available(self) -> int:
        return self.allocatable - self.reserved


class QuotaObservation(ProtocolModel):
    """Fresh global or local quota usage; missing values never mean unlimited."""

    quota_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    scope: QuotaScope
    limit: Annotated[StrictInt, Field(ge=0, le=10**15)]
    used: Annotated[StrictInt, Field(ge=0, le=10**15)]
    observed_at: datetime
    expires_at: datetime
    source_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    source_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]

    @field_validator("quota_ref", "source_ref")
    @classmethod
    def validate_refs(cls, value: str, info: object) -> str:
        return _ref(value, name=str(getattr(info, "field_name", "quota_ref")))

    @field_validator("source_digest")
    @classmethod
    def validate_source_digest(cls, value: str) -> str:
        return _digest(value, name="quota_source_digest")

    @field_validator("observed_at", "expires_at")
    @classmethod
    def validate_times(cls, value: datetime, info: object) -> datetime:
        return _utc(value, name=str(getattr(info, "field_name", "quota_time")))

    @model_validator(mode="after")
    def validate_quota(self) -> QuotaObservation:
        if self.used > self.limit:
            raise ScaleContractError("quota usage cannot exceed its limit")
        if self.expires_at <= self.observed_at:
            raise ValueError("quota expires_at must follow observed_at")
        return self


class ContinuityObservation(ProtocolModel):
    """Fresh continuity state used to gate drain and scale-to-zero."""

    ready: StrictBool
    active_sessions: Annotated[StrictInt, Field(ge=0, le=10**12)]
    active_streams: Annotated[StrictInt, Field(ge=0, le=10**12)]
    observed_at: datetime
    expires_at: datetime
    source_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    source_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]

    @field_validator("source_ref")
    @classmethod
    def validate_source_ref(cls, value: str) -> str:
        return _ref(value, name="continuity_source_ref")

    @field_validator("source_digest")
    @classmethod
    def validate_source_digest(cls, value: str) -> str:
        return _digest(value, name="continuity_source_digest")

    @field_validator("observed_at", "expires_at")
    @classmethod
    def validate_times(cls, value: datetime, info: object) -> datetime:
        return _utc(value, name=str(getattr(info, "field_name", "continuity_time")))

    @model_validator(mode="after")
    def validate_window(self) -> ContinuityObservation:
        if self.expires_at <= self.observed_at:
            raise ValueError("continuity expires_at must follow observed_at")
        return self


class LoadSafetyObservation(ProtocolModel):
    """Overload/noisy-neighbor evidence that keeps scaling policy honest."""

    overloaded: StrictBool
    noisy_neighbor: StrictBool
    reason_ref: Annotated[str, Field(max_length=MAX_REF_LENGTH)] | None = None
    observed_at: datetime
    expires_at: datetime
    source_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    source_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]

    @field_validator("reason_ref", "source_ref")
    @classmethod
    def validate_refs(cls, value: str | None, info: object) -> str | None:
        return (
            _ref(value, name=str(getattr(info, "field_name", "safety_ref")))
            if value is not None
            else None
        )

    @field_validator("source_digest")
    @classmethod
    def validate_source_digest(cls, value: str) -> str:
        return _digest(value, name="safety_source_digest")

    @field_validator("observed_at", "expires_at")
    @classmethod
    def validate_times(cls, value: datetime, info: object) -> datetime:
        return _utc(value, name=str(getattr(info, "field_name", "safety_time")))

    @model_validator(mode="after")
    def validate_window(self) -> LoadSafetyObservation:
        if self.expires_at <= self.observed_at:
            raise ValueError("safety expires_at must follow observed_at")
        if (self.overloaded or self.noisy_neighbor) and self.reason_ref is None:
            raise ScaleContractError(
                "overload and noisy-neighbor evidence require a reason reference"
            )
        return self


class SurfaceProfile(ProtocolModel):
    """Static surface vocabulary used to validate contracts uniformly."""

    allowed_signals: Annotated[tuple[SignalKind, ...], Field(max_length=MAX_SIGNALS)]
    required_capacity_axes: Annotated[
        tuple[CapacityAxis, ...], Field(max_length=MAX_CAPACITY_AXES)
    ]
    engine_authority_axes: Annotated[
        tuple[CapacityAxis, ...], Field(max_length=MAX_CAPACITY_AXES)
    ] = ()
    continuity_modes: Annotated[tuple[ContinuityMode, ...], Field(max_length=3)]
    partition_lease_required: StrictBool = False
    provider_quota_required: StrictBool = False
    provider_quota_axes: Annotated[tuple[CapacityAxis, ...], Field(max_length=4)] = ()


_SURFACE_PROFILES: dict[SurfaceKind, SurfaceProfile] = {
    "ingest_worker": SurfaceProfile(
        allowed_signals=(
            "queue_depth",
            "consumer_lag",
            "backlog_age_ms",
            "p95_latency_ms",
        ),
        required_capacity_axes=(
            "cpu_milli",
            "memory_mib",
            "partition_slots",
            "lease_slots",
            "engine_write_bytes_per_s",
            "fsync_iops",
        ),
        engine_authority_axes=("engine_write_bytes_per_s", "fsync_iops"),
        continuity_modes=("stateless", "externalized"),
        partition_lease_required=True,
    ),
    "dispatch_worker": SurfaceProfile(
        allowed_signals=(
            "queue_depth",
            "consumer_lag",
            "active_sessions",
            "p95_latency_ms",
        ),
        required_capacity_axes=(
            "cpu_milli",
            "memory_mib",
            "partition_slots",
            "lease_slots",
        ),
        continuity_modes=("stateless", "externalized"),
        partition_lease_required=True,
    ),
    "graphos_gateway": SurfaceProfile(
        allowed_signals=(
            "request_rate",
            "in_flight",
            "p95_latency_ms",
            "error_rate_ppm",
        ),
        required_capacity_axes=(
            "cpu_milli",
            "memory_mib",
            "session_slots",
            "read_admission_slots",
            "engine_bytes",
            "fsync_iops",
        ),
        engine_authority_axes=("read_admission_slots", "engine_bytes", "fsync_iops"),
        continuity_modes=("session_affine", "externalized"),
    ),
    "mcp_gateway": SurfaceProfile(
        allowed_signals=(
            "request_rate",
            "in_flight",
            "p95_latency_ms",
            "active_sessions",
            "continuity_load",
            "provider_in_flight",
            "error_rate_ppm",
        ),
        required_capacity_axes=(
            "cpu_milli",
            "memory_mib",
            "session_slots",
            "continuity_slots",
        ),
        continuity_modes=("session_affine", "externalized"),
        provider_quota_required=True,
        provider_quota_axes=("provider_global_quota",),
    ),
    "api_gateway": SurfaceProfile(
        allowed_signals=(
            "request_rate",
            "in_flight",
            "p95_latency_ms",
            "error_rate_ppm",
        ),
        required_capacity_axes=("cpu_milli", "memory_mib", "session_slots"),
        continuity_modes=("stateless", "externalized"),
    ),
    "query_reader": SurfaceProfile(
        allowed_signals=(
            "request_rate",
            "in_flight",
            "p95_latency_ms",
            "error_rate_ppm",
        ),
        required_capacity_axes=(
            "cpu_milli",
            "memory_mib",
            "read_admission_slots",
            "engine_bytes",
            "fsync_iops",
        ),
        engine_authority_axes=("read_admission_slots", "engine_bytes", "fsync_iops"),
        continuity_modes=("stateless", "externalized"),
    ),
    "connector_pool": SurfaceProfile(
        allowed_signals=(
            "queue_depth",
            "consumer_lag",
            "request_rate",
            "provider_in_flight",
            "p95_latency_ms",
            "error_rate_ppm",
        ),
        required_capacity_axes=("cpu_milli", "memory_mib", "upstream_rate"),
        continuity_modes=("stateless", "externalized"),
        provider_quota_required=True,
        provider_quota_axes=("upstream_rate", "provider_global_quota"),
    ),
    "media_pool": SurfaceProfile(
        allowed_signals=(
            "queue_depth",
            "request_rate",
            "gpu_utilization_ppm",
            "p95_latency_ms",
        ),
        required_capacity_axes=(
            "cpu_milli",
            "memory_mib",
            "gpu_count",
            "gpu_memory_mib",
        ),
        continuity_modes=("stateless", "externalized"),
    ),
    "rlm_pool": SurfaceProfile(
        allowed_signals=(
            "queue_depth",
            "request_rate",
            "gpu_utilization_ppm",
            "active_sessions",
        ),
        required_capacity_axes=(
            "cpu_milli",
            "memory_mib",
            "gpu_count",
            "gpu_memory_mib",
        ),
        continuity_modes=("stateless", "externalized"),
    ),
}


class ScaleUnitContract(ProtocolModel):
    """Reusable typed surface contract; no surface-specific evaluator exists."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    unit_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    surface: SurfaceKind
    version: Annotated[StrictInt, Field(gt=0, le=2**31 - 1)] = 1
    policy: ScalePolicy
    allowed_signals: Annotated[tuple[SignalKind, ...], Field(max_length=MAX_SIGNALS)]
    required_capacity_axes: Annotated[
        tuple[CapacityAxis, ...], Field(max_length=MAX_CAPACITY_AXES)
    ]
    replica_demands: Annotated[
        tuple[CapacityDemand, ...], Field(max_length=MAX_DEMANDS)
    ]
    quotas: Annotated[tuple[QuotaBinding, ...], Field(max_length=MAX_QUOTAS)] = ()
    partition_lease: PartitionLeaseContract = Field(
        default_factory=PartitionLeaseContract
    )
    continuity: ContinuityContract
    safety_required: StrictBool = True
    engine_authority_axes: Annotated[
        tuple[CapacityAxis, ...], Field(max_length=MAX_CAPACITY_AXES)
    ] = ()
    contract_id: Annotated[str, Field(min_length=1, max_length=100)] = ""
    contract_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] = ""

    @field_validator("unit_ref")
    @classmethod
    def validate_unit_ref(cls, value: str) -> str:
        return _ref(value, name="unit_ref")

    @model_validator(mode="after")
    def validate_surface_and_identity(self) -> ScaleUnitContract:
        profile = _SURFACE_PROFILES[self.surface]
        allowed = set(self.allowed_signals)
        required_axes = set(self.required_capacity_axes)
        if len(allowed) != len(self.allowed_signals):
            raise ScaleContractError("allowed signal names must be unique")
        if len(required_axes) != len(self.required_capacity_axes):
            raise ScaleContractError("required capacity axes must be unique")
        if not allowed or not allowed.issubset(profile.allowed_signals):
            raise ScaleContractError(
                f"{self.surface} contract declares a signal outside its allowed vocabulary"
            )
        if self.policy.target_signal not in allowed:
            raise ScaleContractError("target_signal must be in allowed_signals")
        if not set(profile.required_capacity_axes).issubset(required_axes):
            raise ScaleContractError(
                f"{self.surface} contract omitted a required hard capacity axis"
            )
        if not set(profile.engine_authority_axes).issubset(required_axes):
            raise ScaleContractError(
                f"{self.surface} engine authority axes must be hard capacity axes"
            )
        declared_engine_axes = set(self.engine_authority_axes)
        if len(declared_engine_axes) != len(self.engine_authority_axes):
            raise ScaleContractError("engine authority axes must be unique")
        if declared_engine_axes != set(profile.engine_authority_axes):
            raise ScaleContractError(
                "engine_authority_axes must match the surface profile"
            )
        if self.continuity.mode not in profile.continuity_modes:
            raise ScaleContractError(
                f"{self.surface} does not support continuity mode {self.continuity.mode}"
            )
        if profile.partition_lease_required and (
            not self.partition_lease.partitioned
            or not self.partition_lease.lease_required
            or not self.partition_lease.fencing_required
        ):
            raise ScaleContractError(
                f"{self.surface} requires partition and fenced lease contracts"
            )
        demand_axes = {demand.axis for demand in self.replica_demands}
        if len(demand_axes) != len(self.replica_demands):
            raise ScaleContractError("replica demand axes must be unique")
        if not required_axes.issubset(demand_axes):
            raise ScaleContractError(
                "every required capacity axis needs a per-replica demand"
            )
        quota_refs = [quota.quota_ref for quota in self.quotas]
        if len(quota_refs) != len(set(quota_refs)):
            raise ScaleContractError("quota bindings must be unique")
        provider_bindings = tuple(
            quota for quota in self.quotas if quota.scope == "provider_global"
        )
        if profile.provider_quota_required:
            if not provider_bindings:
                raise ScaleContractError(
                    f"{self.surface} requires one provider-global quota binding"
                )
            if not any(
                quota.axis in profile.provider_quota_axes for quota in provider_bindings
            ):
                raise ScaleContractError(
                    f"{self.surface} provider quota binding has an unsupported axis"
                )
        identity = {
            "schema_version": self.schema_version,
            "unit_ref": self.unit_ref,
            "surface": self.surface,
            "version": self.version,
            "policy": self.policy.model_dump(mode="json"),
            "allowed_signals": sorted(self.allowed_signals),
            "required_capacity_axes": sorted(self.required_capacity_axes),
            "replica_demands": [
                item.model_dump(mode="json") for item in self.replica_demands
            ],
            "quotas": [item.model_dump(mode="json") for item in self.quotas],
            "partition_lease": self.partition_lease.model_dump(mode="json"),
            "continuity": self.continuity.model_dump(mode="json"),
            "safety_required": self.safety_required,
            "engine_authority_axes": sorted(self.engine_authority_axes),
        }
        expected_id = (
            "scale-unit:" + hashlib.sha256(_canonical(identity)).hexdigest()[:32]
        )
        expected_digest = content_digest(
            {"kind": "scale_unit_contract", "identity": identity}
        )
        if self.contract_id and self.contract_id != expected_id:
            raise ScaleContractError("contract_id does not match immutable identity")
        if self.contract_digest and self.contract_digest != expected_digest:
            raise ScaleContractError(
                "contract_digest does not match immutable identity"
            )
        object.__setattr__(self, "contract_id", expected_id)
        object.__setattr__(self, "contract_digest", expected_digest)
        return self


class DecisionEvidence(ProtocolModel):
    """Opaque bounded evidence attached to a scale decision."""

    kind: EvidenceKind
    ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    status: Literal["accepted", "missing", "stale", "blocked"]
    code: Annotated[str, Field(min_length=1, max_length=64)]

    @field_validator("ref")
    @classmethod
    def validate_ref(cls, value: str) -> str:
        return _ref(value, name="evidence_ref")

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _digest(value, name="evidence_digest")


class ScaleDecision(ProtocolModel):
    """Deterministic scale result; actuation is deliberately outside this module."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    decision_id: Annotated[str, Field(min_length=1, max_length=100)] = ""
    decision_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] = ""
    contract_id: Annotated[str, Field(min_length=1, max_length=100)]
    evaluated_at: datetime
    current_replicas: Annotated[StrictInt, Field(ge=0, le=MAX_REPLICAS)]
    desired_replicas: Annotated[StrictInt, Field(ge=0, le=MAX_REPLICAS)]
    action: ActionKind
    reasons: Annotated[tuple[DecisionReason, ...], Field(max_length=MAX_REASONS)]
    drain_required: StrictBool = False
    evidence: Annotated[
        tuple[DecisionEvidence, ...], Field(max_length=MAX_EVIDENCE)
    ] = ()

    @field_validator("contract_id")
    @classmethod
    def validate_contract_ref(cls, value: str) -> str:
        return _ref(value, name="contract_id")

    @field_validator("evaluated_at")
    @classmethod
    def validate_evaluated_at(cls, value: datetime) -> datetime:
        return _utc(value, name="evaluated_at")

    @model_validator(mode="after")
    def validate_decision(self) -> ScaleDecision:
        if self.action == "scale_up" and self.desired_replicas <= self.current_replicas:
            raise ScaleContractError("scale_up must increase replicas")
        if (
            self.action == "scale_down"
            and self.desired_replicas >= self.current_replicas
        ):
            raise ScaleContractError("scale_down must decrease replicas")
        if self.action == "hold" and self.desired_replicas != self.current_replicas:
            raise ScaleContractError("hold must preserve replica count")
        if self.action == "blocked" and not self.reasons:
            raise ScaleContractError("blocked decisions require reasons")
        if self.action in {"scale_up", "scale_down"} and not self.reasons:
            raise ScaleContractError("scale decisions require a reason")
        identity = {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "evaluated_at": self.evaluated_at.astimezone(UTC).isoformat(),
            "current_replicas": self.current_replicas,
            "desired_replicas": self.desired_replicas,
            "action": self.action,
            "reasons": sorted(self.reasons),
            "drain_required": self.drain_required,
            "evidence": [item.model_dump(mode="json") for item in self.evidence],
        }
        expected_id = (
            "scale-decision:" + hashlib.sha256(_canonical(identity)).hexdigest()[:32]
        )
        expected_digest = content_digest(
            {"kind": "scale_decision", "identity": identity}
        )
        if self.decision_id and self.decision_id != expected_id:
            raise ScaleContractError("decision_id does not match immutable identity")
        if self.decision_digest and self.decision_digest != expected_digest:
            raise ScaleContractError(
                "decision_digest does not match immutable identity"
            )
        object.__setattr__(self, "decision_id", expected_id)
        object.__setattr__(self, "decision_digest", expected_digest)
        return self


def _profile_evidence(
    kind: EvidenceKind,
    ref: str,
    digest: str,
    status: Literal["accepted", "missing", "stale", "blocked"],
    code: str,
) -> DecisionEvidence:
    return DecisionEvidence(kind=kind, ref=ref, digest=digest, status=status, code=code)


def _fresh(observed_at: datetime, expires_at: datetime, now: datetime) -> bool:
    return observed_at <= now < expires_at


def _capacity_map(
    observations: Iterable[CapacityObservation],
) -> dict[CapacityAxis, CapacityObservation]:
    result: dict[CapacityAxis, CapacityObservation] = {}
    for observation in observations:
        if observation.axis in result:
            raise ScaleContractError("duplicate capacity axis observation")
        result[observation.axis] = observation
    return result


def _quota_map(
    observations: Iterable[QuotaObservation],
) -> dict[str, QuotaObservation]:
    result: dict[str, QuotaObservation] = {}
    for observation in observations:
        if observation.quota_ref in result:
            raise ScaleContractError("duplicate quota observation")
        result[observation.quota_ref] = observation
    return result


def _target_replicas(policy: ScalePolicy, value: int) -> int:
    desired = math.ceil(value / policy.target)
    floor = policy.min_replicas
    if floor == 0 and not policy.allow_scale_to_zero:
        floor = 1
    if value == 0 and policy.allow_scale_to_zero:
        desired = 0
    return max(floor, min(policy.max_replicas, desired))


def surface_profile(surface: SurfaceKind) -> SurfaceProfile:
    """Return the immutable vocabulary and hard dependencies for a surface."""

    return _SURFACE_PROFILES[surface]


def evaluate_scale(
    contract: ScaleUnitContract,
    *,
    current_replicas: int,
    signal: SignalObservation | None,
    capacities: Iterable[CapacityObservation],
    quotas: Iterable[QuotaObservation],
    continuity: ContinuityObservation | None,
    safety: LoadSafetyObservation | None,
    now: datetime,
    last_action_at: datetime | None = None,
) -> ScaleDecision:
    """Evaluate one target-tracking decision for any declared service surface.

    Missing/stale dependencies produce ``blocked`` rather than optimistic scale
    actions.  In particular, MCP replicas never multiply a provider-global quota,
    and query readers never claim to fix a saturated engine authority.
    """

    current = _utc(now, name="now")
    if current_replicas < 0 or current_replicas > MAX_REPLICAS:
        raise ScaleContractError("current_replicas is outside bounds")
    if current_replicas < contract.policy.min_replicas:
        raise ScaleContractError("current_replicas is below the contract floor")
    cap_map = _capacity_map(capacities)
    quota_map = _quota_map(quotas)
    evidence: list[DecisionEvidence] = []
    reasons: list[DecisionReason] = []

    if signal is None:
        reasons.append("signal_missing")
        evidence.append(
            _profile_evidence(
                "signal",
                "signal:missing",
                content_digest({"kind": "signal", "state": "missing"}),
                "missing",
                "signal_missing",
            )
        )
    elif signal.signal != contract.policy.target_signal:
        reasons.append("signal_mismatch")
        evidence.append(
            _profile_evidence(
                "signal",
                signal.source_ref,
                signal.source_digest,
                "blocked",
                "signal_mismatch",
            )
        )
    elif not _fresh(signal.observed_at, signal.expires_at, current):
        reasons.append("signal_stale")
        evidence.append(
            _profile_evidence(
                "signal",
                signal.source_ref,
                signal.source_digest,
                "stale",
                "signal_stale",
            )
        )
    else:
        evidence.append(
            _profile_evidence(
                "signal",
                signal.source_ref,
                signal.source_digest,
                "accepted",
                signal.signal,
            )
        )

    missing_capacity = False
    stale_capacity = False
    for axis in contract.required_capacity_axes:
        observation = cap_map.get(axis)
        if observation is None:
            missing_capacity = True
            evidence.append(
                _profile_evidence(
                    "capacity",
                    f"capacity:{axis}",
                    content_digest({"axis": axis, "state": "missing"}),
                    "missing",
                    "capacity_missing",
                )
            )
        elif not _fresh(observation.observed_at, observation.expires_at, current):
            stale_capacity = True
            evidence.append(
                _profile_evidence(
                    "capacity",
                    observation.source_ref,
                    observation.source_digest,
                    "stale",
                    "capacity_stale",
                )
            )
        else:
            evidence.append(
                _profile_evidence(
                    "capacity",
                    observation.source_ref,
                    observation.source_digest,
                    "accepted",
                    axis,
                )
            )
    if missing_capacity:
        reasons.append("capacity_missing")
    if stale_capacity:
        reasons.append("capacity_stale")

    missing_provider_quota = False
    stale_provider_quota = False
    missing_local_quota = False
    stale_local_quota = False
    quota_scope_mismatch = False
    provider_quota_exhausted = False
    local_quota_exhausted = False
    for binding in contract.quotas:
        quota_observation = quota_map.get(binding.quota_ref)
        if quota_observation is None:
            if binding.scope == "provider_global":
                missing_provider_quota = True
            else:
                missing_local_quota = True
            evidence.append(
                _profile_evidence(
                    "quota",
                    binding.quota_ref,
                    content_digest(
                        {"quota_ref": binding.quota_ref, "state": "missing"}
                    ),
                    "missing",
                    "provider_quota_missing"
                    if binding.scope == "provider_global"
                    else "quota_missing",
                )
            )
        elif quota_observation.scope != binding.scope:
            quota_scope_mismatch = True
            evidence.append(
                _profile_evidence(
                    "quota",
                    quota_observation.source_ref,
                    quota_observation.source_digest,
                    "blocked",
                    "quota_scope_mismatch",
                )
            )
        elif not _fresh(
            quota_observation.observed_at, quota_observation.expires_at, current
        ):
            if binding.scope == "provider_global":
                stale_provider_quota = True
            else:
                stale_local_quota = True
            evidence.append(
                _profile_evidence(
                    "quota",
                    quota_observation.source_ref,
                    quota_observation.source_digest,
                    "stale",
                    "provider_quota_stale"
                    if binding.scope == "provider_global"
                    else "quota_stale",
                )
            )
        else:
            evidence.append(
                _profile_evidence(
                    "quota",
                    quota_observation.source_ref,
                    quota_observation.source_digest,
                    "accepted",
                    binding.scope,
                )
            )
            if quota_observation.used >= quota_observation.limit:
                if binding.scope == "provider_global":
                    provider_quota_exhausted = True
                else:
                    local_quota_exhausted = True
    if missing_provider_quota:
        reasons.append("provider_quota_missing")
    if stale_provider_quota:
        reasons.append("provider_quota_stale")
    if missing_local_quota:
        reasons.append("quota_missing")
    if stale_local_quota:
        reasons.append("quota_stale")
    if quota_scope_mismatch:
        reasons.append("quota_scope_mismatch")

    if contract.continuity.mode != "stateless":
        if continuity is None:
            reasons.append("continuity_missing")
            evidence.append(
                _profile_evidence(
                    "continuity",
                    "continuity:missing",
                    content_digest({"kind": "continuity", "state": "missing"}),
                    "missing",
                    "continuity_missing",
                )
            )
        elif not _fresh(continuity.observed_at, continuity.expires_at, current):
            reasons.append("continuity_stale")
            evidence.append(
                _profile_evidence(
                    "continuity",
                    continuity.source_ref,
                    continuity.source_digest,
                    "stale",
                    "continuity_stale",
                )
            )
        elif not continuity.ready:
            reasons.append("continuity_not_ready")
            evidence.append(
                _profile_evidence(
                    "continuity",
                    continuity.source_ref,
                    continuity.source_digest,
                    "blocked",
                    "continuity_not_ready",
                )
            )
        else:
            evidence.append(
                _profile_evidence(
                    "continuity",
                    continuity.source_ref,
                    continuity.source_digest,
                    "accepted",
                    contract.continuity.mode,
                )
            )
    elif continuity is not None and not _fresh(
        continuity.observed_at, continuity.expires_at, current
    ):
        reasons.append("continuity_stale")

    if contract.safety_required:
        if safety is None:
            reasons.append("overload_shedding_active")
            evidence.append(
                _profile_evidence(
                    "safety",
                    "safety:missing",
                    content_digest({"kind": "safety", "state": "missing"}),
                    "missing",
                    "overload_shedding_active",
                )
            )
        elif not _fresh(safety.observed_at, safety.expires_at, current):
            reasons.append("overload_shedding_active")
            evidence.append(
                _profile_evidence(
                    "safety",
                    safety.source_ref,
                    safety.source_digest,
                    "stale",
                    "safety_stale",
                )
            )
        else:
            evidence.append(
                _profile_evidence(
                    "safety",
                    safety.source_ref,
                    safety.source_digest,
                    "accepted",
                    "safety",
                )
            )
            if safety.overloaded:
                reasons.append("overload_shedding_active")
            if safety.noisy_neighbor:
                reasons.append("noisy_neighbor")

    if last_action_at is not None:
        previous = _utc(last_action_at, name="last_action_at")
        if previous > current:
            reasons.append("cooldown")

    if reasons:
        # Dependencies that are not trustworthy block both growth and shrinkage;
        # a controller must not shrink a surface based on stale safety evidence.
        return _decision(
            contract,
            current=current_replicas,
            desired=current_replicas,
            action="blocked",
            reasons=reasons,
            evidence=evidence,
            drain_required=False,
            evaluated_at=current,
        )

    if signal is None:
        raise ScaleContractError(
            "signal validation unexpectedly completed without a signal"
        )
    raw_desired = _target_replicas(contract.policy, signal.value)
    if raw_desired > current_replicas:
        desired = min(current_replicas + contract.policy.scale_up_step, raw_desired)
        action: ActionKind = "scale_up"
    elif raw_desired < current_replicas:
        desired = max(current_replicas - contract.policy.scale_down_step, raw_desired)
        action = "scale_down"
    else:
        return _decision(
            contract,
            current=current_replicas,
            desired=current_replicas,
            action="hold",
            reasons=("at_target",),
            evidence=evidence,
            drain_required=False,
            evaluated_at=current,
        )

    if (
        action == "scale_up"
        and current_replicas == 0
        and not contract.policy.allow_scale_from_zero
    ):
        return _decision(
            contract,
            current=current_replicas,
            desired=current_replicas,
            action="blocked",
            reasons=("scale_from_zero_disabled",),
            evidence=evidence,
            drain_required=False,
            evaluated_at=current,
        )
    if action == "scale_down":
        if desired == 0 and not contract.policy.allow_scale_to_zero:
            desired = max(1, contract.policy.min_replicas)
            if desired == current_replicas:
                return _decision(
                    contract,
                    current=current_replicas,
                    desired=current_replicas,
                    action="hold",
                    reasons=("scale_to_zero_disabled",),
                    evidence=evidence,
                    drain_required=False,
                    evaluated_at=current,
                )
            reasons.append("scale_to_zero_disabled")
        if (
            continuity is not None
            and contract.continuity.block_scale_down_with_active
            and continuity.active_sessions + continuity.active_streams > 0
        ):
            return _decision(
                contract,
                current=current_replicas,
                desired=current_replicas,
                action="blocked",
                reasons=("active_sessions_block_drain",),
                evidence=evidence,
                drain_required=True,
                evaluated_at=current,
            )

    if last_action_at is not None:
        previous = _utc(last_action_at, name="last_action_at")
        cooldown = (
            contract.policy.scale_up_cooldown_s
            if action == "scale_up"
            else contract.policy.scale_down_cooldown_s
        )
        if (current - previous).total_seconds() < cooldown:
            return _decision(
                contract,
                current=current_replicas,
                desired=current_replicas,
                action="hold",
                reasons=("cooldown",),
                evidence=evidence,
                drain_required=False,
                evaluated_at=current,
            )

    if action == "scale_up":
        additional = desired - current_replicas
        for demand in contract.replica_demands:
            observation = cap_map[demand.axis]
            if additional * demand.per_replica > observation.available:
                reasons.append("capacity_exhausted")
                break
        if provider_quota_exhausted:
            reasons.append("provider_quota_exhausted")
        if local_quota_exhausted:
            reasons.append("capacity_exhausted")
        for binding in contract.quotas:
            quota_observation = quota_map[binding.quota_ref]
            if binding.scope == "provider_global":
                if binding.multiplicative:
                    reasons.append("provider_quota_not_multiplicative")
                if (
                    quota_observation.used + additional * binding.per_replica
                    > quota_observation.limit
                ):
                    reasons.append("provider_quota_exhausted")
            elif (
                quota_observation.used + additional * binding.per_replica
                > quota_observation.limit
            ):
                reasons.append("capacity_exhausted")
        for axis in contract.engine_authority_axes:
            if cap_map[axis].available <= 0:
                reasons.append("engine_authority_saturated")
                break
        if reasons:
            return _decision(
                contract,
                current=current_replicas,
                desired=current_replicas,
                action="blocked",
                reasons=reasons,
                evidence=evidence,
                drain_required=False,
                evaluated_at=current,
            )

    drain_required = action == "scale_down" and contract.continuity.drain_required
    if (
        drain_required
        and contract.policy.drain_seconds > contract.continuity.max_drain_seconds
    ):
        return _decision(
            contract,
            current=current_replicas,
            desired=current_replicas,
            action="blocked",
            reasons=("drain_required",),
            evidence=evidence,
            drain_required=True,
            evaluated_at=current,
        )
    return _decision(
        contract,
        current=current_replicas,
        desired=desired,
        action=action,
        reasons=("target_tracking",),
        evidence=evidence,
        drain_required=drain_required,
        evaluated_at=current,
    )


def _decision(
    contract: ScaleUnitContract,
    *,
    current: int,
    desired: int,
    action: ActionKind,
    reasons: Iterable[DecisionReason],
    evidence: Iterable[DecisionEvidence],
    drain_required: bool,
    evaluated_at: datetime,
) -> ScaleDecision:
    normalized_reasons = tuple(dict.fromkeys(reasons))[:MAX_REASONS]
    if action == "blocked" and not normalized_reasons:
        normalized_reasons = ("invalid_bounds",)
    return ScaleDecision(
        contract_id=contract.contract_id,
        evaluated_at=evaluated_at,
        current_replicas=current,
        desired_replicas=desired,
        action=action,
        reasons=normalized_reasons,
        drain_required=drain_required,
        evidence=tuple(evidence)[:MAX_EVIDENCE],
    )


__all__ = [
    "ActionKind",
    "CapacityAxis",
    "CapacityDemand",
    "CapacityObservation",
    "ContinuityContract",
    "ContinuityMode",
    "ContinuityObservation",
    "DecisionEvidence",
    "DecisionReason",
    "EvidenceKind",
    "LoadSafetyObservation",
    "PartitionLeaseContract",
    "QuotaBinding",
    "QuotaObservation",
    "QuotaScope",
    "ScaleContractError",
    "ScaleDecision",
    "ScalePolicy",
    "ScaleUnitContract",
    "SignalKind",
    "SignalObservation",
    "SurfaceKind",
    "SurfaceProfile",
    "content_digest",
    "evaluate_scale",
    "surface_profile",
]
