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
from dataclasses import dataclass
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
        self._validate_signal_vocabulary(profile)
        required_axes = self._validate_capacity_axes(profile)
        self._validate_continuity_and_lease(profile)
        self._validate_replica_demands(required_axes)
        self._validate_quota_bindings(profile)
        self._stamp_contract_identity()
        return self

    def _validate_signal_vocabulary(self, profile: SurfaceProfile) -> set[SignalKind]:
        """Uniqueness + vocabulary checks on ``allowed_signals``/``target_signal``.

        Extracted from :meth:`validate_surface_and_identity`.
        """
        allowed = set(self.allowed_signals)
        if len(allowed) != len(self.allowed_signals):
            raise ScaleContractError("allowed signal names must be unique")
        if not allowed or not allowed.issubset(profile.allowed_signals):
            raise ScaleContractError(
                f"{self.surface} contract declares a signal outside its allowed vocabulary"
            )
        if self.policy.target_signal not in allowed:
            raise ScaleContractError("target_signal must be in allowed_signals")
        return allowed

    def _validate_capacity_axes(self, profile: SurfaceProfile) -> set[CapacityAxis]:
        """Uniqueness + coverage checks on capacity/engine-authority axes.

        Extracted from :meth:`validate_surface_and_identity`. Returns the
        validated ``required_axes`` set for reuse by
        :meth:`_validate_replica_demands`.
        """
        required_axes = set(self.required_capacity_axes)
        if len(required_axes) != len(self.required_capacity_axes):
            raise ScaleContractError("required capacity axes must be unique")
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
        return required_axes

    def _validate_continuity_and_lease(self, profile: SurfaceProfile) -> None:
        """Continuity-mode + partition/lease checks.

        Extracted from :meth:`validate_surface_and_identity`.
        """
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

    def _validate_replica_demands(self, required_axes: set[CapacityAxis]) -> None:
        """Uniqueness + coverage checks on ``replica_demands``.

        Extracted from :meth:`validate_surface_and_identity`.
        """
        demand_axes = {demand.axis for demand in self.replica_demands}
        if len(demand_axes) != len(self.replica_demands):
            raise ScaleContractError("replica demand axes must be unique")
        if not required_axes.issubset(demand_axes):
            raise ScaleContractError(
                "every required capacity axis needs a per-replica demand"
            )

    def _validate_quota_bindings(self, profile: SurfaceProfile) -> None:
        """Uniqueness + provider-quota checks on ``quotas``.

        Extracted from :meth:`validate_surface_and_identity`.
        """
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

    def _stamp_contract_identity(self) -> None:
        """Compute + verify + stamp ``contract_id``/``contract_digest``.

        Extracted from :meth:`validate_surface_and_identity`.
        """
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
        self._validate_action_transition()
        self._stamp_decision_identity()
        return self

    def _validate_action_transition(self) -> None:
        """Action-vs-replica-count + reasons-required checks.

        Extracted from :meth:`validate_decision`.
        """
        self._validate_replica_delta()
        self._validate_reasons_required()

    def _validate_replica_delta(self) -> None:
        """scale_up/scale_down/hold vs current<->desired replica-count checks.

        Extracted from :meth:`_validate_action_transition`.
        """
        if self.action == "scale_up" and self.desired_replicas <= self.current_replicas:
            raise ScaleContractError("scale_up must increase replicas")
        if (
            self.action == "scale_down"
            and self.desired_replicas >= self.current_replicas
        ):
            raise ScaleContractError("scale_down must decrease replicas")
        if self.action == "hold" and self.desired_replicas != self.current_replicas:
            raise ScaleContractError("hold must preserve replica count")

    def _validate_reasons_required(self) -> None:
        """``blocked``/scale_* actions require at least one reason.

        Extracted from :meth:`_validate_action_transition`.
        """
        if self.action == "blocked" and not self.reasons:
            raise ScaleContractError("blocked decisions require reasons")
        if self.action in {"scale_up", "scale_down"} and not self.reasons:
            raise ScaleContractError("scale decisions require a reason")

    def _stamp_decision_identity(self) -> None:
        """Compute + verify + stamp ``decision_id``/``decision_digest``.

        Extracted from :meth:`validate_decision`.
        """
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


def _check_signal(
    signal: SignalObservation | None,
    contract: ScaleUnitContract,
    current: datetime,
    evidence: list[DecisionEvidence],
    reasons: list[DecisionReason],
) -> None:
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


def _check_capacity_axes(
    contract: ScaleUnitContract,
    cap_map: dict[CapacityAxis, CapacityObservation],
    current: datetime,
    evidence: list[DecisionEvidence],
    reasons: list[DecisionReason],
) -> None:
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


@dataclass
class _QuotaCheckState:
    """Accumulated flags from :func:`_check_quota_binding`, one per :func:`_check_quotas` call."""

    missing_provider_quota: bool = False
    stale_provider_quota: bool = False
    missing_local_quota: bool = False
    stale_local_quota: bool = False
    quota_scope_mismatch: bool = False
    provider_quota_exhausted: bool = False
    local_quota_exhausted: bool = False


def _record_missing_quota(
    binding: QuotaBinding, evidence: list[DecisionEvidence], state: _QuotaCheckState
) -> None:
    """Extracted from :func:`_check_quota_binding`."""
    is_provider = binding.scope == "provider_global"
    if is_provider:
        state.missing_provider_quota = True
    else:
        state.missing_local_quota = True
    evidence.append(
        _profile_evidence(
            "quota",
            binding.quota_ref,
            content_digest({"quota_ref": binding.quota_ref, "state": "missing"}),
            "missing",
            "provider_quota_missing" if is_provider else "quota_missing",
        )
    )


def _record_quota_scope_mismatch(
    quota_observation: QuotaObservation,
    evidence: list[DecisionEvidence],
    state: _QuotaCheckState,
) -> None:
    """Extracted from :func:`_check_quota_binding`."""
    state.quota_scope_mismatch = True
    evidence.append(
        _profile_evidence(
            "quota",
            quota_observation.source_ref,
            quota_observation.source_digest,
            "blocked",
            "quota_scope_mismatch",
        )
    )


def _record_stale_quota(
    binding: QuotaBinding,
    quota_observation: QuotaObservation,
    evidence: list[DecisionEvidence],
    state: _QuotaCheckState,
) -> None:
    """Extracted from :func:`_check_quota_binding`."""
    is_provider = binding.scope == "provider_global"
    if is_provider:
        state.stale_provider_quota = True
    else:
        state.stale_local_quota = True
    evidence.append(
        _profile_evidence(
            "quota",
            quota_observation.source_ref,
            quota_observation.source_digest,
            "stale",
            "provider_quota_stale" if is_provider else "quota_stale",
        )
    )


def _record_accepted_quota(
    binding: QuotaBinding,
    quota_observation: QuotaObservation,
    evidence: list[DecisionEvidence],
    state: _QuotaCheckState,
) -> None:
    """Extracted from :func:`_check_quota_binding`."""
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
            state.provider_quota_exhausted = True
        else:
            state.local_quota_exhausted = True


def _check_quota_binding(
    binding: QuotaBinding,
    quota_map: dict[str, QuotaObservation],
    current: datetime,
    evidence: list[DecisionEvidence],
    state: _QuotaCheckState,
) -> None:
    """Evaluate one quota binding; mutates ``evidence``/``state``.

    Extracted from :func:`_check_quotas`.
    """
    quota_observation = quota_map.get(binding.quota_ref)
    if quota_observation is None:
        _record_missing_quota(binding, evidence, state)
        return
    if quota_observation.scope != binding.scope:
        _record_quota_scope_mismatch(quota_observation, evidence, state)
        return
    if not _fresh(quota_observation.observed_at, quota_observation.expires_at, current):
        _record_stale_quota(binding, quota_observation, evidence, state)
        return
    _record_accepted_quota(binding, quota_observation, evidence, state)


def _quota_state_to_reasons(
    state: _QuotaCheckState, reasons: list[DecisionReason]
) -> None:
    """Translate the accumulated :class:`_QuotaCheckState` into reason codes.

    Extracted from :func:`_check_quotas`.
    """
    if state.missing_provider_quota:
        reasons.append("provider_quota_missing")
    if state.stale_provider_quota:
        reasons.append("provider_quota_stale")
    if state.missing_local_quota:
        reasons.append("quota_missing")
    if state.stale_local_quota:
        reasons.append("quota_stale")
    if state.quota_scope_mismatch:
        reasons.append("quota_scope_mismatch")


def _check_quotas(
    contract: ScaleUnitContract,
    quota_map: dict[str, QuotaObservation],
    current: datetime,
    evidence: list[DecisionEvidence],
    reasons: list[DecisionReason],
) -> tuple[bool, bool]:
    """Returns (provider_quota_exhausted, local_quota_exhausted)."""
    state = _QuotaCheckState()
    for binding in contract.quotas:
        _check_quota_binding(binding, quota_map, current, evidence, state)
    _quota_state_to_reasons(state, reasons)
    return state.provider_quota_exhausted, state.local_quota_exhausted


def _check_continuity(
    contract: ScaleUnitContract,
    continuity: ContinuityObservation | None,
    current: datetime,
    evidence: list[DecisionEvidence],
    reasons: list[DecisionReason],
) -> None:
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


def _check_safety(
    contract: ScaleUnitContract,
    safety: LoadSafetyObservation | None,
    current: datetime,
    evidence: list[DecisionEvidence],
    reasons: list[DecisionReason],
) -> None:
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


def _capacity_exhaustion_reasons(
    contract: ScaleUnitContract,
    cap_map: dict[CapacityAxis, CapacityObservation],
    additional: int,
) -> list[DecisionReason]:
    """``["capacity_exhausted"]`` iff any replica demand axis lacks headroom.

    Extracted from :func:`_check_scale_up_feasibility`.
    """
    for demand in contract.replica_demands:
        observation = cap_map[demand.axis]
        if additional * demand.per_replica > observation.available:
            return ["capacity_exhausted"]
    return []


def _quota_growth_reasons(
    contract: ScaleUnitContract,
    quota_map: dict[str, QuotaObservation],
    additional: int,
) -> list[DecisionReason]:
    """Reasons from projecting every quota binding's usage forward by ``additional``.

    Extracted from :func:`_check_scale_up_feasibility`.
    """
    reasons: list[DecisionReason] = []
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
    return reasons


def _engine_authority_reasons(
    contract: ScaleUnitContract, cap_map: dict[CapacityAxis, CapacityObservation]
) -> list[DecisionReason]:
    """``["engine_authority_saturated"]`` iff any engine-authority axis is exhausted.

    Extracted from :func:`_check_scale_up_feasibility`.
    """
    for axis in contract.engine_authority_axes:
        if cap_map[axis].available <= 0:
            return ["engine_authority_saturated"]
    return []


def _check_scale_up_feasibility(
    contract: ScaleUnitContract,
    cap_map: dict[CapacityAxis, CapacityObservation],
    quota_map: dict[str, QuotaObservation],
    *,
    desired: int,
    current_replicas: int,
    current: datetime,
    evidence: list[DecisionEvidence],
    provider_quota_exhausted: bool,
    local_quota_exhausted: bool,
) -> ScaleDecision | None:
    """Returns a blocked ScaleDecision if scale-up is infeasible, else None."""
    additional = desired - current_replicas
    reasons: list[DecisionReason] = list(
        _capacity_exhaustion_reasons(contract, cap_map, additional)
    )
    if provider_quota_exhausted:
        reasons.append("provider_quota_exhausted")
    if local_quota_exhausted:
        reasons.append("capacity_exhausted")
    reasons.extend(_quota_growth_reasons(contract, quota_map, additional))
    reasons.extend(_engine_authority_reasons(contract, cap_map))
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
    return None


@dataclass(frozen=True)
class _ScaleDecisionContext:
    """The scale-decision inputs threaded through every terminal-decision call.

    Extracted from :func:`_compute_scale_decision`, to avoid repeating the same
    contract/current_replicas/current/evidence quadruple at every call site.
    """

    contract: ScaleUnitContract
    current_replicas: int
    current: datetime
    evidence: list[DecisionEvidence]


def _terminal_decision(
    ctx: _ScaleDecisionContext,
    *,
    action: ActionKind,
    reasons: tuple[DecisionReason, ...],
    drain_required: bool = False,
) -> ScaleDecision:
    """A decision holding replicas at their current count.

    Extracted from :func:`_compute_scale_decision`.
    """
    return _decision(
        ctx.contract,
        current=ctx.current_replicas,
        desired=ctx.current_replicas,
        action=action,
        reasons=reasons,
        evidence=ctx.evidence,
        drain_required=drain_required,
        evaluated_at=ctx.current,
    )


def _check_scale_from_zero(
    ctx: _ScaleDecisionContext, policy: ScalePolicy, action: ActionKind
) -> ScaleDecision | None:
    """Extracted from :func:`_compute_scale_decision`."""
    if (
        action == "scale_up"
        and ctx.current_replicas == 0
        and not policy.allow_scale_from_zero
    ):
        return _terminal_decision(
            ctx, action="blocked", reasons=("scale_from_zero_disabled",)
        )
    return None


def _adjust_scale_to_zero(
    ctx: _ScaleDecisionContext,
    policy: ScalePolicy,
    desired: int,
    reasons: list[DecisionReason],
) -> tuple[int, ScaleDecision | None]:
    """Clamp ``desired`` up when scale-to-zero is disabled; hold if that clamp
    lands back on the current replica count.

    Extracted from :func:`_compute_scale_decision`. ``reasons.append`` here is
    pre-existing DEBT (not created by this diff): ``reasons`` is a write-only
    list in the caller -- every terminal ``_decision``/``_terminal_decision``
    call passes a literal reasons tuple instead of reading it back. Kept
    verbatim, not fixed, per lane scope.
    """
    if desired != 0 or policy.allow_scale_to_zero:
        return desired, None
    desired = max(1, policy.min_replicas)
    if desired == ctx.current_replicas:
        return desired, _terminal_decision(
            ctx, action="hold", reasons=("scale_to_zero_disabled",)
        )
    reasons.append("scale_to_zero_disabled")
    return desired, None


def _check_active_sessions_block_drain(
    ctx: _ScaleDecisionContext, continuity: ContinuityObservation | None
) -> ScaleDecision | None:
    """Extracted from :func:`_compute_scale_decision`."""
    if (
        continuity is not None
        and ctx.contract.continuity.block_scale_down_with_active
        and continuity.active_sessions + continuity.active_streams > 0
    ):
        return _terminal_decision(
            ctx,
            action="blocked",
            reasons=("active_sessions_block_drain",),
            drain_required=True,
        )
    return None


def _evaluate_scale_down_gates(
    ctx: _ScaleDecisionContext,
    policy: ScalePolicy,
    desired: int,
    reasons: list[DecisionReason],
    continuity: ContinuityObservation | None,
) -> tuple[int, ScaleDecision | None]:
    """The scale_down-only gates: zero-clamp, then the active-sessions block.

    Extracted from :func:`_compute_scale_decision` so its own ``if action ==
    "scale_down":`` branch stays flat (one call, not two nested checks).
    """
    desired, blocked = _adjust_scale_to_zero(ctx, policy, desired, reasons)
    if blocked is not None:
        return desired, blocked
    return desired, _check_active_sessions_block_drain(ctx, continuity)


def _check_cooldown(
    ctx: _ScaleDecisionContext,
    policy: ScalePolicy,
    action: ActionKind,
    last_action_at: datetime | None,
) -> ScaleDecision | None:
    """Extracted from :func:`_compute_scale_decision`."""
    if last_action_at is None:
        return None
    previous = _utc(last_action_at, name="last_action_at")
    cooldown = (
        policy.scale_up_cooldown_s
        if action == "scale_up"
        else policy.scale_down_cooldown_s
    )
    if (ctx.current - previous).total_seconds() < cooldown:
        return _terminal_decision(ctx, action="hold", reasons=("cooldown",))
    return None


def _check_drain_feasibility(
    ctx: _ScaleDecisionContext, drain_required: bool
) -> ScaleDecision | None:
    """Extracted from :func:`_compute_scale_decision`."""
    if (
        drain_required
        and ctx.contract.policy.drain_seconds
        > ctx.contract.continuity.max_drain_seconds
    ):
        return _terminal_decision(
            ctx, action="blocked", reasons=("drain_required",), drain_required=True
        )
    return None


def _resolve_scale_target(
    ctx: _ScaleDecisionContext, policy: ScalePolicy, raw_desired: int
) -> tuple[int, ActionKind, ScaleDecision | None]:
    """The stepped ``(desired, action)`` target, or the "at_target" hold decision.

    Extracted from :func:`_compute_scale_decision`. The third element is
    ``None`` unless ``raw_desired`` already equals the current replica count.
    """
    if raw_desired > ctx.current_replicas:
        desired = min(ctx.current_replicas + policy.scale_up_step, raw_desired)
        return desired, "scale_up", None
    if raw_desired < ctx.current_replicas:
        desired = max(ctx.current_replicas - policy.scale_down_step, raw_desired)
        return desired, "scale_down", None
    return (
        ctx.current_replicas,
        "hold",
        _terminal_decision(ctx, action="hold", reasons=("at_target",)),
    )


def _finalize_scale_decision(
    ctx: _ScaleDecisionContext,
    cap_map: dict[CapacityAxis, CapacityObservation],
    quota_map: dict[str, QuotaObservation],
    *,
    action: ActionKind,
    desired: int,
    provider_quota_exhausted: bool,
    local_quota_exhausted: bool,
) -> ScaleDecision:
    """The scale_up feasibility gate, the drain-feasibility gate, and (absent a
    block) the final ``target_tracking`` decision.

    Extracted from :func:`_compute_scale_decision`.
    """
    if action == "scale_up":
        blocked = _check_scale_up_feasibility(
            ctx.contract,
            cap_map,
            quota_map,
            desired=desired,
            current_replicas=ctx.current_replicas,
            current=ctx.current,
            evidence=ctx.evidence,
            provider_quota_exhausted=provider_quota_exhausted,
            local_quota_exhausted=local_quota_exhausted,
        )
        if blocked is not None:
            return blocked

    drain_required = action == "scale_down" and ctx.contract.continuity.drain_required
    blocked = _check_drain_feasibility(ctx, drain_required)
    if blocked is not None:
        return blocked
    return _decision(
        ctx.contract,
        current=ctx.current_replicas,
        desired=desired,
        action=action,
        reasons=("target_tracking",),
        evidence=ctx.evidence,
        drain_required=drain_required,
        evaluated_at=ctx.current,
    )


def _compute_scale_decision(
    contract: ScaleUnitContract,
    *,
    current_replicas: int,
    signal: SignalObservation | None,
    cap_map: dict[CapacityAxis, CapacityObservation],
    quota_map: dict[str, QuotaObservation],
    continuity: ContinuityObservation | None,
    last_action_at: datetime | None,
    current: datetime,
    evidence: list[DecisionEvidence],
    provider_quota_exhausted: bool,
    local_quota_exhausted: bool,
) -> ScaleDecision:
    reasons: list[DecisionReason] = []
    if signal is None:
        raise ScaleContractError(
            "signal validation unexpectedly completed without a signal"
        )
    ctx = _ScaleDecisionContext(
        contract=contract,
        current_replicas=current_replicas,
        current=current,
        evidence=evidence,
    )
    policy = contract.policy
    raw_desired = _target_replicas(policy, signal.value)
    desired, action, blocked = _resolve_scale_target(ctx, policy, raw_desired)
    if blocked is not None:
        return blocked

    blocked = _check_scale_from_zero(ctx, policy, action)
    if blocked is not None:
        return blocked

    if action == "scale_down":
        desired, blocked = _evaluate_scale_down_gates(
            ctx, policy, desired, reasons, continuity
        )
        if blocked is not None:
            return blocked

    blocked = _check_cooldown(ctx, policy, action, last_action_at)
    if blocked is not None:
        return blocked

    return _finalize_scale_decision(
        ctx,
        cap_map,
        quota_map,
        action=action,
        desired=desired,
        provider_quota_exhausted=provider_quota_exhausted,
        local_quota_exhausted=local_quota_exhausted,
    )


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

    _check_signal(signal, contract, current, evidence, reasons)
    _check_capacity_axes(contract, cap_map, current, evidence, reasons)
    provider_quota_exhausted, local_quota_exhausted = _check_quotas(
        contract, quota_map, current, evidence, reasons
    )
    _check_continuity(contract, continuity, current, evidence, reasons)
    _check_safety(contract, safety, current, evidence, reasons)

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

    return _compute_scale_decision(
        contract,
        current_replicas=current_replicas,
        signal=signal,
        cap_map=cap_map,
        quota_map=quota_map,
        continuity=continuity,
        last_action_at=last_action_at,
        current=current,
        evidence=evidence,
        provider_quota_exhausted=provider_quota_exhausted,
        local_quota_exhausted=local_quota_exhausted,
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
