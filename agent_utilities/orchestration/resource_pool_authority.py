"""Heterogeneous resource-pool capability and placement authority.

CONCEPT:AU-OS.resource-pool-authority — attested, capability-based placement
for heterogeneous CPU/GPU/storage/network pools.

This module is a pure contract and decision seam.  It does not discover hosts,
execute workloads, or accept a hostname as a capability.  A trusted inventory
adapter supplies an immutable, time-bounded snapshot and an opaque attestation
reference; :func:`place` deterministically evaluates the declared capabilities
and returns bounded evidence.  Unknown, stale, unsupported, or unverified facts
never qualify a placement.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import Field, StrictBool, StrictInt, field_validator, model_validator

from agent_utilities.protocols.epistemic_operations import ProtocolModel

SCHEMA_VERSION: Literal["1"] = "1"
MAX_REF_LENGTH = 256
MAX_ISA_FEATURES = 32
MAX_MIG_PROFILES = 32
MAX_CANDIDATES = 256
MAX_EVIDENCE = 64
MAX_DENIAL_REASONS = 16
MAX_RESOURCE_VALUE = 10**15
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SENSITIVE_RE = re.compile(
    r"(?:authorization|bearer|cookie|password|passwd|private[_-]?key|secret|"
    r"token(?:=|:)|api[_-]?key|raw[_-]?(?:body|span|log))",
    re.IGNORECASE,
)

CapabilityState = Literal["known", "absent", "unknown", "unsupported", "stale"]
Architecture = Literal["x86_64", "aarch64", "armv7", "ppc64le", "riscv64"]
GpuRuntime = Literal["cuda", "rocm", "metal", "oneapi", "none"]
AttestationStatus = Literal["verified", "unverified", "invalid", "unknown"]
AttestationAlgorithm = Literal["sigstore", "cosign", "ed25519", "x509", "opaque"]
DecisionStatus = Literal["placed", "denied"]

DenialReason = Literal[
    "future_snapshot",
    "stale_snapshot",
    "unverified_attestation",
    "forged_attestation",
    "unsupported_architecture",
    "missing_isa",
    "unknown_cpu",
    "insufficient_cpu",
    "unknown_memory",
    "insufficient_memory",
    "gpu_unavailable",
    "gpu_runtime_mismatch",
    "gpu_mig_unavailable",
    "unknown_gpu",
    "nvme_unavailable",
    "nvme_iops_insufficient",
    "disk_insufficient",
    "unknown_storage",
    "network_insufficient",
    "unknown_network",
    "energy_limit",
    "unknown_energy",
    "cost_limit",
    "unknown_cost",
]


class ResourcePoolContractError(ValueError):
    """Raised when a capability or placement contract is unsafe or ambiguous."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_digest(value: object) -> str:
    """Return the stable digest used by snapshots, requirements, and decisions."""

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


class ResourceAmount(ProtocolModel):
    """One numeric capability with an explicit truth state.

    ``value`` is present only for ``known``.  An absent, unknown, unsupported, or
    stale value is intentionally not coerced to zero.
    """

    state: CapabilityState
    value: Annotated[StrictInt, Field(ge=0, le=MAX_RESOURCE_VALUE)] | None = None

    @model_validator(mode="after")
    def validate_truth(self) -> ResourceAmount:
        if self.state == "known":
            if self.value is None or self.value < 0:
                raise ValueError("known resource amounts require non-negative values")
        elif self.value is not None:
            raise ValueError("non-known resource amounts must not assert a value")
        return self

    @classmethod
    def known(cls, value: int) -> ResourceAmount:
        return cls(state="known", value=value)

    @classmethod
    def missing(cls, state: CapabilityState = "unknown") -> ResourceAmount:
        if state == "known":
            raise ValueError("known amount requires a value")
        return cls(state=state)


class ResourceAccounting(ProtocolModel):
    """Allocatable, reserved, and used capacity for one resource class.

    Reservations include active use for this contract, so
    ``used <= reserved <= allocatable`` is the only admitted ordering when all
    three observations are known.
    """

    allocatable: ResourceAmount
    reserved: ResourceAmount
    used: ResourceAmount

    @model_validator(mode="after")
    def validate_order(self) -> ResourceAccounting:
        amounts = (self.allocatable, self.reserved, self.used)
        if all(item.state == "known" for item in amounts):
            allocatable = self.allocatable.value
            reserved = self.reserved.value
            used = self.used.value
            if allocatable is None or reserved is None or used is None:
                raise ValueError("known resource accounting requires numeric values")
            if not (used <= reserved <= allocatable):
                raise ValueError(
                    "resource accounting must satisfy used <= reserved <= allocatable"
                )
        return self

    @property
    def available(self) -> int | None:
        if self.allocatable.state != "known" or self.reserved.state != "known":
            return None
        if self.allocatable.value is None or self.reserved.value is None:
            return None
        return self.allocatable.value - self.reserved.value

    @classmethod
    def known(
        cls, allocatable: int, reserved: int = 0, used: int = 0
    ) -> ResourceAccounting:
        return cls(
            allocatable=ResourceAmount.known(allocatable),
            reserved=ResourceAmount.known(reserved),
            used=ResourceAmount.known(used),
        )

    @classmethod
    def missing(cls, state: CapabilityState = "unknown") -> ResourceAccounting:
        return cls(
            allocatable=ResourceAmount.missing(state),
            reserved=ResourceAmount.missing(state),
            used=ResourceAmount.missing(state),
        )


class CpuCapability(ProtocolModel):
    """CPU architecture, ISA evidence, and schedulable CPU headroom."""

    architecture_state: CapabilityState
    architecture: Architecture | None = None
    isa_state: CapabilityState
    isa: Annotated[tuple[str, ...], Field(max_length=MAX_ISA_FEATURES)] = ()
    accounting: ResourceAccounting

    @field_validator("isa")
    @classmethod
    def validate_isa(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(set(values)))
        if any(
            not re.fullmatch(r"[a-z0-9][a-z0-9_.+\-]{0,31}", value)
            for value in normalized
        ):
            raise ValueError("ISA feature names must be bounded tokens")
        return normalized

    @model_validator(mode="after")
    def validate_truth(self) -> CpuCapability:
        if self.architecture_state == "known" and self.architecture is None:
            raise ValueError("known CPU architecture requires an architecture")
        if self.architecture_state != "known" and self.architecture is not None:
            raise ValueError("non-known CPU architecture must not assert a value")
        if self.isa_state == "known" and not self.isa:
            raise ValueError("known CPU ISA requires at least one feature")
        if self.isa_state != "known" and self.isa:
            raise ValueError("non-known CPU ISA must not assert features")
        return self


class MigProfile(ProtocolModel):
    """One GPU/MIG profile with a bounded available slice count."""

    profile_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    state: CapabilityState
    count: ResourceAmount
    memory_mib: ResourceAmount

    @field_validator("profile_ref")
    @classmethod
    def validate_ref(cls, value: str) -> str:
        return _ref(value, name="profile_ref")


class GpuCapability(ProtocolModel):
    """GPU devices, runtime, memory, and optional MIG slices."""

    state: CapabilityState
    runtime: GpuRuntime | None = None
    runtime_version_ref: Annotated[str, Field(max_length=MAX_REF_LENGTH)] | None = None
    device_count: ResourceAmount
    memory: ResourceAccounting
    mig_profiles: Annotated[
        tuple[MigProfile, ...], Field(max_length=MAX_MIG_PROFILES)
    ] = ()

    @field_validator("runtime_version_ref")
    @classmethod
    def validate_runtime_ref(cls, value: str | None) -> str | None:
        return _ref(value, name="runtime_version_ref") if value is not None else None

    @model_validator(mode="after")
    def validate_truth(self) -> GpuCapability:
        if self.state == "known":
            if self.runtime in {None, "none"} or self.device_count.state != "known":
                raise ValueError(
                    "known GPU capability requires runtime and device count"
                )
        elif self.runtime is not None or self.runtime_version_ref is not None:
            raise ValueError("non-known GPU capability must not assert a runtime")
        refs = [profile.profile_ref for profile in self.mig_profiles]
        if len(refs) != len(set(refs)):
            raise ValueError("MIG profile references must be unique")
        return self


class StorageCapability(ProtocolModel):
    """Generic disk and explicit NVMe/I/O capability states."""

    disk: ResourceAccounting
    disk_read_iops: ResourceAmount
    disk_write_iops: ResourceAmount
    nvme_state: CapabilityState
    nvme_device_count: ResourceAmount
    nvme_read_iops: ResourceAmount
    nvme_write_iops: ResourceAmount

    @model_validator(mode="after")
    def validate_nvme_truth(self) -> StorageCapability:
        if self.nvme_state == "known":
            if self.nvme_device_count.state != "known":
                raise ValueError("known NVMe capability requires device count")
        elif any(
            item.state == "known"
            for item in (
                self.nvme_device_count,
                self.nvme_read_iops,
                self.nvme_write_iops,
            )
        ):
            raise ValueError("non-known NVMe capability must not assert measurements")
        return self


class NetworkCapability(ProtocolModel):
    """Network throughput and latency with explicit unknown handling."""

    state: CapabilityState
    ingress_mbps: ResourceAmount
    egress_mbps: ResourceAmount
    latency_us: ResourceAmount


class EnergyCapability(ProtocolModel):
    """Power envelope for energy-aware placement."""

    state: CapabilityState
    max_power_watts: ResourceAmount
    observed_power_watts: ResourceAmount


class CostCapability(ProtocolModel):
    """Bounded operator-published cost estimate; no billing secrets."""

    state: CapabilityState
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] | None = None
    micros_per_hour: ResourceAmount

    @model_validator(mode="after")
    def validate_currency(self) -> CostCapability:
        if self.state == "known" and self.currency is None:
            raise ValueError("known cost capability requires a currency")
        if self.state != "known" and self.currency is not None:
            raise ValueError("non-known cost capability must not assert currency")
        return self


class ResourceCapabilities(ProtocolModel):
    """Complete capability vector for a pool; each optional class is stateful."""

    cpu: CpuCapability
    memory: ResourceAccounting
    gpu: GpuCapability
    storage: StorageCapability
    network: NetworkCapability
    energy: EnergyCapability
    cost: CostCapability


class CapabilityAttestation(ProtocolModel):
    """Opaque signature metadata bound to a capability payload digest."""

    status: AttestationStatus
    algorithm: AttestationAlgorithm
    signer_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    signature_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    subject_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    observed_at: datetime
    expires_at: datetime

    @field_validator("signer_ref", "signature_ref")
    @classmethod
    def validate_refs(cls, value: str, info: object) -> str:
        return _ref(value, name=str(getattr(info, "field_name", "attestation_ref")))

    @field_validator("subject_digest")
    @classmethod
    def validate_subject_digest(cls, value: str) -> str:
        return _digest(value, name="subject_digest")

    @field_validator("observed_at", "expires_at")
    @classmethod
    def validate_times(cls, value: datetime, info: object) -> datetime:
        return _utc(value, name=str(getattr(info, "field_name", "attestation_time")))

    @model_validator(mode="after")
    def validate_window(self) -> CapabilityAttestation:
        if self.expires_at <= self.observed_at:
            raise ValueError("attestation expires_at must follow observed_at")
        return self


class ResourcePoolSnapshot(ProtocolModel):
    """Immutable, revisioned resource-pool snapshot.

    ``subject_digest`` binds attestation metadata to the exact capability vector;
    changing AVX2, GPU runtime, NVMe, or capacity facts invalidates the attestation.
    """

    schema_version: Literal["1"] = SCHEMA_VERSION
    snapshot_id: Annotated[str, Field(min_length=1, max_length=100)] = ""
    snapshot_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] = ""
    pool_ref: Annotated[str, Field(min_length=1, max_length=MAX_REF_LENGTH)]
    revision: Annotated[StrictInt, Field(gt=0, le=2**63 - 1)]
    observed_at: datetime
    expires_at: datetime
    capabilities: ResourceCapabilities
    attestation: CapabilityAttestation

    @field_validator("pool_ref")
    @classmethod
    def validate_pool_ref(cls, value: str) -> str:
        return _ref(value, name="pool_ref")

    @field_validator("observed_at", "expires_at")
    @classmethod
    def validate_times(cls, value: datetime, info: object) -> datetime:
        return _utc(value, name=str(getattr(info, "field_name", "snapshot_time")))

    @model_validator(mode="after")
    def validate_attestation_and_identity(self) -> ResourcePoolSnapshot:
        if self.expires_at <= self.observed_at:
            raise ValueError("snapshot expires_at must follow observed_at")
        if self.attestation.observed_at > self.observed_at:
            raise ValueError("attestation cannot be observed after the snapshot")
        if self.attestation.expires_at < self.expires_at:
            raise ValueError("attestation must cover the snapshot validity window")
        capability_digest = capability_payload_digest(self.capabilities)
        if self.attestation.subject_digest != capability_digest:
            raise ResourcePoolContractError(
                "attestation subject digest does not match capability payload"
            )
        identity = self.identity_payload()
        expected_id = (
            "pool-snapshot:" + hashlib.sha256(_canonical(identity)).hexdigest()[:32]
        )
        expected_digest = content_digest(
            {"kind": "resource_pool_snapshot", "identity": identity}
        )
        if self.snapshot_id and self.snapshot_id != expected_id:
            raise ResourcePoolContractError(
                "snapshot_id does not match immutable identity"
            )
        if self.snapshot_digest and self.snapshot_digest != expected_digest:
            raise ResourcePoolContractError(
                "snapshot_digest does not match immutable identity"
            )
        object.__setattr__(self, "snapshot_id", expected_id)
        object.__setattr__(self, "snapshot_digest", expected_digest)
        return self

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pool_ref": self.pool_ref,
            "revision": self.revision,
            "observed_at": self.observed_at.astimezone(UTC).isoformat(),
            "expires_at": self.expires_at.astimezone(UTC).isoformat(),
            "capabilities": self.capabilities.model_dump(mode="json"),
            "attestation": self.attestation.model_dump(mode="json"),
        }

    def fresh_at(self, now: datetime) -> bool:
        current = _utc(now, name="now")
        return self.observed_at <= current < self.expires_at


def capability_payload_digest(capabilities: ResourceCapabilities) -> str:
    """Digest only the capability vector that an attestation signs."""

    return content_digest(capabilities.model_dump(mode="json"))


class GpuRequirement(ProtocolModel):
    count: Annotated[StrictInt, Field(ge=0, le=128)] = 0
    memory_mib: Annotated[StrictInt, Field(ge=0, le=2**31 - 1)] = 0
    runtime: GpuRuntime | None = None
    mig_profile_ref: Annotated[str, Field(max_length=MAX_REF_LENGTH)] | None = None

    @field_validator("mig_profile_ref")
    @classmethod
    def validate_profile_ref(cls, value: str | None) -> str | None:
        return _ref(value, name="mig_profile_ref") if value is not None else None

    @model_validator(mode="after")
    def validate_requirement(self) -> GpuRequirement:
        if self.count == 0 and (
            self.memory_mib
            or self.runtime is not None
            or self.mig_profile_ref is not None
        ):
            raise ValueError(
                "GPU memory/runtime/MIG requirements need a positive GPU count"
            )
        return self


class NvmeRequirement(ProtocolModel):
    required: StrictBool = False
    device_count: Annotated[StrictInt, Field(ge=0, le=128)] = 0
    read_iops: Annotated[StrictInt, Field(ge=0, le=10**9)] = 0
    write_iops: Annotated[StrictInt, Field(ge=0, le=10**9)] = 0

    @model_validator(mode="after")
    def validate_requirement(self) -> NvmeRequirement:
        if not self.required and any(
            (self.device_count, self.read_iops, self.write_iops)
        ):
            raise ValueError("NVMe thresholds require required=true")
        return self


class NetworkRequirement(ProtocolModel):
    ingress_mbps: Annotated[StrictInt, Field(ge=0, le=10**9)] = 0
    egress_mbps: Annotated[StrictInt, Field(ge=0, le=10**9)] = 0
    max_latency_us: Annotated[StrictInt, Field(ge=0, le=10**9)] | None = None


class PlacementRequirement(ProtocolModel):
    """Capability-only workload requirement with a derived immutable identity."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    requirement_id: Annotated[str, Field(min_length=1, max_length=100)] = ""
    architecture: Architecture | None = None
    required_isa: Annotated[tuple[str, ...], Field(max_length=MAX_ISA_FEATURES)] = ()
    cpu_milli: Annotated[StrictInt, Field(ge=1, le=2**31 - 1)]
    memory_mib: Annotated[StrictInt, Field(ge=1, le=2**31 - 1)]
    disk_mib: Annotated[StrictInt, Field(ge=0, le=2**63 - 1)] = 0
    disk_read_iops: Annotated[StrictInt, Field(ge=0, le=10**9)] = 0
    disk_write_iops: Annotated[StrictInt, Field(ge=0, le=10**9)] = 0
    gpu: GpuRequirement = Field(default_factory=GpuRequirement)
    nvme: NvmeRequirement = Field(default_factory=NvmeRequirement)
    network: NetworkRequirement = Field(default_factory=NetworkRequirement)
    max_power_watts: Annotated[StrictInt, Field(ge=0, le=10**9)] | None = None
    max_cost_micros_per_hour: Annotated[StrictInt, Field(ge=0, le=10**15)] | None = None

    @field_validator("required_isa")
    @classmethod
    def validate_isa(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(set(values)))
        if any(
            not re.fullmatch(r"[a-z0-9][a-z0-9_.+\-]{0,31}", value)
            for value in normalized
        ):
            raise ValueError("required ISA feature names must be bounded tokens")
        return normalized

    @model_validator(mode="after")
    def validate_identity(self) -> PlacementRequirement:
        identity = {
            "schema_version": self.schema_version,
            "architecture": self.architecture,
            "required_isa": sorted(self.required_isa),
            "cpu_milli": self.cpu_milli,
            "memory_mib": self.memory_mib,
            "disk_mib": self.disk_mib,
            "disk_read_iops": self.disk_read_iops,
            "disk_write_iops": self.disk_write_iops,
            "gpu": self.gpu.model_dump(mode="json"),
            "nvme": self.nvme.model_dump(mode="json"),
            "network": self.network.model_dump(mode="json"),
            "max_power_watts": self.max_power_watts,
            "max_cost_micros_per_hour": self.max_cost_micros_per_hour,
        }
        expected = (
            "placement-requirement:"
            + hashlib.sha256(_canonical(identity)).hexdigest()[:32]
        )
        if self.requirement_id and self.requirement_id != expected:
            raise ResourcePoolContractError(
                "requirement_id does not match immutable identity"
            )
        object.__setattr__(self, "requirement_id", expected)
        return self


class PlacementEvidence(ProtocolModel):
    """Bounded per-candidate evidence; no raw host facts or private metadata."""

    snapshot_id: Annotated[str, Field(min_length=1, max_length=100)]
    snapshot_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    eligible: StrictBool
    denial_reasons: Annotated[
        tuple[DenialReason, ...], Field(max_length=MAX_DENIAL_REASONS)
    ] = ()

    @field_validator("snapshot_id")
    @classmethod
    def validate_snapshot_id(cls, value: str) -> str:
        return _ref(value, name="snapshot_id")

    @field_validator("snapshot_digest")
    @classmethod
    def validate_snapshot_digest(cls, value: str) -> str:
        return _digest(value, name="snapshot_digest")

    @model_validator(mode="after")
    def validate_decision(self) -> PlacementEvidence:
        if self.eligible and self.denial_reasons:
            raise ValueError("eligible evidence cannot carry denial reasons")
        if not self.eligible and not self.denial_reasons:
            raise ValueError("denied evidence must expose a reason")
        return self


class PlacementDecision(ProtocolModel):
    """Deterministic capability-based placement result."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    decision_id: Annotated[str, Field(min_length=1, max_length=100)] = ""
    decision_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] = ""
    requirement_id: Annotated[str, Field(min_length=1, max_length=100)]
    evaluated_at: datetime
    status: DecisionStatus
    selected_pool_ref: Annotated[str, Field(max_length=MAX_REF_LENGTH)] | None = None
    selected_snapshot_id: Annotated[str, Field(max_length=100)] | None = None
    selected_snapshot_digest: (
        Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] | None
    ) = None
    denial_reasons: Annotated[
        tuple[DenialReason, ...], Field(max_length=MAX_DENIAL_REASONS)
    ] = ()
    evidence: Annotated[
        tuple[PlacementEvidence, ...], Field(max_length=MAX_EVIDENCE)
    ] = ()
    candidate_count: Annotated[StrictInt, Field(ge=0, le=MAX_CANDIDATES)]
    evidence_truncated: StrictBool = False

    @field_validator("requirement_id", "selected_pool_ref", "selected_snapshot_id")
    @classmethod
    def validate_refs(cls, value: str | None, info: object) -> str | None:
        return (
            _ref(value, name=str(getattr(info, "field_name", "placement_ref")))
            if value is not None
            else None
        )

    @field_validator("selected_snapshot_digest")
    @classmethod
    def validate_selected_digest(cls, value: str | None) -> str | None:
        return (
            _digest(value, name="selected_snapshot_digest")
            if value is not None
            else None
        )

    @field_validator("evaluated_at")
    @classmethod
    def validate_evaluated_at(cls, value: datetime) -> datetime:
        return _utc(value, name="evaluated_at")

    def _assert_evidence_shape(self) -> None:
        if self.candidate_count < len(self.evidence):
            raise ValueError("evidence cannot contain more rows than candidates")
        if self.candidate_count <= MAX_EVIDENCE and (
            self.candidate_count != len(self.evidence) or self.evidence_truncated
        ):
            raise ValueError("small candidate sets require complete evidence")
        if self.candidate_count > MAX_EVIDENCE and not self.evidence_truncated:
            raise ValueError("large candidate sets must declare evidence truncation")
        evidence_ids = [item.snapshot_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("placement evidence snapshot IDs must be unique")

    def _assert_status_consistency(self) -> None:
        if self.status == "placed":
            if not (
                self.selected_pool_ref
                and self.selected_snapshot_id
                and self.selected_snapshot_digest
            ):
                raise ValueError("placed decision requires selected pool and snapshot")
            if self.denial_reasons:
                raise ValueError(
                    "placed decision cannot carry aggregate denial reasons"
                )
        elif not self.denial_reasons:
            raise ValueError("denied decision must expose denial reasons")

    def _decision_identity(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requirement_id": self.requirement_id,
            "evaluated_at": self.evaluated_at.astimezone(UTC).isoformat(),
            "status": self.status,
            "selected_pool_ref": self.selected_pool_ref,
            "selected_snapshot_id": self.selected_snapshot_id,
            "selected_snapshot_digest": self.selected_snapshot_digest,
            "denial_reasons": sorted(self.denial_reasons),
            "evidence": [item.model_dump(mode="json") for item in self.evidence],
            "candidate_count": self.candidate_count,
            "evidence_truncated": self.evidence_truncated,
        }

    def _assert_decision_identity(self, identity: dict[str, Any]) -> tuple[str, str]:
        expected_id = (
            "placement-decision:"
            + hashlib.sha256(_canonical(identity)).hexdigest()[:32]
        )
        expected_digest = content_digest(
            {"kind": "placement_decision", "identity": identity}
        )
        if self.decision_id and self.decision_id != expected_id:
            raise ResourcePoolContractError(
                "decision_id does not match immutable identity"
            )
        if self.decision_digest and self.decision_digest != expected_digest:
            raise ResourcePoolContractError(
                "decision_digest does not match immutable identity"
            )
        return expected_id, expected_digest

    @model_validator(mode="after")
    def validate_result(self) -> PlacementDecision:
        self._assert_evidence_shape()
        self._assert_status_consistency()
        identity = self._decision_identity()
        expected_id, expected_digest = self._assert_decision_identity(identity)
        object.__setattr__(self, "decision_id", expected_id)
        object.__setattr__(self, "decision_digest", expected_digest)
        return self


def _known(accounting: ResourceAccounting) -> bool:
    return all(
        item.state == "known"
        for item in (
            accounting.allocatable,
            accounting.reserved,
            accounting.used,
        )
    )


def _amount_at_least(amount: ResourceAmount, minimum: int) -> bool:
    return (
        amount.state == "known" and amount.value is not None and amount.value >= minimum
    )


def _check_snapshot_freshness_and_attestation(
    snapshot: ResourcePoolSnapshot, now: datetime, reasons: list[DenialReason]
) -> None:
    if now < snapshot.observed_at:
        reasons.append("future_snapshot")
    elif now >= snapshot.expires_at:
        reasons.append("stale_snapshot")
    if snapshot.attestation.status == "invalid":
        reasons.append("forged_attestation")
    elif snapshot.attestation.status != "verified":
        reasons.append("unverified_attestation")


def _check_cpu_capability(
    snapshot: ResourcePoolSnapshot,
    requirement: PlacementRequirement,
    reasons: list[DenialReason],
) -> None:
    cpu = snapshot.capabilities.cpu
    if requirement.architecture is not None:
        if cpu.architecture_state != "known":
            reasons.append("unknown_cpu")
        elif cpu.architecture != requirement.architecture:
            reasons.append("unsupported_architecture")
    if requirement.required_isa:
        if cpu.isa_state != "known":
            reasons.append("unknown_cpu")
        elif not set(requirement.required_isa).issubset(cpu.isa):
            reasons.append("missing_isa")
    if not _known(cpu.accounting):
        reasons.append("unknown_cpu")
    elif (
        cpu.accounting.available is None
        or cpu.accounting.available < requirement.cpu_milli
    ):
        reasons.append("insufficient_cpu")


def _check_memory_capability(
    snapshot: ResourcePoolSnapshot,
    requirement: PlacementRequirement,
    reasons: list[DenialReason],
) -> None:
    memory = snapshot.capabilities.memory
    if not _known(memory):
        reasons.append("unknown_memory")
    elif memory.available is None or memory.available < requirement.memory_mib:
        reasons.append("insufficient_memory")


def _check_disk_capacity(
    disk: ResourceAccounting, required_mib: int, reasons: list[DenialReason]
) -> None:
    if not required_mib:
        return
    if not _known(disk):
        reasons.append("unknown_storage")
    elif disk.available is None or disk.available < required_mib:
        reasons.append("disk_insufficient")


def _check_disk_iops(
    amount: ResourceAmount, required: int, reasons: list[DenialReason]
) -> None:
    if required and not _amount_at_least(amount, required):
        reasons.append(
            "unknown_storage" if amount.state != "known" else "disk_insufficient"
        )


def _check_storage_disk_capability(
    snapshot: ResourcePoolSnapshot,
    requirement: PlacementRequirement,
    reasons: list[DenialReason],
) -> None:
    storage = snapshot.capabilities.storage
    _check_disk_capacity(storage.disk, requirement.disk_mib, reasons)
    _check_disk_iops(storage.disk_read_iops, requirement.disk_read_iops, reasons)
    _check_disk_iops(storage.disk_write_iops, requirement.disk_write_iops, reasons)


def _check_gpu_mig_profile(
    gpu: GpuCapability, gpu_req: GpuRequirement, reasons: list[DenialReason]
) -> None:
    if not gpu_req.mig_profile_ref:
        return
    matching = [
        profile
        for profile in gpu.mig_profiles
        if profile.profile_ref == gpu_req.mig_profile_ref
    ]
    if not matching or not _amount_at_least(matching[0].count, gpu_req.count):
        reasons.append("gpu_mig_unavailable")


def _check_gpu_known_capacity(
    gpu: GpuCapability, gpu_req: GpuRequirement, reasons: list[DenialReason]
) -> None:
    if not _amount_at_least(gpu.device_count, gpu_req.count):
        reasons.append("gpu_unavailable")
    if gpu_req.runtime is not None and gpu.runtime != gpu_req.runtime:
        reasons.append("gpu_runtime_mismatch")
    if gpu_req.memory_mib and (
        gpu.memory.available is None or gpu.memory.available < gpu_req.memory_mib
    ):
        reasons.append("unknown_gpu" if not _known(gpu.memory) else "gpu_unavailable")
    _check_gpu_mig_profile(gpu, gpu_req, reasons)


def _check_gpu_capability(
    snapshot: ResourcePoolSnapshot,
    requirement: PlacementRequirement,
    reasons: list[DenialReason],
) -> None:
    gpu_req = requirement.gpu
    gpu = snapshot.capabilities.gpu
    if not gpu_req.count:
        return
    if gpu.state == "absent":
        reasons.append("gpu_unavailable")
    elif gpu.state != "known":
        reasons.append("unknown_gpu")
    else:
        _check_gpu_known_capacity(gpu, gpu_req, reasons)


def _check_nvme_capability(
    snapshot: ResourcePoolSnapshot,
    requirement: PlacementRequirement,
    reasons: list[DenialReason],
) -> None:
    storage = snapshot.capabilities.storage
    nvme_req = requirement.nvme
    if nvme_req.required:
        if storage.nvme_state == "absent":
            reasons.append("nvme_unavailable")
        elif storage.nvme_state != "known":
            reasons.append("unknown_storage")
        else:
            if not _amount_at_least(storage.nvme_device_count, nvme_req.device_count):
                reasons.append("nvme_unavailable")
            if not _amount_at_least(storage.nvme_read_iops, nvme_req.read_iops):
                reasons.append("nvme_iops_insufficient")
            if not _amount_at_least(storage.nvme_write_iops, nvme_req.write_iops):
                reasons.append("nvme_iops_insufficient")


def _network_requirement_active(network_req: NetworkRequirement) -> bool:
    return bool(
        network_req.ingress_mbps
        or network_req.egress_mbps
        or network_req.max_latency_us is not None
    )


def _check_network_thresholds(
    network: NetworkCapability, network_req: NetworkRequirement
) -> DenialReason | None:
    if network.state != "known":
        return "unknown_network"
    if not _amount_at_least(network.ingress_mbps, network_req.ingress_mbps):
        return "network_insufficient"
    if not _amount_at_least(network.egress_mbps, network_req.egress_mbps):
        return "network_insufficient"
    if network_req.max_latency_us is not None and (
        network.latency_us.state != "known"
        or network.latency_us.value is None
        or network.latency_us.value > network_req.max_latency_us
    ):
        return "network_insufficient"
    return None


def _check_network_capability(
    snapshot: ResourcePoolSnapshot,
    requirement: PlacementRequirement,
    reasons: list[DenialReason],
) -> None:
    network_req = requirement.network
    network = snapshot.capabilities.network
    if not _network_requirement_active(network_req):
        return
    reason = _check_network_thresholds(network, network_req)
    if reason is not None:
        reasons.append(reason)


def _check_energy_capability(
    snapshot: ResourcePoolSnapshot,
    requirement: PlacementRequirement,
    reasons: list[DenialReason],
) -> None:
    energy = snapshot.capabilities.energy
    if requirement.max_power_watts is not None:
        if energy.state != "known" or not _amount_at_least(energy.max_power_watts, 0):
            reasons.append("unknown_energy")
        elif (
            energy.max_power_watts.value is None
            or energy.max_power_watts.value > requirement.max_power_watts
        ):
            reasons.append("energy_limit")


def _check_cost_capability(
    snapshot: ResourcePoolSnapshot,
    requirement: PlacementRequirement,
    reasons: list[DenialReason],
) -> None:
    cost = snapshot.capabilities.cost
    if requirement.max_cost_micros_per_hour is not None:
        if cost.state != "known" or not _amount_at_least(cost.micros_per_hour, 0):
            reasons.append("unknown_cost")
        elif (
            cost.micros_per_hour.value is None
            or cost.micros_per_hour.value > requirement.max_cost_micros_per_hour
        ):
            reasons.append("cost_limit")


def _check_snapshot(
    snapshot: ResourcePoolSnapshot,
    requirement: PlacementRequirement,
    now: datetime,
) -> tuple[DenialReason, ...]:
    reasons: list[DenialReason] = []
    _check_snapshot_freshness_and_attestation(snapshot, now, reasons)
    _check_cpu_capability(snapshot, requirement, reasons)
    _check_memory_capability(snapshot, requirement, reasons)
    _check_storage_disk_capability(snapshot, requirement, reasons)
    _check_gpu_capability(snapshot, requirement, reasons)
    _check_nvme_capability(snapshot, requirement, reasons)
    _check_network_capability(snapshot, requirement, reasons)
    _check_energy_capability(snapshot, requirement, reasons)
    _check_cost_capability(snapshot, requirement, reasons)

    return tuple(dict.fromkeys(reasons))[:MAX_DENIAL_REASONS]


def _candidate_rank(snapshot: ResourcePoolSnapshot) -> tuple[int, int, int, int, str]:
    """Rank only capability headroom; the digest is a stable final tie-breaker."""

    capabilities = snapshot.capabilities
    cpu_available = capabilities.cpu.accounting.available or 0
    memory_available = capabilities.memory.available or 0
    gpu_available = capabilities.gpu.device_count.value or 0
    nvme_iops = capabilities.storage.nvme_read_iops.value or 0
    return (
        -cpu_available,
        -memory_available,
        -gpu_available,
        -nvme_iops,
        snapshot.snapshot_digest,
    )


def _validate_placement_candidates(
    candidates: tuple[ResourcePoolSnapshot, ...],
) -> None:
    if len(candidates) > MAX_CANDIDATES:
        raise ResourcePoolContractError("placement candidate set exceeds bound")
    ids = [snapshot.snapshot_id for snapshot in candidates]
    if len(ids) != len(set(ids)):
        raise ResourcePoolContractError(
            "placement candidates contain duplicate snapshots"
        )


_CheckedSnapshots = tuple[tuple[ResourcePoolSnapshot, tuple[DenialReason, ...]], ...]


def _check_all_snapshots(
    requirement: PlacementRequirement,
    candidates: tuple[ResourcePoolSnapshot, ...],
    current: datetime,
) -> _CheckedSnapshots:
    return tuple(
        (snapshot, _check_snapshot(snapshot, requirement, current))
        for snapshot in sorted(candidates, key=lambda item: item.snapshot_digest)
    )


def _build_all_evidence(checked: _CheckedSnapshots) -> tuple[PlacementEvidence, ...]:
    return tuple(
        PlacementEvidence(
            snapshot_id=snapshot.snapshot_id,
            snapshot_digest=snapshot.snapshot_digest,
            eligible=not reasons,
            denial_reasons=reasons,
        )
        for snapshot, reasons in checked
    )


def _select_eligible_candidate(
    checked: _CheckedSnapshots,
) -> ResourcePoolSnapshot | None:
    eligible = [snapshot for snapshot, reasons in checked if not reasons]
    return min(eligible, key=_candidate_rank) if eligible else None


def _build_evidence_rows(
    all_evidence: tuple[PlacementEvidence, ...],
    selected: ResourcePoolSnapshot | None,
) -> list[PlacementEvidence]:
    evidence_rows = list(all_evidence[:MAX_EVIDENCE])
    if selected is not None and all(
        item.snapshot_id != selected.snapshot_id for item in evidence_rows
    ):
        selected_evidence = next(
            item for item in all_evidence if item.snapshot_id == selected.snapshot_id
        )
        evidence_rows[-1] = selected_evidence
        evidence_rows.sort(key=lambda item: item.snapshot_digest)
    return evidence_rows


def _resolve_decision_fields(
    selected: ResourcePoolSnapshot | None, checked: _CheckedSnapshots
) -> tuple[
    DecisionStatus, tuple[DenialReason, ...], str | None, str | None, str | None
]:
    """``(status, denial_reasons, selected_pool_ref, selected_snapshot_id,
    selected_snapshot_digest)``."""
    if selected is not None:
        return (
            "placed",
            (),
            selected.pool_ref,
            selected.snapshot_id,
            selected.snapshot_digest,
        )
    denial_reasons = tuple(
        sorted({reason for _, reasons in checked for reason in reasons})
    )[:MAX_DENIAL_REASONS]
    if not denial_reasons:
        denial_reasons = ("unknown_cpu",)
    return "denied", denial_reasons, None, None, None


def place(
    requirement: PlacementRequirement,
    snapshots: Iterable[ResourcePoolSnapshot],
    *,
    now: datetime,
) -> PlacementDecision:
    """Choose a pool from truthful capabilities, or deny with bounded reasons.

    Input order and hostnames never influence the result.  Candidates are sorted
    by available capability headroom and a snapshot digest tie-breaker.  A stale,
    unverified, unsupported, or unknown capability is not treated as capacity.
    """

    current = _utc(now, name="now")
    candidates = tuple(snapshots)
    _validate_placement_candidates(candidates)

    checked = _check_all_snapshots(requirement, candidates, current)
    all_evidence = _build_all_evidence(checked)
    selected = _select_eligible_candidate(checked)
    evidence_rows = _build_evidence_rows(all_evidence, selected)
    (
        status,
        denial_reasons,
        selected_pool_ref,
        selected_snapshot_id,
        selected_snapshot_digest,
    ) = _resolve_decision_fields(selected, checked)

    return PlacementDecision(
        requirement_id=requirement.requirement_id,
        evaluated_at=current,
        status=status,
        selected_pool_ref=selected_pool_ref,
        selected_snapshot_id=selected_snapshot_id,
        selected_snapshot_digest=selected_snapshot_digest,
        denial_reasons=denial_reasons,
        evidence=tuple(evidence_rows),
        candidate_count=len(candidates),
        evidence_truncated=len(all_evidence) > MAX_EVIDENCE,
    )


__all__ = [
    "Architecture",
    "AttestationAlgorithm",
    "AttestationStatus",
    "CapabilityAttestation",
    "CapabilityState",
    "CostCapability",
    "CpuCapability",
    "DenialReason",
    "EnergyCapability",
    "GpuCapability",
    "GpuRequirement",
    "GpuRuntime",
    "MigProfile",
    "NetworkCapability",
    "NetworkRequirement",
    "NvmeRequirement",
    "PlacementDecision",
    "PlacementEvidence",
    "PlacementRequirement",
    "ResourceAccounting",
    "ResourceAmount",
    "ResourceCapabilities",
    "ResourcePoolContractError",
    "ResourcePoolSnapshot",
    "StorageCapability",
    "capability_payload_digest",
    "content_digest",
    "place",
]
