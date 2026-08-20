"""Tests for CONCEPT:AU-OS.resource-pool-authority — hardware-aware placement."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from agent_utilities.orchestration.resource_pool_authority import (
    CapabilityAttestation,
    CostCapability,
    CpuCapability,
    EnergyCapability,
    GpuCapability,
    NetworkCapability,
    NvmeRequirement,
    PlacementRequirement,
    ResourceAccounting,
    ResourceAmount,
    ResourceCapabilities,
    ResourcePoolSnapshot,
    StorageCapability,
    capability_payload_digest,
    place,
)

NOW = datetime(2026, 8, 19, 12, tzinfo=UTC)


def snapshot(
    *,
    pool_ref: str = "pool:gr1080",
    memory_mib: int = 22 * 1024,
    memory_state: str = "known",
    attestation_status: str = "verified",
    nvme_state: str = "absent",
) -> ResourcePoolSnapshot:
    capabilities = ResourceCapabilities(
        cpu=CpuCapability(
            architecture_state="known",
            architecture="x86_64",
            isa_state="known",
            isa=("avx2",),
            accounting=ResourceAccounting.known(8_000),
        ),
        memory=(
            ResourceAccounting.known(memory_mib)
            if memory_state == "known"
            else ResourceAccounting.missing(memory_state)
        ),
        gpu=GpuCapability(
            state="known",
            runtime="cuda",
            runtime_version_ref="runtime:cuda-12",
            device_count=ResourceAmount.known(1),
            memory=ResourceAccounting.known(16 * 1024),
        ),
        storage=StorageCapability(
            disk=ResourceAccounting.known(512 * 1024),
            disk_read_iops=ResourceAmount.known(20_000),
            disk_write_iops=ResourceAmount.known(20_000),
            nvme_state=nvme_state,
            nvme_device_count=ResourceAmount.missing(nvme_state),
            nvme_read_iops=ResourceAmount.missing(nvme_state),
            nvme_write_iops=ResourceAmount.missing(nvme_state),
        ),
        network=NetworkCapability(
            state="known",
            ingress_mbps=ResourceAmount.known(10_000),
            egress_mbps=ResourceAmount.known(10_000),
            latency_us=ResourceAmount.known(500),
        ),
        energy=EnergyCapability(
            state="known",
            max_power_watts=ResourceAmount.known(250),
            observed_power_watts=ResourceAmount.known(90),
        ),
        cost=CostCapability(
            state="known",
            currency="USD",
            micros_per_hour=ResourceAmount.known(100),
        ),
    )
    observed_at = NOW - timedelta(hours=1)
    expires_at = NOW + timedelta(hours=1)
    return ResourcePoolSnapshot(
        pool_ref=pool_ref,
        revision=1,
        observed_at=observed_at,
        expires_at=expires_at,
        capabilities=capabilities,
        attestation=CapabilityAttestation(
            status=attestation_status,
            algorithm="ed25519",
            signer_ref="signer:inventory",
            signature_ref="signature:gr1080:1",
            subject_digest=capability_payload_digest(capabilities),
            observed_at=observed_at,
            expires_at=expires_at,
        ),
    )


def large_build_requirement() -> PlacementRequirement:
    return PlacementRequirement(
        architecture="x86_64",
        required_isa=("avx2",),
        cpu_milli=4_000,
        memory_mib=32 * 1024,
        nvme=NvmeRequirement(
            required=True,
            device_count=1,
            read_iops=50_000,
            write_iops=30_000,
        ),
    )


def test_gr1080_does_not_qualify_for_large_build_or_nvme() -> None:
    decision = place(
        large_build_requirement(),
        (snapshot(),),
        now=NOW,
    )

    assert decision.status == "denied"
    assert decision.selected_pool_ref is None
    assert "insufficient_memory" in decision.denial_reasons
    assert "nvme_unavailable" in decision.denial_reasons


def test_known_capability_placement_is_order_independent_and_replayable() -> None:
    requirement = PlacementRequirement(
        architecture="x86_64",
        required_isa=("avx2",),
        cpu_milli=1_000,
        memory_mib=4 * 1024,
    )
    first = snapshot(pool_ref="pool:gr1080")
    second = snapshot(pool_ref="pool:edge-a", memory_mib=24 * 1024)

    left = place(requirement, (first, second), now=NOW)
    right = place(requirement, (second, first), now=NOW)

    assert left.status == "placed"
    assert left.selected_pool_ref == right.selected_pool_ref
    assert left.decision_digest == right.decision_digest


def test_unknown_and_unverified_capabilities_fail_closed() -> None:
    requirement = PlacementRequirement(cpu_milli=1_000, memory_mib=1_024)
    unknown_decision = place(requirement, (snapshot(memory_state="unknown"),), now=NOW)
    assert unknown_decision.status == "denied"
    assert "unknown_memory" in unknown_decision.denial_reasons

    unverified_decision = place(
        requirement,
        (snapshot(attestation_status="unverified"),),
        now=NOW,
    )
    assert unverified_decision.status == "denied"
    assert "unverified_attestation" in unverified_decision.denial_reasons


def test_stale_snapshot_and_forged_subject_are_rejected() -> None:
    requirement = PlacementRequirement(cpu_milli=1_000, memory_mib=1_024)
    stale = place(
        requirement,
        (snapshot(),),
        now=NOW + timedelta(hours=2),
    )
    assert stale.status == "denied"
    assert "stale_snapshot" in stale.denial_reasons

    base = snapshot()
    with pytest.raises(ValidationError, match="subject digest"):
        ResourcePoolSnapshot(
            pool_ref=base.pool_ref,
            revision=2,
            observed_at=base.observed_at,
            expires_at=base.expires_at,
            capabilities=base.capabilities,
            attestation=CapabilityAttestation(
                status=base.attestation.status,
                algorithm=base.attestation.algorithm,
                signer_ref=base.attestation.signer_ref,
                signature_ref=base.attestation.signature_ref,
                subject_digest="sha256:" + "f" * 64,
                observed_at=base.attestation.observed_at,
                expires_at=base.attestation.expires_at,
            ),
        )


def test_accounting_never_allows_used_above_reserved_or_allocatable() -> None:
    with pytest.raises(ValidationError, match="used <= reserved <= allocatable"):
        ResourceAccounting.known(allocatable=8_000, reserved=7_000, used=7_001)
