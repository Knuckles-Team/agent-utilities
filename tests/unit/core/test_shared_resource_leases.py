"""NE-183 shared-resource lease acceptance fixtures.

These are deliberately bounded contract fixtures.  They exercise a durable
SQLite reference authority so two independent authority objects (and, in the
cross-process case, two workers) contend on one persisted cell.  Production
deployment uses the engine-native replicated authority instead.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from agent_utilities.core.shared_resource_leases import (
    LeaseDenied,
    LeaseScopeMismatch,
    ResourceCell,
    ResourceLeaseRequest,
    SQLiteResourceLeaseAuthority,
    StaleLeaseEpoch,
    StaleLeaseFence,
)


def _cell(
    *, device: str = "GPU-uuid/MIG-1g.5gb", epoch: int = 1, capacity: int = 1
) -> ResourceCell:
    return ResourceCell(
        resource_kind="gpu_concurrency",
        resource_id="gpu-group-a",
        node_id="node-a",
        device_id=device,
        capacity=capacity,
        epoch=epoch,
        reserved_floor=0,
        policy_digest="policy-v1",
        tenant_quotas=(("tenant-a", 1), ("tenant-b", 1)),
    )


def _request(
    *,
    tenant: str = "tenant-a",
    principal: str = "worker-a",
    key: str = "request-a",
    device: str = "GPU-uuid/MIG-1g.5gb",
    epoch: int = 1,
    priority: str = "interactive",
    policy_digest: str = "policy-v1",
) -> ResourceLeaseRequest:
    return ResourceLeaseRequest(
        resource_kind="gpu_concurrency",
        resource_id="gpu-group-a",
        node_id="node-a",
        device_id=device,
        tenant_ref=tenant,
        principal_ref=principal,
        amount=1,
        idempotency_key=key,
        lease_epoch=epoch,
        ttl_ms=100,
        priority_class=priority,
        policy_digest=policy_digest,
    )


def _cross_process_contender(path: str, output: object) -> None:
    """Child-side contender used only by the bounded two-process fixture."""

    authority = SQLiteResourceLeaseAuthority(path)
    try:
        authority.acquire(
            _request(tenant="tenant-b", principal="worker-b", key="child"), now_ms=10
        )
    except LeaseDenied:
        # A full cell held by the first process must deny the second process.
        output.put("denied")
    else:
        output.put("admitted")


def test_two_process_contenders_share_one_durable_cell(tmp_path: Path) -> None:
    path = str(tmp_path / "leases.sqlite")
    first = SQLiteResourceLeaseAuthority(path, cells=(_cell(),))
    lease = first.acquire(_request(), now_ms=1)

    # Separate authority objects represent independent worker processes.  The
    # persisted transaction, not a Python lock, owns the contention decision.
    second = SQLiteResourceLeaseAuthority(path)
    with pytest.raises(LeaseDenied):
        second.acquire(
            _request(tenant="tenant-b", principal="worker-b", key="second"), now_ms=2
        )

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_cross_process_contender, args=(path, queue)
    )
    process.start()
    process.join(timeout=5)
    assert process.exitcode == 0
    assert queue.get(timeout=1) == "denied"

    first.release(
        lease.lease_id,
        tenant_ref="tenant-a",
        principal_ref="worker-a",
        fence_token=lease.fence_token,
        lease_epoch=lease.lease_epoch,
        now_ms=3,
    )
    admitted = second.acquire(
        _request(tenant="tenant-b", principal="worker-b", key="after-release"),
        now_ms=4,
    )
    assert admitted.request.tenant_ref == "tenant-b"


def test_crashed_holder_expires_and_is_reclaimed(tmp_path: Path) -> None:
    path = str(tmp_path / "leases.sqlite")
    first = SQLiteResourceLeaseAuthority(path, cells=(_cell(),))
    second = SQLiteResourceLeaseAuthority(path)
    first.acquire(_request(key="crashed"), now_ms=1)
    with pytest.raises(LeaseDenied):
        second.acquire(
            _request(tenant="tenant-b", principal="worker-b", key="wait"), now_ms=50
        )
    assert second.reclaim_expired(now_ms=101) == ("gpu_concurrency:gpu-group-a:1",)
    recovered = second.acquire(
        _request(tenant="tenant-b", principal="worker-b", key="recovered"),
        now_ms=102,
    )
    assert recovered.request.tenant_ref == "tenant-b"


def test_mig_device_change_fences_old_epoch(tmp_path: Path) -> None:
    path = str(tmp_path / "leases.sqlite")
    authority = SQLiteResourceLeaseAuthority(path, cells=(_cell(),))
    old = authority.acquire(_request(key="old"), now_ms=1)
    authority.rebind_device(
        resource_kind="gpu_concurrency",
        resource_id="gpu-group-a",
        node_id="node-a",
        device_id="GPU-uuid/MIG-2g.10gb",
        epoch=2,
        policy_digest="policy-v2",
    )
    with pytest.raises(StaleLeaseEpoch):
        authority.renew(
            old.lease_id,
            tenant_ref="tenant-a",
            principal_ref="worker-a",
            fence_token=old.fence_token,
            lease_epoch=1,
            ttl_ms=100,
            now_ms=2,
        )
    # ``rebind_device`` fenced the cell onto ``policy-v2`` along with the new
    # epoch/device (``SQLiteResourceLeaseAuthority.rebind_device`` --
    # ``_validate_cell_scope`` requires an exact match on ALL THREE: device,
    # epoch, AND policy digest). A request still carrying the pre-rebind
    # digest is exactly the "stale policy assumption" this scope check exists
    # to deny, so the new acquire must present the current digest.
    new = authority.acquire(
        _request(
            key="new-device",
            device="GPU-uuid/MIG-2g.10gb",
            epoch=2,
            policy_digest="policy-v2",
        ),
        now_ms=3,
    )
    assert new.request.device_id.endswith("MIG-2g.10gb")


def test_tenant_quota_and_reserved_floor_preserve_fairness(tmp_path: Path) -> None:
    path = str(tmp_path / "leases.sqlite")
    cell = ResourceCell(
        resource_kind="kv_cache_slots",
        resource_id="kv-a",
        node_id="node-a",
        device_id="kv-service",
        capacity=3,
        reserved_floor=1,
        tenant_quotas=(("tenant-a", 1), ("tenant-b", 2)),
    )
    authority = SQLiteResourceLeaseAuthority(path, cells=(cell,))

    def request(
        tenant: str, key: str, priority: str = "background"
    ) -> ResourceLeaseRequest:
        return ResourceLeaseRequest(
            resource_kind="kv_cache_slots",
            resource_id="kv-a",
            node_id="node-a",
            device_id="kv-service",
            tenant_ref=tenant,
            principal_ref=f"worker-{tenant}",
            amount=1,
            idempotency_key=key,
            ttl_ms=100,
            priority_class=priority,
            policy_digest="policy:default",
        )

    # Background tenants share only the spare two slots; tenant-a cannot take
    # a second slot even though tenant-b has not used its quota.
    authority.register_cell(cell)
    authority.acquire(request("tenant-a", "a-1"), now_ms=1)
    authority.acquire(request("tenant-b", "b-1"), now_ms=1)
    with pytest.raises(LeaseDenied):
        authority.acquire(request("tenant-a", "a-2"), now_ms=1)
    # The reserved interactive floor remains available during the background flood.
    interactive = authority.acquire(
        request("tenant-b", "interactive", "interactive"), now_ms=1
    )
    assert interactive.request.priority_class == "interactive"


def test_forged_owner_fence_and_key_are_denied(tmp_path: Path) -> None:
    path = str(tmp_path / "leases.sqlite")
    # capacity=2 (vs. the shared ``_cell()`` default of 1): this test's last
    # assertion exercises idempotency-key isolation across tenants, not
    # capacity denial -- tenant-a's lease is still legitimately active
    # throughout (its forged release/renew attempts are correctly denied and
    # never mutate state), so a capacity-1 cell would deny tenant-b's later
    # acquire for a reason unrelated to what this test is checking. Both
    # tenant quotas (1 each) still cap any single tenant at its own share.
    authority = SQLiteResourceLeaseAuthority(path, cells=(_cell(capacity=2),))
    lease = authority.acquire(_request(key="original"), now_ms=1)
    with pytest.raises(LeaseScopeMismatch):
        authority.release(
            lease.lease_id,
            tenant_ref="tenant-b",
            principal_ref="worker-b",
            fence_token=lease.fence_token,
            lease_epoch=lease.lease_epoch,
            now_ms=2,
        )
    with pytest.raises(StaleLeaseFence):
        authority.renew(
            lease.lease_id,
            tenant_ref="tenant-a",
            principal_ref="worker-a",
            fence_token=lease.fence_token + 1,
            lease_epoch=lease.lease_epoch,
            ttl_ms=100,
            now_ms=2,
        )
    # A tenant cannot replay the other tenant's idempotency key as the same
    # lease: tenant identity is part of both the key and the request digest.
    other = authority.acquire(
        _request(tenant="tenant-b", principal="worker-b", key="original"),
        now_ms=2,
    )
    assert other.lease_id != lease.lease_id
