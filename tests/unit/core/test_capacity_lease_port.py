"""The AU capacity port must preserve EG's owner and fence binding."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.core.capacity_lease_port import (
    BoundCapacityLeasePort,
    CapacityPortUnavailable,
)


def _lease() -> dict[str, object]:
    return {
        "lease_id": "lease-1",
        "cell_id": "cell-1",
        "resource_class": "worker",
        "amount": 1,
        "idempotency_key": "request-1",
        "tenant_ref": "tenant-a",
        "actor_digest": "actor-a",
        "work_item_id": "work-1",
        "lease_epoch": 3,
        "fence_token": 7,
    }


class _Namespace:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.lease = _lease()

    def acquire(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("acquire", request))
        return {
            "schema_version": "1",
            "decision": "accepted",
            "leases": [self.lease],
        }

    def renew(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("renew", request))
        return {"schema_version": "1", "decision": "renewed", "leases": [self.lease]}

    def release(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(("release", request))
        return {"schema_version": "1", "decision": "released", "leases": []}


def _port(namespace: _Namespace) -> BoundCapacityLeasePort:
    return BoundCapacityLeasePort(
        SimpleNamespace(capacity_leases=namespace),
        tenant_ref="tenant-a",
        owner_digest="actor-a",
        work_item_id="work-1",
    )


def test_acquire_renew_release_use_one_bound_identity_and_native_fence() -> None:
    namespace = _Namespace()
    port = _port(namespace)
    acquired = port.acquire(
        cell_id="cell-1",
        resource_class="worker",
        amount=1,
        idempotency_key="request-1",
        now_ms=10,
    )
    lease = acquired["leases"][0]
    port.renew(lease, ttl_ms=20)
    port.release(lease)

    acquire_request = namespace.calls[0][1]
    assert acquire_request["tenant_ref"] == "tenant-a"
    assert acquire_request["owner_digest"] == "actor-a"
    assert acquire_request["work_item_id"] == "work-1"
    assert acquire_request["demands"] == [
        {"cell_id": "cell-1", "resource_class": "worker", "amount": 1}
    ]
    for _, request in namespace.calls[1:]:
        assert request["tenant_ref"] == "tenant-a"
        assert request["owner_digest"] == "actor-a"
        assert request["leases"] == [
            {"lease_id": "lease-1", "lease_epoch": 3, "fence_token": 7}
        ]


def test_cross_owner_and_stale_fence_are_rejected_before_mutation() -> None:
    namespace = _Namespace()
    port = _port(namespace)
    forged = _lease()
    forged["actor_digest"] = "other-actor"
    with pytest.raises(CapacityPortUnavailable, match="owner binding"):
        port.release(forged)
    forged = _lease()
    forged["fence_token"] = 0
    with pytest.raises(CapacityPortUnavailable, match="fence_token"):
        port.renew(forged, ttl_ms=100)
    assert namespace.calls == []


def test_acquire_fails_closed_on_changed_demand_or_missing_namespace() -> None:
    namespace = _Namespace()
    namespace.lease["cell_id"] = "another-cell"
    with pytest.raises(CapacityPortUnavailable, match="changed its demand"):
        _port(namespace).acquire(
            cell_id="cell-1",
            resource_class="worker",
            amount=1,
            idempotency_key="request-1",
        )
    with pytest.raises(CapacityPortUnavailable, match="namespace"):
        BoundCapacityLeasePort(
            SimpleNamespace(),
            tenant_ref="tenant-a",
            owner_digest="actor-a",
            work_item_id="work-1",
        )
