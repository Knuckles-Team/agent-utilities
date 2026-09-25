"""Bound control-plane access to epistemic-graph's native capacity leases.

The composition root supplies the authenticated tenant and owner digest once.
Callers choose a capacity cell and demand, but cannot change that identity on
acquire, renewal, or release. EG owns the durable lease and fencing decisions.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Mapping
from typing import Any


class CapacityPortUnavailable(RuntimeError):
    """The synchronous EG CapacityLease namespace is unavailable."""


class BoundCapacityLeasePort:
    """Synchronous, fail-closed port for one verified WorkItem owner."""

    def __init__(
        self, client: Any, *, tenant_ref: str, owner_digest: str, work_item_id: str
    ) -> None:
        for field, value in (
            ("tenant_ref", tenant_ref),
            ("owner_digest", owner_digest),
            ("work_item_id", work_item_id),
        ):
            if not isinstance(value, str) or not value or len(value.encode()) > 512:
                raise ValueError(f"{field} must be a bounded nonempty string")
        namespace = getattr(client, "capacity_leases", None)
        if namespace is None:
            raise CapacityPortUnavailable("EG capacity_leases namespace is missing")
        self._namespace = namespace
        self.tenant_ref = tenant_ref
        self.owner_digest = owner_digest
        self.work_item_id = work_item_id

    def _call(self, name: str, request: dict[str, Any]) -> dict[str, Any]:
        operation = getattr(self._namespace, name, None)
        if not callable(operation) or inspect.iscoroutinefunction(operation):
            raise CapacityPortUnavailable(f"EG capacity_leases.{name} is unavailable")
        response = operation(request)
        if inspect.isawaitable(response):
            close = getattr(response, "close", None)
            if callable(close):
                close()
            raise CapacityPortUnavailable(
                "EG capacity lease port requires a sync client"
            )
        if not isinstance(response, Mapping) or response.get("schema_version") != "1":
            raise CapacityPortUnavailable("EG capacity lease response is malformed")
        return dict(response)

    def _lease(self, lease: Mapping[str, Any]) -> dict[str, Any]:
        if (
            lease.get("tenant_ref") != self.tenant_ref
            or lease.get("actor_digest") != self.owner_digest
            or lease.get("work_item_id") != self.work_item_id
        ):
            raise CapacityPortUnavailable("EG capacity lease owner binding changed")
        for field in ("lease_id", "cell_id"):
            if not isinstance(lease.get(field), str) or not lease[field]:
                raise CapacityPortUnavailable(f"EG capacity lease {field} is missing")
        for field in ("lease_epoch", "fence_token"):
            value = lease.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CapacityPortUnavailable(f"EG capacity lease {field} is invalid")
        return dict(lease)

    def acquire(
        self,
        *,
        cell_id: str,
        resource_class: str,
        amount: int,
        idempotency_key: str,
        priority: str = "orchestration",
        ttl_ms: int = 30_000,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Acquire one native lease; return its server-issued fence and binding."""
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("cell_id must be nonempty")
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotency_key must be nonempty")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 1:
            raise ValueError("amount must be positive")
        request = {
            "schema_version": "1",
            "tenant_ref": self.tenant_ref,
            "work_item_id": self.work_item_id,
            "owner_digest": self.owner_digest,
            "idempotency_key": idempotency_key,
            "priority": priority,
            "demands": [
                {"cell_id": cell_id, "resource_class": resource_class, "amount": amount}
            ],
            "lease_id": None,
            "ttl_ms": ttl_ms,
            "now_ms": int(time.time() * 1000) if now_ms is None else now_ms,
            "cost_budget_micros": None,
            "token_budget": None,
        }
        answer = self._call("acquire", request)
        if answer.get("decision") not in {"accepted", "replayed"}:
            return answer
        leases = answer.get("leases")
        if not isinstance(leases, list) or len(leases) != 1:
            raise CapacityPortUnavailable("EG capacity acquisition omitted its lease")
        lease = self._lease(leases[0])
        if (
            lease.get("cell_id") != cell_id
            or lease.get("resource_class") != resource_class
            or lease.get("amount") != amount
            or lease.get("idempotency_key") != idempotency_key
        ):
            raise CapacityPortUnavailable("EG capacity acquisition changed its demand")
        return answer

    def _mutate(
        self, operation: str, lease: Mapping[str, Any], *, ttl_ms: int | None = None
    ) -> dict[str, Any]:
        held = self._lease(lease)
        return self._call(
            operation,
            {
                "schema_version": "1",
                "tenant_ref": self.tenant_ref,
                "owner_digest": self.owner_digest,
                "leases": [
                    {
                        "lease_id": held["lease_id"],
                        "lease_epoch": held["lease_epoch"],
                        "fence_token": held["fence_token"],
                    }
                ],
                "now_ms": int(time.time() * 1000),
                "ttl_ms": ttl_ms,
                "idempotency_key": None,
            },
        )

    def renew(self, lease: Mapping[str, Any], *, ttl_ms: int) -> dict[str, Any]:
        """Renew using the engine-issued fence and fixed owner identity."""
        if (
            isinstance(ttl_ms, bool)
            or not isinstance(ttl_ms, int)
            or not 1 <= ttl_ms <= 86_400_000
        ):
            raise ValueError("ttl_ms must be within 1..86400000")
        return self._mutate("renew", lease, ttl_ms=ttl_ms)

    def release(self, lease: Mapping[str, Any]) -> dict[str, Any]:
        """Release using the engine-issued fence and fixed owner identity."""
        return self._mutate("release", lease)
