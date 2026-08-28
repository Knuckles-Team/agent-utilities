"""Shared, fenced resource leases for distributed capacity admission.

This module is the client-side contract for resources that are shared by more
than one process: accelerator concurrency and memory, KV-cache slots, model
concurrency, and token budgets.  A local semaphore or a module-global counter
can still be useful as a *hint*, but it is not an admission authority.  Real
callers must use :class:`EngineNativeResourceLeaseAuthority`, which delegates
the compare-and-swap, expiry, and fencing decision to the epistemic-graph
engine.

``SQLiteResourceLeaseAuthority`` is intentionally explicit and small.  It is a
durable reference/test authority for a single host (and is useful for a
non-engine development profile); production distributed deployments must use
the native transport.  Keeping this adapter here gives the cross-process
contention and crash-expiry contract a real persistence boundary without
silently making a process-local dictionary authoritative.

The binding is deliberately stronger than a resource name.  Every request and
lease carries the exact resource class/id, node, physical device identity
(including a MIG UUID/profile when present), authenticated tenant, principal,
authority epoch, policy digest, and a caller idempotency key.  Renewal and
release require both the owner identity and the current fence token.  A stale
or forged holder therefore cannot continue after failover, device replacement,
or lease expiry.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "RESOURCE_LEASE_SCHEMA",
    "RESOURCE_KINDS",
    "LeaseAuthorityUnavailable",
    "LeaseDenied",
    "LeaseExpired",
    "LeaseNotFound",
    "StaleLeaseFence",
    "StaleLeaseEpoch",
    "LeaseScopeMismatch",
    "LeaseIdempotencyConflict",
    "ResourceCell",
    "ResourceLeaseRequest",
    "ResourceLease",
    "SharedResourceLeaseAuthority",
    "InMemoryResourceLeaseAuthority",
    "SQLiteResourceLeaseAuthority",
    "EngineNativeResourceLeaseAuthority",
    "lease_scope_digest",
    "hold_resource_lease",
]

RESOURCE_LEASE_SCHEMA = "resource-lease.v1"
RESOURCE_KINDS = frozenset(
    {
        "gpu_concurrency",
        "gpu_memory_bytes",
        "kv_cache_slots",
        "model_concurrency",
        "token_budget",
    }
)
_ACTIVE_STATES = frozenset({"active", "renewed"})
_TERMINAL_STATES = frozenset({"released", "expired", "reclaimed", "stale"})
_MAX_TTL_MS = 24 * 60 * 60 * 1000


def _text(value: object, name: str) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        raise ValueError(f"{name} must be non-empty")
    if any(ord(char) < 32 or ord(char) == 127 for char in rendered):
        raise ValueError(f"{name} contains a control character")
    return rendered


def _positive_int(value: object, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds its bound")
    return value


def _now_ms() -> int:
    return int(time.time() * 1000)


class LeaseAuthorityUnavailable(RuntimeError):
    """The durable shared-resource authority is not available."""

    code = "shared_resource_authority_unavailable"


class LeaseDenied(RuntimeError):
    """The authoritative resource policy denied admission."""

    code = "shared_resource_lease_denied"


class LeaseExpired(LeaseDenied):
    code = "shared_resource_lease_expired"


class LeaseNotFound(LeaseDenied):
    code = "shared_resource_lease_not_found"


class StaleLeaseFence(LeaseDenied):
    code = "shared_resource_stale_fence"


class StaleLeaseEpoch(LeaseDenied):
    code = "shared_resource_stale_epoch"


class LeaseScopeMismatch(LeaseDenied):
    code = "shared_resource_scope_mismatch"


class LeaseIdempotencyConflict(LeaseDenied):
    code = "shared_resource_idempotency_conflict"


@dataclass(frozen=True, slots=True)
class ResourceCell:
    """One authoritative capacity cell and its physical binding."""

    resource_kind: str
    resource_id: str
    node_id: str
    device_id: str
    capacity: int
    epoch: int = 1
    reserved_floor: int = 0
    policy_digest: str = "policy:default"
    tenant_quotas: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        kind = _text(self.resource_kind, "resource_kind")
        if kind not in RESOURCE_KINDS:
            raise ValueError(f"unsupported resource_kind: {kind!r}")
        object.__setattr__(self, "resource_kind", kind)
        for name in ("resource_id", "node_id", "device_id", "policy_digest"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        _positive_int(self.capacity, "capacity")
        _positive_int(self.epoch, "epoch")
        if isinstance(self.reserved_floor, bool) or not isinstance(
            self.reserved_floor, int
        ):
            raise ValueError("reserved_floor must be an integer")
        if not 0 <= self.reserved_floor <= self.capacity:
            raise ValueError("reserved_floor must be within capacity")
        seen: set[str] = set()
        normalized: list[tuple[str, int]] = []
        for tenant, quota in self.tenant_quotas:
            tenant = _text(tenant, "tenant quota key")
            if tenant in seen:
                raise ValueError("tenant quotas must not contain duplicates")
            seen.add(tenant)
            normalized.append((tenant, _positive_int(quota, "tenant quota")))
        object.__setattr__(self, "tenant_quotas", tuple(sorted(normalized)))

    @property
    def logical_key(self) -> str:
        return ":".join((self.resource_kind, self.resource_id, self.node_id))

    @property
    def key(self) -> str:
        return ":".join((self.logical_key, self.device_id))

    def quota_for(self, tenant_ref: str) -> int | None:
        for tenant, quota in self.tenant_quotas:
            if tenant == tenant_ref:
                return quota
        return None


@dataclass(frozen=True, slots=True)
class ResourceLeaseRequest:
    """Immutable, authenticated request for one shared-resource lease."""

    resource_kind: str
    resource_id: str
    node_id: str
    device_id: str
    tenant_ref: str
    principal_ref: str
    amount: int
    idempotency_key: str
    lease_epoch: int = 1
    ttl_ms: int = 30_000
    priority_class: str = "interactive"
    policy_digest: str = "policy:default"
    schema: str = RESOURCE_LEASE_SCHEMA

    def __post_init__(self) -> None:
        kind = _text(self.resource_kind, "resource_kind")
        if kind not in RESOURCE_KINDS:
            raise ValueError(f"unsupported resource_kind: {kind!r}")
        object.__setattr__(self, "resource_kind", kind)
        for name in (
            "resource_id",
            "node_id",
            "device_id",
            "tenant_ref",
            "principal_ref",
            "idempotency_key",
            "policy_digest",
            "schema",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.tenant_ref.casefold() in {"anonymous", "unknown", "*"}:
            raise ValueError("tenant_ref must identify an authenticated tenant")
        if self.principal_ref.casefold() in {"anonymous", "unknown", "*"}:
            raise ValueError("principal_ref must identify an authenticated principal")
        _positive_int(self.amount, "amount")
        _positive_int(self.lease_epoch, "lease_epoch")
        _positive_int(self.ttl_ms, "ttl_ms", maximum=_MAX_TTL_MS)
        priority = _text(self.priority_class, "priority_class").casefold()
        if priority not in {"interactive", "foreground", "background", "best_effort"}:
            raise ValueError("priority_class is not recognized")
        object.__setattr__(self, "priority_class", priority)

    @property
    def cell_logical_key(self) -> str:
        return ":".join((self.resource_kind, self.resource_id, self.node_id))

    @property
    def cell_key(self) -> str:
        return ":".join((self.cell_logical_key, self.device_id))

    @property
    def digest(self) -> str:
        return lease_scope_digest(self)

    def _canonical_dict(self) -> dict[str, object]:
        """Fields covered by the request digest (excluding the digest itself)."""
        return {
            "schema": self.schema,
            "resource_kind": self.resource_kind,
            "resource_id": self.resource_id,
            "node_id": self.node_id,
            "device_id": self.device_id,
            "tenant_ref": self.tenant_ref,
            "principal_ref": self.principal_ref,
            "amount": self.amount,
            "idempotency_key": self.idempotency_key,
            "lease_epoch": self.lease_epoch,
            "ttl_ms": self.ttl_ms,
            "priority_class": self.priority_class,
            "policy_digest": self.policy_digest,
        }

    def as_dict(self) -> dict[str, object]:
        return {**self._canonical_dict(), "request_digest": self.digest}


@dataclass(frozen=True, slots=True)
class ResourceLease:
    """Authoritative grant returned by a shared-resource authority."""

    lease_id: str
    request: ResourceLeaseRequest
    fence_token: int
    issued_at_ms: int
    expires_at_ms: int
    state: str = "active"
    authority_revision: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        _positive_int(self.fence_token, "fence_token")
        if (
            isinstance(self.issued_at_ms, bool)
            or not isinstance(self.issued_at_ms, int)
            or self.issued_at_ms < 0
        ):
            raise ValueError("issued_at_ms must be a non-negative integer")
        if self.expires_at_ms <= self.issued_at_ms:
            raise ValueError("expires_at_ms must be after issued_at_ms")
        if self.state not in _ACTIVE_STATES | _TERMINAL_STATES:
            raise ValueError("unknown resource lease state")

    @property
    def lease_epoch(self) -> int:
        return self.request.lease_epoch

    @property
    def request_digest(self) -> str:
        return self.request.digest

    def is_expired(self, now_ms: int | None = None) -> bool:
        now = _now_ms() if now_ms is None else int(now_ms)
        return self.state not in _ACTIVE_STATES or now >= self.expires_at_ms

    def assert_live(self, *, now_ms: int | None = None) -> None:
        if self.is_expired(now_ms):
            raise LeaseExpired(self.lease_id)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.request.schema,
            "lease_id": self.lease_id,
            "request": self.request.as_dict(),
            "fence_token": self.fence_token,
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "state": self.state,
            "authority_revision": self.authority_revision,
        }


def lease_scope_digest(request: ResourceLeaseRequest) -> str:
    """Hash the exact scope/budget, never secrets or lease body contents."""

    canonical = json.dumps(
        request._canonical_dict(), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@runtime_checkable
class SharedResourceLeaseAuthority(Protocol):
    """The sole admission interface for shared capacity."""

    def acquire(
        self, request: ResourceLeaseRequest, *, now_ms: int | None = None
    ) -> ResourceLease: ...

    def renew(
        self,
        lease_id: str,
        *,
        tenant_ref: str,
        principal_ref: str,
        fence_token: int,
        lease_epoch: int,
        ttl_ms: int,
        now_ms: int | None = None,
    ) -> ResourceLease: ...

    def release(
        self,
        lease_id: str,
        *,
        tenant_ref: str,
        principal_ref: str,
        fence_token: int,
        lease_epoch: int,
        now_ms: int | None = None,
    ) -> None: ...

    def reclaim_expired(self, *, now_ms: int | None = None) -> tuple[str, ...]: ...


@dataclass
class _MemoryRecord:
    lease: ResourceLease


class InMemoryResourceLeaseAuthority:
    """Thread-safe reference authority for isolated unit tests only.

    It intentionally requires explicit cell registration and never becomes a
    default in production.  For cross-process behavior use the SQLite adapter
    in a development profile or the native engine transport.
    """

    test_only = True

    def __init__(self, cells: Sequence[ResourceCell] = ()) -> None:
        self._lock = threading.RLock()
        self._cells: dict[str, ResourceCell] = {
            cell.logical_key: cell for cell in cells
        }
        self._leases: dict[str, _MemoryRecord] = {}
        self._idempotency: dict[tuple[str, str, str], str] = {}
        self._next_fence = 1
        self._revision = 0

    def register_cell(self, cell: ResourceCell) -> None:
        with self._lock:
            prior = self._cells.get(cell.logical_key)
            if prior is not None and cell.epoch < prior.epoch:
                raise StaleLeaseEpoch("cell epoch moved backwards")
            self._cells[cell.logical_key] = cell

    def _cell_for(self, request: ResourceLeaseRequest) -> ResourceCell:
        cell = self._cells.get(request.cell_logical_key)
        if cell is None:
            raise LeaseDenied("unknown shared resource cell")
        if cell.device_id != request.device_id:
            raise LeaseScopeMismatch("resource device/MIG identity changed")
        if cell.epoch != request.lease_epoch:
            raise StaleLeaseEpoch("resource authority epoch is stale")
        if cell.policy_digest != request.policy_digest:
            raise LeaseScopeMismatch("resource policy digest does not match")
        return cell

    def _reclaim_locked(self, now_ms: int) -> tuple[str, ...]:
        reclaimed: list[str] = []
        for lease_id, record in self._leases.items():
            if (
                record.lease.state in _ACTIVE_STATES
                and now_ms >= record.lease.expires_at_ms
            ):
                record.lease = replace(record.lease, state="reclaimed")
                reclaimed.append(lease_id)
        return tuple(reclaimed)

    def _active_amount_locked(self, logical_key: str, now_ms: int) -> int:
        return sum(
            record.lease.request.amount
            for record in self._leases.values()
            if record.lease.request.cell_logical_key == logical_key
            and record.lease.state in _ACTIVE_STATES
            and record.lease.expires_at_ms > now_ms
        )

    def _idempotent_replay_locked(
        self, idem: tuple[str, str, str], request: ResourceLeaseRequest, now: int
    ) -> ResourceLease | None:
        """Return the prior lease for ``idem`` if one exists, else ``None``.

        Extracted from :meth:`acquire`: the idempotency-replay branch (digest
        check + liveness assertion) is a single reviewable unit, separate from
        capacity admission.
        """
        prior_id = self._idempotency.get(idem)
        if prior_id is None:
            return None
        prior = self._leases[prior_id].lease
        if prior.request.digest != request.digest:
            raise LeaseIdempotencyConflict("idempotency key changed request scope")
        prior.assert_live(now_ms=now)
        return prior

    def _tenant_active_amount_locked(
        self, logical_key: str, tenant_ref: str, now_ms: int
    ) -> int:
        return sum(
            record.lease.request.amount
            for record in self._leases.values()
            if record.lease.request.cell_logical_key == logical_key
            and record.lease.request.tenant_ref == tenant_ref
            and record.lease.state in _ACTIVE_STATES
            and record.lease.expires_at_ms > now_ms
        )

    def _assert_within_quota_locked(
        self, cell: ResourceCell, request: ResourceLeaseRequest, now: int
    ) -> None:
        quota = cell.quota_for(request.tenant_ref)
        if quota is None:
            return
        tenant_active = self._tenant_active_amount_locked(
            request.cell_logical_key, request.tenant_ref, now
        )
        if tenant_active + request.amount > quota:
            raise LeaseDenied("tenant resource quota exhausted")

    @staticmethod
    def _capacity_ceiling(cell: ResourceCell, request: ResourceLeaseRequest) -> int:
        ceiling = cell.capacity
        if request.priority_class in {"background", "best_effort"}:
            ceiling = max(0, ceiling - cell.reserved_floor)
        return ceiling

    def _issue_lease_locked(
        self, request: ResourceLeaseRequest, *, now: int
    ) -> ResourceLease:
        lease_id = f"{request.resource_kind}:{request.resource_id}:{self._next_fence}"
        fence = self._next_fence
        self._next_fence += 1
        self._revision += 1
        lease = ResourceLease(
            lease_id=lease_id,
            request=request,
            fence_token=fence,
            issued_at_ms=now,
            expires_at_ms=now + request.ttl_ms,
            authority_revision=self._revision,
        )
        self._leases[lease_id] = _MemoryRecord(lease)
        return lease

    def acquire(
        self, request: ResourceLeaseRequest, *, now_ms: int | None = None
    ) -> ResourceLease:
        now = _now_ms() if now_ms is None else int(now_ms)
        with self._lock:
            self._reclaim_locked(now)
            cell = self._cell_for(request)
            idem = (
                request.cell_logical_key,
                request.tenant_ref,
                request.idempotency_key,
            )
            replay = self._idempotent_replay_locked(idem, request, now)
            if replay is not None:
                return replay
            active = self._active_amount_locked(request.cell_logical_key, now)
            self._assert_within_quota_locked(cell, request, now)
            ceiling = self._capacity_ceiling(cell, request)
            available = ceiling - active
            if available < request.amount:
                raise LeaseDenied(
                    f"shared resource capacity exhausted ({available} available)"
                )
            lease = self._issue_lease_locked(request, now=now)
            self._idempotency[idem] = lease.lease_id
            return lease

    def _owner_locked(
        self,
        lease_id: str,
        *,
        tenant_ref: str,
        principal_ref: str,
        fence_token: int,
        lease_epoch: int,
        now_ms: int,
    ) -> _MemoryRecord:
        record = self._leases.get(lease_id)
        if record is None:
            raise LeaseNotFound(lease_id)
        lease = record.lease
        if (
            lease.request.tenant_ref != tenant_ref
            or lease.request.principal_ref != principal_ref
        ):
            raise LeaseScopeMismatch("lease owner identity does not match")
        if lease.request.lease_epoch != lease_epoch:
            raise StaleLeaseEpoch("lease epoch is stale")
        cell = self._cells.get(lease.request.cell_logical_key)
        if (
            cell is None
            or cell.device_id != lease.request.device_id
            or cell.epoch != lease_epoch
        ):
            raise StaleLeaseEpoch("resource device or epoch is stale")
        if lease.fence_token != fence_token:
            raise StaleLeaseFence("lease fence token does not match")
        if lease.is_expired(now_ms):
            record.lease = replace(lease, state="expired")
            raise LeaseExpired(lease_id)
        return record

    def renew(
        self,
        lease_id: str,
        *,
        tenant_ref: str,
        principal_ref: str,
        fence_token: int,
        lease_epoch: int,
        ttl_ms: int,
        now_ms: int | None = None,
    ) -> ResourceLease:
        now = _now_ms() if now_ms is None else int(now_ms)
        _positive_int(ttl_ms, "ttl_ms", maximum=_MAX_TTL_MS)
        with self._lock:
            record = self._owner_locked(
                lease_id,
                tenant_ref=tenant_ref,
                principal_ref=principal_ref,
                fence_token=fence_token,
                lease_epoch=lease_epoch,
                now_ms=now,
            )
            self._revision += 1
            record.lease = replace(
                record.lease,
                expires_at_ms=now + ttl_ms,
                state="renewed",
                authority_revision=self._revision,
            )
            return record.lease

    def release(
        self,
        lease_id: str,
        *,
        tenant_ref: str,
        principal_ref: str,
        fence_token: int,
        lease_epoch: int,
        now_ms: int | None = None,
    ) -> None:
        now = _now_ms() if now_ms is None else int(now_ms)
        with self._lock:
            record = self._leases.get(lease_id)
            if record is None:
                raise LeaseNotFound(lease_id)
            if (
                record.lease.request.tenant_ref != tenant_ref
                or record.lease.request.principal_ref != principal_ref
            ):
                raise LeaseScopeMismatch("lease owner identity does not match")
            if record.lease.fence_token != fence_token:
                raise StaleLeaseFence("lease fence token does not match")
            if record.lease.request.lease_epoch != lease_epoch:
                raise StaleLeaseEpoch("lease epoch does not match")
            if record.lease.state in _TERMINAL_STATES:
                return
            self._owner_locked(
                lease_id,
                tenant_ref=tenant_ref,
                principal_ref=principal_ref,
                fence_token=fence_token,
                lease_epoch=lease_epoch,
                now_ms=now,
            )
            record.lease = replace(record.lease, state="released")
            self._revision += 1

    def reclaim_expired(self, *, now_ms: int | None = None) -> tuple[str, ...]:
        now = _now_ms() if now_ms is None else int(now_ms)
        with self._lock:
            return self._reclaim_locked(now)

    def rebind_device(
        self,
        *,
        resource_kind: str,
        resource_id: str,
        node_id: str,
        device_id: str,
        epoch: int,
        policy_digest: str,
    ) -> ResourceCell:
        """Atomically fence a device/MIG replacement and publish its new epoch."""

        with self._lock:
            logical = ":".join(
                (
                    _text(resource_kind, "resource_kind"),
                    _text(resource_id, "resource_id"),
                    _text(node_id, "node_id"),
                )
            )
            prior = self._cells.get(logical)
            if prior is None:
                raise LeaseDenied("cannot rebind an unknown resource cell")
            if epoch <= prior.epoch:
                raise StaleLeaseEpoch("device epoch must increase")
            for record in self._leases.values():
                if (
                    record.lease.request.cell_logical_key == logical
                    and record.lease.state in _ACTIVE_STATES
                ):
                    record.lease = replace(record.lease, state="stale")
            cell = replace(
                prior,
                device_id=_text(device_id, "device_id"),
                epoch=epoch,
                policy_digest=_text(policy_digest, "policy_digest"),
            )
            self._cells[logical] = cell
            self._revision += 1
            return cell

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "authority": "in_memory_test_only",
                "cells": len(self._cells),
                "leases": len(self._leases),
                "active": sum(
                    record.lease.state in _ACTIVE_STATES
                    for record in self._leases.values()
                ),
                "revision": self._revision,
            }


class SQLiteResourceLeaseAuthority:
    """Durable single-host reference authority with cross-process CAS.

    The database is only a development/reference authority.  A production
    cluster must point callers at the engine-native transport so lease state is
    replicated with the same authority as graph placement and WorkItems.
    """

    test_only = True

    def __init__(
        self, path: str | os.PathLike[str], cells: Sequence[ResourceCell] = ()
    ) -> None:
        self.path = os.fspath(path)
        if not self.path:
            raise ValueError("lease database path is required")
        parent = os.path.dirname(os.path.abspath(self.path))
        if not os.path.isdir(parent):
            raise ValueError("lease database parent must already exist")
        self._initialize(cells)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize(self, cells: Sequence[ResourceCell]) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS resource_cells (
                    logical_key TEXT PRIMARY KEY,
                    resource_kind TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    capacity INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    reserved_floor INTEGER NOT NULL,
                    policy_digest TEXT NOT NULL,
                    tenant_quotas TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS resource_leases (
                    lease_id TEXT PRIMARY KEY,
                    logical_key TEXT NOT NULL,
                    resource_kind TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    tenant_ref TEXT NOT NULL,
                    principal_ref TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    lease_epoch INTEGER NOT NULL,
                    ttl_ms INTEGER NOT NULL,
                    priority_class TEXT NOT NULL,
                    policy_digest TEXT NOT NULL,
                    request_digest TEXT NOT NULL,
                    fence_token INTEGER NOT NULL UNIQUE,
                    issued_at_ms INTEGER NOT NULL,
                    expires_at_ms INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    authority_revision INTEGER NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS resource_lease_idem
                    ON resource_leases(logical_key, tenant_ref, principal_ref, idempotency_key);
                CREATE TABLE IF NOT EXISTS resource_meta (
                    name TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO resource_meta(name, value) VALUES ('fence', 0), ('revision', 0);
                """
            )
            for cell in cells:
                self._upsert_cell(conn, cell)

    @staticmethod
    def _upsert_cell(conn: sqlite3.Connection, cell: ResourceCell) -> None:
        quotas = json.dumps(
            dict(cell.tenant_quotas), sort_keys=True, separators=(",", ":")
        )
        conn.execute(
            """INSERT INTO resource_cells(logical_key, resource_kind, resource_id, node_id,
                       device_id, capacity, epoch, reserved_floor, policy_digest, tenant_quotas)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(logical_key) DO UPDATE SET
                 resource_kind=excluded.resource_kind, resource_id=excluded.resource_id,
                 node_id=excluded.node_id, device_id=excluded.device_id,
                 capacity=excluded.capacity, epoch=excluded.epoch,
                 reserved_floor=excluded.reserved_floor, policy_digest=excluded.policy_digest,
                 tenant_quotas=excluded.tenant_quotas""",
            (
                cell.logical_key,
                cell.resource_kind,
                cell.resource_id,
                cell.node_id,
                cell.device_id,
                cell.capacity,
                cell.epoch,
                cell.reserved_floor,
                cell.policy_digest,
                quotas,
            ),
        )

    @staticmethod
    def _cell(row: sqlite3.Row | tuple[Any, ...]) -> ResourceCell:
        values = tuple(row)
        quotas = json.loads(values[9])
        if not isinstance(quotas, dict):
            raise LeaseDenied("stored tenant quota metadata is malformed")
        return ResourceCell(
            resource_kind=values[1],
            resource_id=values[2],
            node_id=values[3],
            device_id=values[4],
            capacity=int(values[5]),
            epoch=int(values[6]),
            reserved_floor=int(values[7]),
            policy_digest=values[8],
            tenant_quotas=tuple((str(k), int(v)) for k, v in quotas.items()),
        )

    @staticmethod
    def _lease(row: sqlite3.Row | tuple[Any, ...]) -> ResourceLease:
        values = tuple(row)
        request = ResourceLeaseRequest(
            resource_kind=values[2],
            resource_id=values[3],
            node_id=values[4],
            device_id=values[5],
            tenant_ref=values[6],
            principal_ref=values[7],
            amount=int(values[8]),
            idempotency_key=values[9],
            lease_epoch=int(values[10]),
            ttl_ms=int(values[11]),
            priority_class=values[12],
            policy_digest=values[13],
        )
        if request.digest != values[14]:
            raise LeaseDenied("stored lease request digest is invalid")
        return ResourceLease(
            lease_id=values[0],
            request=request,
            fence_token=int(values[15]),
            issued_at_ms=int(values[16]),
            expires_at_ms=int(values[17]),
            state=values[18],
            authority_revision=int(values[19]),
        )

    @staticmethod
    def _next(conn: sqlite3.Connection, name: str) -> int:
        conn.execute(
            "UPDATE resource_meta SET value = value + 1 WHERE name = ?", (name,)
        )
        return int(
            conn.execute(
                "SELECT value FROM resource_meta WHERE name = ?", (name,)
            ).fetchone()[0]
        )

    def register_cell(self, cell: ResourceCell) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM resource_cells WHERE logical_key = ?",
                (cell.logical_key,),
            ).fetchone()
            if row is not None and cell.epoch < int(row[6]):
                conn.rollback()
                raise StaleLeaseEpoch("cell epoch moved backwards")
            self._upsert_cell(conn, cell)
            conn.commit()

    def _begin(self) -> sqlite3.Connection:
        conn = self._connect()
        conn.execute("BEGIN IMMEDIATE")
        return conn

    @staticmethod
    def _prune(conn: sqlite3.Connection, now_ms: int) -> None:
        conn.execute(
            "UPDATE resource_leases SET state = 'reclaimed' WHERE state IN ('active','renewed') AND expires_at_ms <= ?",
            (now_ms,),
        )

    @staticmethod
    def _validate_cell_scope(cell: ResourceCell, request: ResourceLeaseRequest) -> None:
        if cell.device_id != request.device_id:
            raise LeaseScopeMismatch("resource device/MIG identity changed")
        if cell.epoch != request.lease_epoch:
            raise StaleLeaseEpoch("resource authority epoch is stale")
        if cell.policy_digest != request.policy_digest:
            raise LeaseScopeMismatch("resource policy digest does not match")

    def _idempotent_replay_sql(
        self, conn: sqlite3.Connection, request: ResourceLeaseRequest, now: int
    ) -> ResourceLease | None:
        """Return the prior lease for ``request``'s idempotency key, or ``None``.

        Extracted from :meth:`acquire`: does not commit — the caller commits once
        it decides between the replay and the fresh-issue path, matching the
        original single commit point.
        """
        idem = conn.execute(
            "SELECT * FROM resource_leases WHERE logical_key = ? AND tenant_ref = ? AND principal_ref = ? AND idempotency_key = ?",
            (
                request.cell_logical_key,
                request.tenant_ref,
                request.principal_ref,
                request.idempotency_key,
            ),
        ).fetchone()
        if idem is None:
            return None
        prior = self._lease(idem)
        if prior.request.digest != request.digest:
            raise LeaseIdempotencyConflict("idempotency key changed request scope")
        prior.assert_live(now_ms=now)
        return prior

    @staticmethod
    def _assert_capacity_sql(
        conn: sqlite3.Connection,
        cell: ResourceCell,
        request: ResourceLeaseRequest,
        now: int,
    ) -> None:
        active = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM resource_leases WHERE logical_key = ? AND state IN ('active','renewed') AND expires_at_ms > ?",
            (request.cell_logical_key, now),
        ).fetchone()[0]
        tenant_active = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM resource_leases WHERE logical_key = ? AND tenant_ref = ? AND state IN ('active','renewed') AND expires_at_ms > ?",
            (request.cell_logical_key, request.tenant_ref, now),
        ).fetchone()[0]
        quota = cell.quota_for(request.tenant_ref)
        if quota is not None and int(tenant_active) + request.amount > quota:
            raise LeaseDenied("tenant resource quota exhausted")
        ceiling = (
            cell.capacity
            if request.priority_class in {"interactive", "foreground"}
            else max(0, cell.capacity - cell.reserved_floor)
        )
        available = ceiling - int(active)
        if available < request.amount:
            raise LeaseDenied(
                f"shared resource capacity exhausted ({available} available)"
            )

    def _insert_lease_sql(
        self, conn: sqlite3.Connection, request: ResourceLeaseRequest, now: int
    ) -> ResourceLease:
        fence = self._next(conn, "fence")
        revision = self._next(conn, "revision")
        lease_id = f"{request.resource_kind}:{request.resource_id}:{fence}"
        conn.execute(
            """INSERT INTO resource_leases(lease_id, logical_key, resource_kind, resource_id, node_id,
               device_id, tenant_ref, principal_ref, amount, idempotency_key, lease_epoch, ttl_ms,
               priority_class, policy_digest, request_digest, fence_token, issued_at_ms, expires_at_ms,
               state, authority_revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)""",
            (
                lease_id,
                request.cell_logical_key,
                request.resource_kind,
                request.resource_id,
                request.node_id,
                request.device_id,
                request.tenant_ref,
                request.principal_ref,
                request.amount,
                request.idempotency_key,
                request.lease_epoch,
                request.ttl_ms,
                request.priority_class,
                request.policy_digest,
                request.digest,
                fence,
                now,
                now + request.ttl_ms,
                revision,
            ),
        )
        row = conn.execute(
            "SELECT * FROM resource_leases WHERE lease_id = ?", (lease_id,)
        ).fetchone()
        return self._lease(row)

    def acquire(
        self, request: ResourceLeaseRequest, *, now_ms: int | None = None
    ) -> ResourceLease:
        now = _now_ms() if now_ms is None else int(now_ms)
        conn = self._begin()
        try:
            self._prune(conn, now)
            row = conn.execute(
                "SELECT * FROM resource_cells WHERE logical_key = ?",
                (request.cell_logical_key,),
            ).fetchone()
            if row is None:
                raise LeaseDenied("unknown shared resource cell")
            cell = self._cell(row)
            self._validate_cell_scope(cell, request)
            prior = self._idempotent_replay_sql(conn, request, now)
            if prior is not None:
                conn.commit()
                return prior
            self._assert_capacity_sql(conn, cell, request, now)
            lease = self._insert_lease_sql(conn, request, now)
            conn.commit()
            return lease
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _assert_owner_identity(
        row: sqlite3.Row | tuple[Any, ...], *, tenant_ref: str, principal_ref: str
    ) -> None:
        if row[6] != tenant_ref or row[7] != principal_ref:
            raise LeaseScopeMismatch("lease owner identity does not match")

    @staticmethod
    def _assert_owner_fence(
        row: sqlite3.Row | tuple[Any, ...], *, lease_epoch: int, fence_token: int
    ) -> None:
        if int(row[10]) != lease_epoch:
            raise StaleLeaseEpoch("lease epoch is stale")
        if int(row[15]) != fence_token:
            raise StaleLeaseFence("lease fence token does not match")

    @staticmethod
    def _assert_owner_cell_scope(
        conn: sqlite3.Connection,
        row: sqlite3.Row | tuple[Any, ...],
        lease_epoch: int,
    ) -> None:
        cell = conn.execute(
            "SELECT * FROM resource_cells WHERE logical_key = ?", (row[1],)
        ).fetchone()
        if cell is None or cell[4] != row[5] or int(cell[6]) != lease_epoch:
            raise StaleLeaseEpoch("resource device or epoch is stale")

    @staticmethod
    def _expire_if_stale(
        conn: sqlite3.Connection,
        lease_id: str,
        row: sqlite3.Row | tuple[Any, ...],
        now_ms: int,
    ) -> None:
        if row[18] not in _ACTIVE_STATES or int(row[17]) <= now_ms:
            conn.execute(
                "UPDATE resource_leases SET state = 'expired' WHERE lease_id = ?",
                (lease_id,),
            )
            raise LeaseExpired(lease_id)

    def _owner_row(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        *,
        tenant_ref: str,
        principal_ref: str,
        fence_token: int,
        lease_epoch: int,
        now_ms: int,
    ) -> sqlite3.Row | tuple[Any, ...]:
        row = conn.execute(
            "SELECT * FROM resource_leases WHERE lease_id = ?", (lease_id,)
        ).fetchone()
        if row is None:
            raise LeaseNotFound(lease_id)
        self._assert_owner_identity(
            row, tenant_ref=tenant_ref, principal_ref=principal_ref
        )
        self._assert_owner_fence(row, lease_epoch=lease_epoch, fence_token=fence_token)
        self._assert_owner_cell_scope(conn, row, lease_epoch)
        self._expire_if_stale(conn, lease_id, row, now_ms)
        return row

    def renew(
        self,
        lease_id: str,
        *,
        tenant_ref: str,
        principal_ref: str,
        fence_token: int,
        lease_epoch: int,
        ttl_ms: int,
        now_ms: int | None = None,
    ) -> ResourceLease:
        now = _now_ms() if now_ms is None else int(now_ms)
        _positive_int(ttl_ms, "ttl_ms", maximum=_MAX_TTL_MS)
        conn = self._begin()
        try:
            row = self._owner_row(
                conn,
                lease_id,
                tenant_ref=tenant_ref,
                principal_ref=principal_ref,
                fence_token=fence_token,
                lease_epoch=lease_epoch,
                now_ms=now,
            )
            revision = self._next(conn, "revision")
            conn.execute(
                "UPDATE resource_leases SET ttl_ms = ?, expires_at_ms = ?, state = 'renewed', authority_revision = ? WHERE lease_id = ?",
                (ttl_ms, now + ttl_ms, revision, lease_id),
            )
            row = conn.execute(
                "SELECT * FROM resource_leases WHERE lease_id = ?", (lease_id,)
            ).fetchone()
            conn.commit()
            return self._lease(row)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def release(
        self,
        lease_id: str,
        *,
        tenant_ref: str,
        principal_ref: str,
        fence_token: int,
        lease_epoch: int,
        now_ms: int | None = None,
    ) -> None:
        now = _now_ms() if now_ms is None else int(now_ms)
        conn = self._begin()
        try:
            row = conn.execute(
                "SELECT * FROM resource_leases WHERE lease_id = ?", (lease_id,)
            ).fetchone()
            if row is None:
                raise LeaseNotFound(lease_id)
            if row[6] != tenant_ref or row[7] != principal_ref:
                raise LeaseScopeMismatch("lease owner identity does not match")
            if int(row[15]) != fence_token:
                raise StaleLeaseFence("lease fence token does not match")
            if int(row[10]) != lease_epoch:
                raise StaleLeaseEpoch("lease epoch does not match")
            if row[18] in _TERMINAL_STATES:
                conn.commit()
                return
            self._owner_row(
                conn,
                lease_id,
                tenant_ref=tenant_ref,
                principal_ref=principal_ref,
                fence_token=fence_token,
                lease_epoch=lease_epoch,
                now_ms=now,
            )
            conn.execute(
                "UPDATE resource_leases SET state = 'released' WHERE lease_id = ?",
                (lease_id,),
            )
            self._next(conn, "revision")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def reclaim_expired(self, *, now_ms: int | None = None) -> tuple[str, ...]:
        now = _now_ms() if now_ms is None else int(now_ms)
        conn = self._begin()
        try:
            rows = conn.execute(
                "SELECT lease_id FROM resource_leases WHERE state IN ('active','renewed') AND expires_at_ms <= ?",
                (now,),
            ).fetchall()
            ids = tuple(str(row[0]) for row in rows)
            self._prune(conn, now)
            if ids:
                self._next(conn, "revision")
            conn.commit()
            return ids
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def rebind_device(
        self,
        *,
        resource_kind: str,
        resource_id: str,
        node_id: str,
        device_id: str,
        epoch: int,
        policy_digest: str,
    ) -> ResourceCell:
        logical = ":".join(
            (
                _text(resource_kind, "resource_kind"),
                _text(resource_id, "resource_id"),
                _text(node_id, "node_id"),
            )
        )
        conn = self._begin()
        try:
            row = conn.execute(
                "SELECT * FROM resource_cells WHERE logical_key = ?", (logical,)
            ).fetchone()
            if row is None:
                raise LeaseDenied("cannot rebind an unknown resource cell")
            prior = self._cell(row)
            if epoch <= prior.epoch:
                raise StaleLeaseEpoch("device epoch must increase")
            conn.execute(
                "UPDATE resource_leases SET state = 'stale' WHERE logical_key = ? AND state IN ('active','renewed')",
                (logical,),
            )
            cell = replace(
                prior,
                device_id=_text(device_id, "device_id"),
                epoch=epoch,
                policy_digest=_text(policy_digest, "policy_digest"),
            )
            self._upsert_cell(conn, cell)
            self._next(conn, "revision")
            conn.commit()
            return cell
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def status(self) -> dict[str, object]:
        with self._connect() as conn:
            return {
                "authority": "sqlite_reference",
                "path": self.path,
                "cells": int(
                    conn.execute("SELECT COUNT(*) FROM resource_cells").fetchone()[0]
                ),
                "leases": int(
                    conn.execute("SELECT COUNT(*) FROM resource_leases").fetchone()[0]
                ),
                "active": int(
                    conn.execute(
                        "SELECT COUNT(*) FROM resource_leases WHERE state IN ('active','renewed')"
                    ).fetchone()[0]
                ),
            }


_METHODS = {
    "AcquireResourceLease": "acquire",
    "RenewResourceLease": "renew",
    "ReleaseResourceLease": "release",
    "ReclaimExpiredResourceLeases": "reclaim_expired",
    "RebindResourceDevice": "rebind_device",
    "ResourceLeaseStatus": "status",
}


def _bounded_mapping(value: Any, operation: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or len(value) > 96:
        raise LeaseAuthorityUnavailable(f"{operation} returned an invalid result")
    return dict(value)


class EngineNativeResourceLeaseAuthority:
    """Strict transport to the engine's replicated resource-lease authority."""

    test_only = False

    def __init__(self, client: Any) -> None:
        namespace = getattr(client, "resource_leases", None)
        if namespace is None:
            raise LeaseAuthorityUnavailable(LeaseAuthorityUnavailable.code)
        self._client = client
        self._namespace = namespace

    def _operation(self, method: str) -> Any:
        attr = _METHODS.get(method)
        if attr is None:
            raise ValueError("unknown resource-lease method")
        operation = getattr(self._namespace, attr, None)
        if not callable(operation):
            raise LeaseAuthorityUnavailable(LeaseAuthorityUnavailable.code)
        return operation

    def _require(self, method: str) -> None:
        supports = getattr(self._client, "supports", None)
        if not callable(supports) or supports(method) is not True:
            raise LeaseAuthorityUnavailable(LeaseAuthorityUnavailable.code)

    def _call(self, method: str, request: Mapping[str, object]) -> dict[str, Any]:
        self._require(method)
        value = self._operation(method)(request=dict(request))
        if inspect.isawaitable(value):
            close = getattr(value, "close", None)
            if callable(close):
                close()
            raise TypeError("resource lease authority requires a synchronous client")
        return _bounded_mapping(value, method)

    @staticmethod
    def _decode(value: Mapping[str, Any]) -> ResourceLease:
        body = dict(value)
        request = body.get("request")
        if not isinstance(request, Mapping):
            raise LeaseAuthorityUnavailable(
                "native lease response omitted request binding"
            )
        typed = ResourceLeaseRequest(
            resource_kind=request.get("resource_kind", ""),
            resource_id=request.get("resource_id", ""),
            node_id=request.get("node_id", ""),
            device_id=request.get("device_id", ""),
            tenant_ref=request.get("tenant_ref", ""),
            principal_ref=request.get("principal_ref", ""),
            amount=int(request.get("amount", 0)),
            idempotency_key=request.get("idempotency_key", ""),
            lease_epoch=int(request.get("lease_epoch", 0)),
            ttl_ms=int(request.get("ttl_ms", 0)),
            priority_class=request.get("priority_class", ""),
            policy_digest=request.get("policy_digest", ""),
            schema=request.get("schema", RESOURCE_LEASE_SCHEMA),
        )
        if request.get("request_digest") not in {None, typed.digest}:
            raise LeaseAuthorityUnavailable("native lease response digest mismatch")
        return ResourceLease(
            lease_id=body.get("lease_id", ""),
            request=typed,
            fence_token=int(body.get("fence_token", 0)),
            issued_at_ms=int(body.get("issued_at_ms", 0)),
            expires_at_ms=int(body.get("expires_at_ms", 0)),
            state=str(body.get("state", "")),
            authority_revision=int(body.get("authority_revision", 0)),
        )

    def acquire(
        self, request: ResourceLeaseRequest, *, now_ms: int | None = None
    ) -> ResourceLease:
        payload = request.as_dict()
        if now_ms is not None:
            payload["now_ms"] = int(now_ms)
        lease = self._decode(self._call("AcquireResourceLease", payload))
        if lease.request.digest != request.digest:
            raise LeaseScopeMismatch(
                "native authority returned a different lease scope"
            )
        return lease

    def renew(
        self,
        lease_id: str,
        *,
        tenant_ref: str,
        principal_ref: str,
        fence_token: int,
        lease_epoch: int,
        ttl_ms: int,
        now_ms: int | None = None,
    ) -> ResourceLease:
        payload: dict[str, object] = {
            "lease_id": lease_id,
            "tenant_ref": tenant_ref,
            "principal_ref": principal_ref,
            "fence_token": fence_token,
            "lease_epoch": lease_epoch,
            "ttl_ms": ttl_ms,
        }
        if now_ms is not None:
            payload["now_ms"] = int(now_ms)
        lease = self._decode(self._call("RenewResourceLease", payload))
        if (
            lease.request.tenant_ref != tenant_ref
            or lease.request.principal_ref != principal_ref
        ):
            raise LeaseScopeMismatch("native authority returned a cross-owner renewal")
        return lease

    def release(
        self,
        lease_id: str,
        *,
        tenant_ref: str,
        principal_ref: str,
        fence_token: int,
        lease_epoch: int,
        now_ms: int | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "lease_id": lease_id,
            "tenant_ref": tenant_ref,
            "principal_ref": principal_ref,
            "fence_token": fence_token,
            "lease_epoch": lease_epoch,
        }
        if now_ms is not None:
            payload["now_ms"] = int(now_ms)
        self._call("ReleaseResourceLease", payload)

    def reclaim_expired(self, *, now_ms: int | None = None) -> tuple[str, ...]:
        payload: dict[str, object] = {}
        if now_ms is not None:
            payload["now_ms"] = int(now_ms)
        result = self._call("ReclaimExpiredResourceLeases", payload)
        values = result.get("lease_ids", ())
        if not isinstance(values, list | tuple) or len(values) > 100_000:
            raise LeaseAuthorityUnavailable("native reclaim result is malformed")
        return tuple(str(value) for value in values)

    def rebind_device(
        self,
        *,
        resource_kind: str,
        resource_id: str,
        node_id: str,
        device_id: str,
        epoch: int,
        policy_digest: str,
    ) -> dict[str, Any]:
        return self._call(
            "RebindResourceDevice",
            {
                "resource_kind": resource_kind,
                "resource_id": resource_id,
                "node_id": node_id,
                "device_id": device_id,
                "epoch": epoch,
                "policy_digest": policy_digest,
            },
        )

    def status(self) -> dict[str, Any]:
        return self._call("ResourceLeaseStatus", {})


@contextmanager
def hold_resource_lease(
    authority: SharedResourceLeaseAuthority,
    request: ResourceLeaseRequest,
    *,
    now_ms: int | None = None,
) -> Iterator[ResourceLease]:
    """Acquire and release one lease, retaining fail-closed release semantics."""

    lease = authority.acquire(request, now_ms=now_ms)
    try:
        yield lease
    finally:
        authority.release(
            lease.lease_id,
            tenant_ref=request.tenant_ref,
            principal_ref=request.principal_ref,
            fence_token=lease.fence_token,
            lease_epoch=lease.lease_epoch,
            now_ms=now_ms,
        )
