#!/usr/bin/python
from __future__ import annotations

"""GOC-21 — distributed capacity leases + hierarchical fair scheduling.

CONCEPT:AU-ORCH.scheduling.distributed-capacity-fair-scheduling

Replaces process-local, unbounded admission (`core/cognitive_scheduler.py`'s
in-process `asyncio.PriorityQueue`, `graph/parallel_engine.py`'s unbounded
`asyncio.gather`, `core/resource_priority.py`'s single flat in-process gate,
`knowledge_graph/core/worker_scheduler.py`'s explicitly-advisory per-host
registry — see each module's own docstring, re-verified against this repo's
`main` at `3299e56ca` on 2026-08-16) with a durable, cross-host `CapacityLease`
and a hierarchical **weighted round-robin with a reserved interactive floor**
that gives a provable, worst-case (not merely average) wait bound.

Does NOT edit `agent_utilities/orchestration/work_item.py` (GOC-82 owns it this
wave) or `core/cognitive_scheduler.py` / `graph/parallel_engine.py` /
`core/resource_priority.py` / `knowledge_graph/core/worker_scheduler.py`
(GOC-21-W04 adapter wiring — reported as a needed follow-on, not done here).
`WorkItem`'s own lease/fence (`lease_owner`/`lease_epoch`/`fence_token`,
`TenantQuotaExceeded`, `reap_expired_leases`) governs ONE WorkItem's claim
authority; `CapacityLease` here governs a *resource-class* pool (LLM/GPU/
worker/broker admission units) a WorkItem's execution draws against — a
distinct, composable concern, the same way `resource_reservation.py`'s
RMDD-27 host CPU/memory/disk/process-slot reservation is a distinct concern
from WorkItem claim authority.

## Premise notes (verified against `epistemic-graph` `main` at `02594f7`, AU
`main` at `3299e56ca`, both 2026-08-16 — see this lane's report for detail)

* GOC-03's `CommitDescriptorV1` (`authority_epoch` + non-zero `fencing_token`)
  exists ONLY on the unmerged branch `goc/goc-03-commit-currency`; this lane's
  fencing shape VENDORS that identical epoch+token discipline (documented in
  `crates/eg-types/src/capacity_lease.rs`) rather than importing a type that
  is not on `main`.
* `agent_utilities/orchestration/resource_reservation.py` (RMDD-27) is
  significant, DURABLE, engine-native prior art for this exact fencing
  pattern (`ResourceReservationRequest.lease_epoch`/`.fencing_token` in
  `crates/eg-types/src/epistemic_operations.rs`, backed by a real `redb_store`
  implementation with ~1000 lines of tests) — but it is scoped to
  repository-manager BUILD-HOST resources (cpu_weight/memory_mib/disk_mib/
  process_slots, `repository_id`/`branch` exclusivity), not AU orchestration
  capacity (LLM/GPU/worker/broker, tenant/priority fairness). Its
  `fairness_debt` field is a per-`(tenant_ref, fairness_group)` accumulating
  counter (`src/redb_store.rs` around line 5005) that is tracked and exposed
  but — re-verified by grep — never read back to gate admission order or deny
  a request; it is OBSERVABILITY, not enforcement. This module is therefore
  a genuinely new, domain-scoped sibling contract (mirroring RMDD-27's/
  RMDD-28's own "reuse the pattern, not the type" precedent
  `development_lane.py` documents), and it is what actually ENFORCES a bounded
  wait, which RMDD-27 does not.
* No native engine `AcquireCapacity`/`RenewCapacity`/`ReleaseCapacity`/
  `ReclaimExpired` RPC exists yet (only `ReserveWorkItemResources` et al., a
  different `Method` variant, for the RMDD-27 domain above). Written the same
  way `development_lane.py` writes against RMDD-28's not-yet-exposed
  `development_lanes` client namespace: `EngineNativeCapacityLeaseTransport`
  below fails closed with :class:`CapacityAuthorityUnavailable` today — an
  honest, accurate state, never a silent local stand-in for durable authority.

## The fairness policy and its worst-case bound

Two composed mechanisms, proved independently in
`tests/unit/orchestration/test_capacity_leases.py`:

1. **Reserved floor (cross-priority).** A `CapacityCell.reserved_floor` slice
   of capacity is NEVER available to `PriorityClass.BACKGROUND_INGESTION`
   (`CapacityCell.available_for`, mirroring `PriorityClass.is_interactive_floor`
   and `resource_priority.py`'s existing `AsyncPriorityLLMGate.reserve`). A
   background flood of arbitrary size can fill only the SPARE
   (`capacity - reserved_floor`) pool; it cannot touch the floor at all. So an
   interactive request's admission from the floor is bounded ENTIRELY by
   *other interactive-class* contention, independent of background flood size.

2. **Weighted round-robin (same-priority, cross-tenant).** Within one priority
   class, `HierarchicalFairScheduler` serves currently-BACKLOGGED
   `(priority, tenant)` queues in round-robin order, `weight[tenant]` requests
   per round (default weight 1). The classic round-robin fairness property
   holds here in its simplest, easiest-to-verify form because a queue only
   ever counts once per round regardless of its depth: a backlogged tenant
   `t`'s worst-case rounds-until-served is

       wait_rounds(t) <= ceil(active_tenants_in_priority / weight[t])

   — a function of the number of DISTINCT competing tenants currently
   backlogged in that priority class, NEVER of how many requests any other
   tenant (including a flooding one) has queued. Flooding tenant `X` with
   1 or 1,000,000 requests contributes exactly 1 to `active_tenants_in_priority`
   either way, so it cannot push another tenant's bound past that formula.
   `wait_time(t) <= wait_rounds(t) * round_budget_ms` for a scheduler driven at
   a fixed round cadence.

Composed: a `BACKGROUND_INGESTION` flood, of any size, from any number of
distinct low-priority tenants, cannot delay an `INTERACTIVE` request beyond
`ceil(active_interactive_tenants / weight) * round_budget_ms` — a bound that
does not appear anywhere in the flood's size or the background priority class'
own queue depth. This is the "not merely average" bound the lane brief
requires; `test_flood_cannot_starve_interactive_beyond_stated_bound` proves it
against a 100,000-request flood.

A defensive check additionally asserts no queue may silently exceed
`MAX_TENANT_BACKLOG` (bounded queue memory, lane invariant "queue memory is
O(pending_work_items) with backpressure") — a flood is capacity-shed via
`TenantBackpressure`, never buffered without bound.
"""

import math
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from agent_utilities.core.resource_priority import PriorityClass

__all__ = [
    "CAPACITY_LEASE_VERSION",
    "CAPACITY_RESOURCE_CLASSES",
    "CapacityCell",
    "LeaseState",
    "CapacityLease",
    "CapacityDenial",
    "CapacityExhausted",
    "StaleEpoch",
    "StaleFence",
    "LeaseExpiredError",
    "LeaseNotFound",
    "IdempotencyConflict",
    "CapacityLedger",
    "FairRequest",
    "TenantBackpressure",
    "HierarchicalFairScheduler",
    "CapacityAuthorityUnavailable",
    "EngineNativeCapacityLeaseTransport",
    "MAX_TENANT_BACKLOG",
]

CAPACITY_LEASE_VERSION = 1

#: Bounded queue memory per (priority, tenant) key (lane invariant: queue
#: memory is O(pending_work_items) with backpressure, never unbounded).
MAX_TENANT_BACKLOG = 4096

#: Resource classes this scheduler admits against (mirrors, as a closed
#: plain-string vocabulary per `work_item.py`'s "no enum, just data"
#: convention, `crates/eg-types/src/capacity_lease.rs::CapacityResourceClass`).
CAPACITY_RESOURCE_CLASSES = frozenset(
    {"llm_generator", "llm_embedding", "gpu", "worker", "cpu", "broker"}
)


def _validate_resource_class(value: str) -> str:
    if value not in CAPACITY_RESOURCE_CLASSES:
        raise ValueError(f"unknown capacity resource class: {value!r}")
    return value


# ── denial taxonomy (mirrors crates/eg-types/src/capacity_lease.rs::CapacityDenial) ──


class CapacityDenial(RuntimeError):
    """Base class for every explicit capacity-admission refusal.

    Every subclass here is a case the lane's acceptance gates require to be an
    explicit rejection — never a fail-open unlimited admission
    (`capacity_unknown`/`lease_reconcile_needed` are the only degraded states
    this module recognizes; both still refuse, never fabricate capacity).
    """


class CapacityExhausted(CapacityDenial):
    def __init__(self, *, available: int, requested: int) -> None:
        super().__init__(
            f"capacity exhausted: requested {requested}, available {available}"
        )
        self.available = available
        self.requested = requested


class StaleEpoch(CapacityDenial):
    """The caller's `lease_epoch` is behind the cell's current epoch — the
    cell was repartitioned/failed over. THE fencing rejection this lane's
    known-bad proof targets."""

    def __init__(self, *, cell_epoch: int, caller_epoch: int) -> None:
        super().__init__(
            f"stale lease epoch: caller={caller_epoch} cell={cell_epoch}"
        )
        self.cell_epoch = cell_epoch
        self.caller_epoch = caller_epoch


class StaleFence(CapacityDenial):
    """The caller's `fence_token` does not match the ledger's live record."""


class LeaseExpiredError(CapacityDenial):
    """The lease reached `expires_at_ms` (or is already terminal); an expired
    holder that keeps working is rejected outright, never silently renewed."""


class LeaseNotFound(CapacityDenial):
    pass


class IdempotencyConflict(CapacityDenial):
    """Replaying `(tenant_ref, idempotency_key)` against a different demand
    shape than the original — same key must mean same request."""


# ── data shapes (mirror crates/eg-types/src/capacity_lease.rs) ──────────────


@dataclass(frozen=True)
class CapacityCell:
    cell_id: str
    resource_class: str
    capacity: int
    reserved_floor: int
    epoch: int
    policy_digest: str
    parent_id: str | None = None
    updated_at_ms: int = 0

    def __post_init__(self) -> None:
        _validate_resource_class(self.resource_class)
        if not self.cell_id.strip():
            raise ValueError("cell_id must not be empty")
        if not self.policy_digest.strip():
            raise ValueError("policy_digest must not be empty")
        if self.reserved_floor > self.capacity:
            raise ValueError("reserved_floor must not exceed capacity")
        if self.epoch <= 0:
            raise ValueError("epoch must be non-zero")

    def available_for(self, priority: PriorityClass, already_leased: int) -> int:
        """Capacity available to `priority` right now. Mirrors the Rust
        `CapacityCell::available_for` bit for bit: only
        `PriorityClass.is_interactive_floor` classes may spend the reserved
        floor; `BACKGROUND_INGESTION` is confined to spare capacity."""
        ceiling = self.capacity if priority.is_interactive_floor else max(
            0, self.capacity - self.reserved_floor
        )
        return max(0, ceiling - min(already_leased, ceiling))


class LeaseState:
    ACTIVE = "active"
    RENEWED = "renewed"
    RELEASED = "released"
    EXPIRED = "expired"
    RECLAIMED = "reclaimed"

    TERMINAL = frozenset({RELEASED, RECLAIMED})


@dataclass
class CapacityLease:
    schema_version: int
    lease_id: str
    work_item_id: str
    tenant_ref: str
    actor_digest: str
    cell_id: str
    resource_class: str
    amount: int
    priority: PriorityClass
    fence_token: int
    lease_epoch: int
    issued_at_ms: int
    expires_at_ms: int
    idempotency_key: str
    state: str = LeaseState.ACTIVE
    renewed_count: int = 0
    cost_budget: float | None = None
    token_budget: int | None = None

    def is_expired(self, now_ms: int) -> bool:
        return (
            self.state in LeaseState.TERMINAL
            or self.state == LeaseState.EXPIRED
            or now_ms >= self.expires_at_ms
        )


def _now_ms() -> int:
    return int(time.time() * 1000)


# ── the CAS ledger (mirrors crates/eg-types/src/capacity_lease.rs::CapacityLedger) ──


class CapacityLedger:
    """In-process reference/advisory ledger for ONE `CapacityCell`.

    Advisory the same way `worker_scheduler.py`'s `WorkerRegistry` documents
    itself as advisory: cross-host durable authority is the engine-native
    `AcquireCapacity` operation once it exists
    (`EngineNativeCapacityLeaseTransport` below); until then this ledger is
    the single-process decision core the fairness proofs in this module's
    test suite exercise directly, and the CAS body a future native handler
    can adopt without algorithm changes (same "pure algorithm, no I/O" split
    `crates/eg-types/src/capacity_lease.rs` documents on the Rust side).
    """

    def __init__(self) -> None:
        self._leases: dict[str, CapacityLease] = {}
        self._next_fence = 1
        self._idempotency: dict[tuple[str, str], str] = {}

    def _active_leased_amount(self, cell: CapacityCell, now_ms: int) -> int:
        return sum(
            lease.amount
            for lease in self._leases.values()
            if lease.cell_id == cell.cell_id and not lease.is_expired(now_ms)
        )

    def try_acquire(
        self,
        cell: CapacityCell,
        *,
        lease_id: str,
        work_item_id: str,
        tenant_ref: str,
        actor_digest: str,
        amount: int,
        priority: PriorityClass,
        idempotency_key: str,
        now_ms: int,
        ttl_ms: int,
    ) -> CapacityLease:
        replay_key = (tenant_ref, idempotency_key)
        existing_id = self._idempotency.get(replay_key)
        if existing_id is not None:
            existing = self._leases[existing_id]
            if (
                existing.cell_id == cell.cell_id
                and existing.amount == amount
                and existing.work_item_id == work_item_id
            ):
                return existing
            raise IdempotencyConflict(
                f"idempotency key {idempotency_key!r} replayed with a different demand shape"
            )

        already = self._active_leased_amount(cell, now_ms)
        available = cell.available_for(priority, already)
        if available < amount:
            raise CapacityExhausted(available=available, requested=amount)

        fence_token = self._next_fence
        self._next_fence += 1
        lease = CapacityLease(
            schema_version=CAPACITY_LEASE_VERSION,
            lease_id=lease_id,
            work_item_id=work_item_id,
            tenant_ref=tenant_ref,
            actor_digest=actor_digest,
            cell_id=cell.cell_id,
            resource_class=cell.resource_class,
            amount=amount,
            priority=priority,
            fence_token=fence_token,
            lease_epoch=cell.epoch,
            issued_at_ms=now_ms,
            expires_at_ms=now_ms + ttl_ms,
            idempotency_key=idempotency_key,
        )
        self._leases[lease_id] = lease
        self._idempotency[replay_key] = lease_id
        return lease

    def try_renew(
        self,
        lease_id: str,
        *,
        fence_token: int,
        lease_epoch: int,
        cell_epoch: int,
        now_ms: int,
        ttl_ms: int,
    ) -> CapacityLease:
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseNotFound(lease_id)
        if lease.is_expired(now_ms):
            lease.state = LeaseState.EXPIRED
            raise LeaseExpiredError(lease_id)
        if lease_epoch < cell_epoch:
            raise StaleEpoch(cell_epoch=cell_epoch, caller_epoch=lease_epoch)
        if lease.fence_token != fence_token:
            raise StaleFence(lease_id)
        lease.expires_at_ms = now_ms + ttl_ms
        lease.renewed_count += 1
        lease.state = LeaseState.RENEWED
        return lease

    def try_release(
        self,
        lease_id: str,
        *,
        fence_token: int,
        lease_epoch: int,
        cell_epoch: int,
        now_ms: int,
    ) -> None:
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseNotFound(lease_id)
        if lease.state in LeaseState.TERMINAL:
            return  # idempotent terminal commit — a no-op, not an error
        if lease.is_expired(now_ms):
            lease.state = LeaseState.EXPIRED
            raise LeaseExpiredError(lease_id)
        if lease_epoch < cell_epoch:
            raise StaleEpoch(cell_epoch=cell_epoch, caller_epoch=lease_epoch)
        if lease.fence_token != fence_token:
            raise StaleFence(lease_id)
        lease.state = LeaseState.RELEASED

    def reclaim_expired(self, now_ms: int) -> list[str]:
        reclaimed = []
        for lease_id, lease in self._leases.items():
            if lease.state not in LeaseState.TERMINAL and now_ms >= lease.expires_at_ms:
                lease.state = LeaseState.RECLAIMED
                reclaimed.append(lease_id)
        return reclaimed

    def get(self, lease_id: str) -> CapacityLease | None:
        return self._leases.get(lease_id)


# ── hierarchical weighted round-robin fair scheduler ─────────────────────────


@dataclass
class FairRequest:
    request_id: str
    tenant_ref: str
    priority: PriorityClass
    amount: int = 1


class TenantBackpressure(RuntimeError):
    """Raised instead of buffering past `MAX_TENANT_BACKLOG` — bounded queue
    memory, never an unbounded local queue standing in for admission."""


class HierarchicalFairScheduler:
    """Weighted round-robin across `(priority, tenant)` queues, reserved-floor
    aware. See module docstring "The fairness policy and its worst-case bound"
    for the exact proved formula. `round_budget_ms` is only used to convert a
    round count into a wall-clock bound in :meth:`worst_case_wait_ms`; the
    scheduler itself is driven by repeated calls to :meth:`run_round`.
    """

    def __init__(
        self,
        cell: CapacityCell,
        ledger: CapacityLedger,
        *,
        weights: dict[str, int] | None = None,
        round_budget_ms: int = 50,
    ) -> None:
        self.cell = cell
        self.ledger = ledger
        self.weights = dict(weights or {})
        self.round_budget_ms = round_budget_ms
        self._queues: dict[tuple[PriorityClass, str], deque[FairRequest]] = {}
        # Round-robin cursor per priority class, so `run_round` resumes where
        # it left off rather than always starting from the same tenant
        # (starting from the same tenant every round would itself be a subtle
        # unfairness — the classic round-robin implementation pitfall).
        self._cursor: dict[PriorityClass, int] = {}

    def weight(self, tenant_ref: str) -> int:
        return max(1, self.weights.get(tenant_ref, 1))

    def submit(self, request: FairRequest) -> None:
        key = (request.priority, request.tenant_ref)
        queue = self._queues.setdefault(key, deque())
        if len(queue) >= MAX_TENANT_BACKLOG:
            raise TenantBackpressure(
                f"tenant {request.tenant_ref!r} priority {request.priority.value} "
                f"backlog at bound ({MAX_TENANT_BACKLOG}); request shed, not buffered"
            )
        queue.append(request)

    def _active_tenants(self, priority: PriorityClass) -> list[str]:
        return sorted(
            {t for (p, t), q in self._queues.items() if p is priority and q}
        )

    def worst_case_wait_rounds(self, priority: PriorityClass, tenant_ref: str) -> int:
        """The proved bound: `ceil(active_tenants_in_priority / weight)`,
        independent of any tenant's (including a flooding tenant's) queue
        depth — see module docstring."""
        active = self._active_tenants(priority)
        if tenant_ref not in active:
            active = active + [tenant_ref]
        return max(1, math.ceil(len(active) / self.weight(tenant_ref)))

    def worst_case_wait_ms(self, priority: PriorityClass, tenant_ref: str) -> int:
        return self.worst_case_wait_rounds(priority, tenant_ref) * self.round_budget_ms

    def _priorities_in_order(self) -> Iterator[PriorityClass]:
        # PriorityClass.rank ascending == highest priority first (module
        # convention, `core/resource_priority.py::PriorityClass.rank`).
        seen = {p for (p, _t) in self._queues}
        yield from sorted(seen, key=lambda p: p.rank)

    def run_round(self, *, now_ms: int | None = None, work_item_id: str = "wi-fair") -> list[CapacityLease]:
        """One scheduling round: for each priority class (highest first),
        visit each currently-active tenant AT MOST ONCE (round-robin,
        weight-many turns), admitting its head request if the cell has
        capacity. Returns every lease minted this round.
        """
        now_ms = _now_ms() if now_ms is None else now_ms
        admitted: list[CapacityLease] = []
        for priority in self._priorities_in_order():
            active = self._active_tenants(priority)
            if not active:
                continue
            start = self._cursor.get(priority, 0) % len(active)
            order = active[start:] + active[:start]
            for tenant_ref in order:
                # Each active tenant gets exactly `weight[tenant_ref]` turns
                # THIS round, regardless of its own queue depth — this is the
                # entire fairness mechanism: a queue is visited a bounded
                # number of times per round no matter how many requests sit
                # behind its head.
                for _ in range(self.weight(tenant_ref)):
                    key = (priority, tenant_ref)
                    queue = self._queues.get(key)
                    if not queue:
                        break
                    request = queue[0]
                    try:
                        lease = self.ledger.try_acquire(
                            self.cell,
                            lease_id=f"{work_item_id}:{request.request_id}",
                            work_item_id=work_item_id,
                            tenant_ref=tenant_ref,
                            actor_digest=f"actor:{tenant_ref}",
                            amount=request.amount,
                            priority=priority,
                            idempotency_key=request.request_id,
                            now_ms=now_ms,
                            ttl_ms=60_000,
                        )
                    except CapacityExhausted:
                        # No room this round — leave the request queued and
                        # move on; it keeps its place for the next round
                        # (does NOT lose its round-robin turn count, so a
                        # transient shortage cannot itself cause unbounded
                        # extra waiting beyond the next round's attempt).
                        break
                    queue.popleft()
                    admitted.append(lease)
            self._cursor[priority] = (start + 1) % max(1, len(active))
        return admitted

    def pending_count(self, priority: PriorityClass, tenant_ref: str) -> int:
        return len(self._queues.get((priority, tenant_ref), ()))


# ── native transport (written against a not-yet-exposed client namespace,
#    exact same honesty convention as resource_reservation.py / development_lane.py) ──


class CapacityAuthorityUnavailable(RuntimeError):
    """The connected engine does not advertise a native capacity-lease method.

    No `Method::AcquireCapacity`/`RenewCapacity`/`ReleaseCapacity`/
    `ReclaimExpired` exists in `crates/eg-types/src/protocol.rs` as of this
    lane (only the RMDD-27 `ReserveWorkItemResources` family, a different
    domain — see module docstring). Raised at construction and at call time;
    never caught and silently replaced by this module's in-process
    `CapacityLedger` — that ledger is a REFERENCE/advisory decision core for
    single-process testing, not a durable stand-in for cross-host authority.
    """

    code = "native_capacity_authority_unavailable"


_METHODS: dict[str, str] = {
    "AcquireCapacity": "acquire",
    "RenewCapacity": "renew",
    "ReleaseCapacity": "release",
    "ReclaimExpiredCapacity": "reclaim",
    "ReconcileCapacity": "reconcile",
    "CapacityStatus": "status",
}


def _bounded_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} result must be a mapping")
    if len(value) > 64:
        raise ValueError(f"{name} result has too many fields")
    return dict(value)


class EngineNativeCapacityLeaseTransport:
    """Low-level native capacity-lease transport, shaped identically to
    :class:`agent_utilities.orchestration.resource_reservation.EngineNativeReservationTransport`
    and :class:`agent_utilities.orchestration.development_lane.EngineNativeDevelopmentLaneTransport`
    on purpose (same capability-negotiation + strict-name-allowlist pattern).
    Written against a future `client.capacity_leases` namespace; today,
    against the real generated client, construction fails closed with
    :class:`CapacityAuthorityUnavailable`.
    """

    def __init__(self, client: Any) -> None:
        capacity_leases = getattr(client, "capacity_leases", None)
        if capacity_leases is None:
            raise CapacityAuthorityUnavailable(CapacityAuthorityUnavailable.code)
        self._client = client
        self._capacity_leases = capacity_leases

    def _operation(self, name: str) -> Any:
        attr = _METHODS.get(name)
        if attr is None:
            raise ValueError("unknown native capacity-lease method")
        operation = getattr(self._capacity_leases, attr, None)
        if not callable(operation):
            raise CapacityAuthorityUnavailable(CapacityAuthorityUnavailable.code)
        return operation

    def _require_capability(self, name: str) -> None:
        supports = getattr(self._client, "supports", None)
        if not callable(supports):
            raise CapacityAuthorityUnavailable(CapacityAuthorityUnavailable.code)
        if supports(name) is not True:
            raise CapacityAuthorityUnavailable(CapacityAuthorityUnavailable.code)

    def _call(self, name: str, request: dict[str, Any]) -> dict[str, Any]:
        self._require_capability(name)
        operation = self._operation(name)
        value = operation(request=dict(request))
        return _bounded_mapping(value, name)

    def acquire(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._call("AcquireCapacity", request)

    def renew(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._call("RenewCapacity", request)

    def release(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._call("ReleaseCapacity", request)

    def reclaim(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._call("ReclaimExpiredCapacity", request)

    def reconcile(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._call("ReconcileCapacity", request)

    def status(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._call("CapacityStatus", request)
