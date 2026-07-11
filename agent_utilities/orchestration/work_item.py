#!/usr/bin/python
from __future__ import annotations

"""The ONE engine-native ``WorkItem`` state machine (AU-P1-1).

CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch — unifies Goal/Task/AgentTask/Loop/dispatch

AU-P0-3 wired the engine-native claim/lease/CAS/fencing primitive into the
``:AgentTask`` dispatch path (:mod:`agent_utilities.orchestration.engine_claim`,
:mod:`agent_utilities.orchestration.agent_dispatch_worker`). Before that, and
still today for the older subsystems, "a unit of work advancing toward done"
is represented FOUR independent ways with four different status vocabularies:

* ``:Task`` (:mod:`agent_utilities.knowledge_graph.core.engine_tasks`) — the
  ingestion/background-job queue: ``pending|running|blocked|scheduled|
  completed|failed|cancelled|dead_letter``, with priority lanes, an admission
  gate, a zombie reaper, and exponential-backoff retry-then-dead-letter.
* ``:AgentTask`` (:class:`~agent_utilities.models.knowledge_graph.AgentTaskNode`)
  — the DAG-aware unit the agent-dispatch fleet claims: ``pending|blocked|
  ready|running|completed|failed|cancelled|unroutable``, lease-based
  (:class:`~agent_utilities.models.knowledge_graph.AgentLeaseNode`),
  checkpoint-resumable.
* Loop/goal (:mod:`agent_utilities.knowledge_graph.research.loops`) —
  ``TERMINAL_STATUS = {completed, failed, cancelled, rejected}``, CAS-claimed
  via ``claim_loop``; :class:`~agent_utilities.models.goal.GoalStatus` layers
  an ephemeral ``pending|running|validating|completed|failed|cancelled|
  paused|orphaned`` view on top for the ``/goal`` UX.
* The dispatch envelope (:mod:`agent_utilities.orchestration.agent_dispatch`)
  — ``pending -> completed|failed|skipped|expired``, no persisted state
  machine of its own (it rides whichever of the above it wraps).

This module makes ONE versioned state machine authoritative going forward::

    submitted -> ready -> leased(fencing_token) -> running(heartbeat,attempt)
        -> succeeded(result_ref) | failed(error_ref) | cancelled | dead_letter

**What is MIGRATED (WorkItem is authoritative) vs SHIMMED (read-only
normalization, write path unchanged) vs FOLLOW-UP (untouched this session) —
see the AU-P1-1 report for the full enumeration; the short version:**

* MIGRATED: the ``:AgentTask``/agent-dispatch claim path, opt-in via
  ``AGENT_CLAIM_BACKEND=workitem`` (:mod:`engine_claim`). See
  :func:`ensure_agent_task_work_item`/:func:`claim_agent_task_via_work_item`
  — every claim/heartbeat/commit/dependency-release for an ``:AgentTask``
  routed through this backend flows through THIS module's state machine; the
  legacy ``:AgentTask.status``/``:AgentLease`` node are mirrored (never
  read) for the unmigrated readers (``fleet_reconciler``, dashboards).
* SHIMMED (read-only): :func:`work_item_view_of_loop` (Goal/Loop) and
  :func:`work_item_view_of_task` (the ``:Task`` ingestion queue) project
  each subsystem's native status onto :class:`WorkItemStatus` for unified
  observability WITHOUT touching either subsystem's write path
  (``submit_loop``/``claim_loop``/``run_goal_loop`` and
  ``submit_task``/``_claim_next_task``/``_fail_or_retry_task`` are all
  UNCHANGED).
* FOLLOW-UP (not touched): rewriting ``engine_tasks.py``'s ingestion queue
  (lanes, admission gate, reaper, scheduled/blocked promotion sweep) onto
  WorkItem as sole storage, and the simpler team-collaboration ``TaskNode``
  (``pending|in_progress|completed``) — both out of scope for one session;
  the KG-backend (``AGENT_CLAIM_BACKEND=kg``, still the default) and
  engine-native backend (``AGENT_CLAIM_BACKEND=engine``) remain fully intact.

Reuses, never reinvents:

* **Atomicity** — every transition below is a single
  ``backend.compare_and_set_node_fields(node_id, conditions, updates)`` call,
  the SAME primitive ``TaskManagerMixin._claim_next_task`` (``:Task``) and
  ``research.loops.claim_loop`` (Loop) already use. A CAS that loses the race
  is a normal "someone else got there first" outcome, never an exception.
* **Lease/fencing** — the AgentTask bridge below mirrors a companion
  :class:`~agent_utilities.models.knowledge_graph.AgentLeaseNode` so the
  EXISTING ``agent_dispatch_worker._fence_still_valid`` gate keeps working
  unchanged for readers that don't know about WorkItem yet.
* **Idempotent commit** — :func:`commit_result`'s CAS precondition requires
  ``status == "running"``; once a commit wins, a second commit's identical
  CAS necessarily loses (the status is already terminal), so double-commit
  is a no-op by construction rather than a special-cased check.
* **Atomic dependency release** — :func:`_release_one_dependency` decrements
  ``dep_count`` AND flips ``submitted -> ready`` in the SAME CAS call, so
  there is never a window where the counter is decremented but the status
  hasn't caught up (or vice versa).
"""

import logging
import time
import uuid
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "WorkItemStatus",
    "TERMINAL_WORK_ITEM_STATUSES",
    "WorkItemBackendUnavailable",
    "TenantQuotaExceeded",
    "new_work_item_id",
    "submit_work_item",
    "get_work_item",
    "claim_specific",
    "claim_next",
    "mark_running",
    "heartbeat",
    "commit_result",
    "cancel_work_item",
    "reap_expired_leases",
    "tenant_in_flight_count",
    "agent_task_work_item_id",
    "ensure_agent_task_work_item",
    "claim_agent_task_via_work_item",
    "work_item_view_of_loop",
    "work_item_view_of_task",
]


class WorkItemStatus(StrEnum):
    """The eight states of the ONE work-item lifecycle (AU-P1-1)."""

    SUBMITTED = "submitted"
    READY = "ready"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEAD_LETTER = "dead_letter"


TERMINAL_WORK_ITEM_STATUSES = frozenset(
    {
        WorkItemStatus.SUCCEEDED.value,
        WorkItemStatus.FAILED.value,
        WorkItemStatus.CANCELLED.value,
        WorkItemStatus.DEAD_LETTER.value,
    }
)

_NODE_LABEL = "WorkItem"

#: Default renewable-lease TTL (mirrors ``agent_dispatch_worker.CLAIM_TTL_S``;
#: not imported directly to avoid a hard dependency edge — this module is
#: reused BY agent_dispatch_worker's bridge, not the other way around).
DEFAULT_LEASE_TTL_S = 3600.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE_S = 30.0
#: Bounded retries for optimistic-concurrency CAS loops (dependency release,
#: downstream indexing, cancel) — mirrors ``engine_tasks._CLAIM_MAX_RETRIES``.
_CAS_LOOP_MAX_RETRIES = 8

#: Fields returned by :func:`get_work_item` (everything but ``id``, which is
#: always requested separately so the row is never empty on a hit).
_FIELDS: tuple[str, ...] = (
    "status",
    "kind",
    "payload_ref",
    "tenant",
    "idempotency_key",
    "depends_on",
    "dep_count",
    "downstream_ids",
    "prio_bucket",
    "deadline_unix",
    "budget",
    "resource_class",
    "fairness_group",
    "attempt",
    "max_attempts",
    "backoff_base_s",
    "next_retry_at",
    "lease_owner",
    "lease_epoch",
    "lease_expires_at",
    "heartbeat_at",
    "correlation_id",
    "created_at",
    "updated_at",
    "submitted_at",
    "completed_at",
    "result_ref",
    "error_ref",
)


class WorkItemBackendUnavailable(RuntimeError):
    """Raised when the connected engine has no atomic ``compare_and_set_node_fields``.

    Fails LOUD rather than silently degrading to a non-atomic read-then-write
    — a WorkItem transition without a real CAS backing it is not a safe
    claim/commit, and AU-P0-3's fail-closed discipline forbids a fabricated
    success in that situation.
    """


class TenantQuotaExceeded(RuntimeError):
    """Raised by :func:`submit_work_item` when a tenant is at its in-flight cap."""


def _now() -> float:
    return time.time()


def new_work_item_id() -> str:
    """Full 128-bit id (AU-P0-3 discipline: no truncated/32-bit ids)."""
    return f"workitem:{uuid.uuid4().hex}"


def _default_correlation_id() -> str:
    try:
        from agent_utilities.observability import correlation

        return correlation.ensure_correlation_id()
    except Exception:  # noqa: BLE001 — best-effort context
        return ""


def _default_token() -> str:
    try:
        from agent_utilities.orchestration.agent_dispatch_worker import worker_token

        return worker_token()
    except Exception:  # noqa: BLE001 — standalone fallback
        import os
        import socket

        return f"{socket.gethostname()}:{os.getpid()}:work-item"


def _cas(
    engine: Any, node_id: str, conditions: dict[str, Any], updates: dict[str, Any]
) -> bool:
    """Route one atomic CAS to the engine's control-plane backend.

    Mirrors ``TaskManagerMixin._control`` (falls back to ``engine.backend``,
    then to ``engine`` itself for test fakes that implement the CAS directly).
    Raises :class:`WorkItemBackendUnavailable` — never silently no-ops — when
    no backend on the chain implements ``compare_and_set_node_fields``.
    """
    backend = getattr(engine, "_control", None)
    if backend is None:
        backend = getattr(engine, "backend", None)
    if backend is None:
        backend = engine
    fn = getattr(backend, "compare_and_set_node_fields", None)
    if not callable(fn):
        raise WorkItemBackendUnavailable(
            f"{type(engine).__name__} has no compare_and_set_node_fields — "
            "WorkItem requires an engine-native atomic CAS backend (AU-P0-3); "
            "see TaskManagerMixin._claim_next_task / research.loops.claim_loop "
            "for the same requirement on the existing Task/Loop claim paths."
        )
    return bool(fn(node_id, conditions, updates))


def _link(engine: Any, source_id: str, target_id: str, rel_type: str) -> None:
    """Best-effort edge write (never blocks a WorkItem transition on a graph-viz edge)."""
    try:
        linker = getattr(engine, "link_nodes", None) or getattr(
            engine, "add_edge", None
        )
        if callable(linker):
            linker(source_id, target_id, rel_type)
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "work_item: edge write %s -> %s (%s) failed: %s",
            source_id,
            target_id,
            rel_type,
            e,
        )


def _task_depends_on_edge_type() -> str:
    try:
        from agent_utilities.models.knowledge_graph import RegistryEdgeType

        return str(RegistryEdgeType.TASK_DEPENDS_ON)
    except Exception:  # noqa: BLE001
        return "task_depends_on"


# ── read ─────────────────────────────────────────────────────────────────


def get_work_item(engine: Any, item_id: str) -> dict[str, Any] | None:
    """Read one WorkItem's full property row, or ``None`` if it doesn't exist."""
    return_clause = ", ".join(f"w.{f} AS {f}" for f in _FIELDS)
    rows = engine.query_cypher(
        f"MATCH (w:{_NODE_LABEL} {{id: $id}}) RETURN w.id AS id, {return_clause}",
        {"id": item_id},
    )
    if not rows:
        return None
    row = dict(rows[0])
    row["depends_on"] = list(row.get("depends_on") or [])
    row["downstream_ids"] = list(row.get("downstream_ids") or [])
    return row


def tenant_in_flight_count(engine: Any, tenant: str) -> int:
    """Count of this tenant's non-terminal WorkItems (the per-tenant quota check)."""
    rows = engine.query_cypher(
        f"MATCH (w:{_NODE_LABEL} {{tenant: $tenant}}) WHERE NOT w.status IN $terminal "
        "RETURN count(w) AS c",
        {"tenant": tenant, "terminal": list(TERMINAL_WORK_ITEM_STATUSES)},
    )
    if not rows:
        return 0
    return int(rows[0].get("c") or 0)


# ── submit ───────────────────────────────────────────────────────────────


def submit_work_item(
    engine: Any,
    *,
    kind: str,
    payload_ref: str = "",
    tenant: str = "",
    depends_on: Sequence[str] = (),
    priority: Any = 2,
    deadline_unix: float | None = None,
    budget: float | None = None,
    resource_class: str = "default",
    fairness_group: str = "",
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
    idempotency_key: str | None = None,
    correlation_id: str | None = None,
    work_item_id: str | None = None,
    max_tenant_in_flight: int | None = None,
    now: float | None = None,
) -> str:
    """Submit one WorkItem; returns its id.

    Idempotent when ``work_item_id`` is caller-supplied and already exists
    (an upsert no-op — mirrors ``engine_tasks.submit_task``'s deterministic-
    ``job_id`` dedup pattern, e.g. for the scheduler's ``sched:<name>:<minute>``
    ids). No dependency ⇒ immediately ``ready``; any unresolved dependency ⇒
    ``submitted`` with ``dep_count`` set, released atomically as each parent
    succeeds (see :func:`_release_one_dependency`).
    """
    from agent_utilities.knowledge_graph.core.engine_tasks import _coerce_prio_bucket
    from agent_utilities.models.knowledge_graph import WorkItemNode

    now = now if now is not None else _now()
    item_id = work_item_id or new_work_item_id()

    if get_work_item(engine, item_id) is not None:
        return item_id  # idempotent upsert

    if tenant and max_tenant_in_flight is not None:
        in_flight = tenant_in_flight_count(engine, tenant)
        if in_flight >= max_tenant_in_flight:
            raise TenantQuotaExceeded(
                f"tenant {tenant!r} has {in_flight} in-flight WorkItems "
                f">= quota {max_tenant_in_flight}"
            )

    dep_ids = [d for d in dict.fromkeys(depends_on) if d]
    resolved_deps: list[str] = []
    dep_count = 0
    for dep_id in dep_ids:
        dep = get_work_item(engine, dep_id)
        if dep is None:
            # Dependency isn't (yet) a tracked WorkItem — conservatively block
            # on it (mirrors fleet_reconciler's conservative "missing dep is
            # NOT satisfied"); it simply won't be indexed for push-release
            # until it exists, so a caller must submit parents before/along
            # with children for the push-release path to fire.
            dep_count += 1
            resolved_deps.append(dep_id)
            continue
        resolved_deps.append(dep_id)
        if dep.get("status") != WorkItemStatus.SUCCEEDED.value:
            dep_count += 1

    status = WorkItemStatus.SUBMITTED if dep_count else WorkItemStatus.READY

    node = WorkItemNode(
        id=item_id,
        name=f"WorkItem: {kind}:{payload_ref or item_id}",
        tenant=tenant,
        kind=kind,
        status=status.value,
        payload_ref=payload_ref,
        idempotency_key=idempotency_key or item_id,
        depends_on=resolved_deps,
        dep_count=dep_count,
        prio_bucket=_coerce_prio_bucket(priority),
        deadline_unix=deadline_unix,
        budget=budget,
        resource_class=resource_class,
        fairness_group=fairness_group,
        max_attempts=max(1, int(max_attempts)),
        backoff_base_s=float(backoff_base_s),
        correlation_id=correlation_id
        if correlation_id is not None
        else _default_correlation_id(),
        created_at=now,
        updated_at=now,
        submitted_at=now,
    )
    engine.add_node(
        item_id, _NODE_LABEL, properties=node.model_dump(exclude={"id", "type"})
    )

    edge_type = _task_depends_on_edge_type()
    for dep_id in resolved_deps:
        if get_work_item(engine, dep_id) is None:
            continue  # untracked dep: nothing to index for push-release, still counted above
        _link(engine, item_id, dep_id, edge_type)
        _append_downstream(engine, dep_id, item_id, now=now)

    return item_id


def _append_downstream(
    engine: Any, parent_id: str, child_id: str, *, now: float | None = None
) -> None:
    """CAS-append ``child_id`` to the parent's reverse dependency index."""
    now = now if now is not None else _now()
    for _ in range(_CAS_LOOP_MAX_RETRIES):
        parent = get_work_item(engine, parent_id)
        if parent is None:
            return
        current = list(parent.get("downstream_ids") or [])
        if child_id in current:
            return
        updated = current + [child_id]
        won = _cas(
            engine,
            parent_id,
            {"downstream_ids": current},
            {"downstream_ids": updated, "updated_at": now},
        )
        if won:
            return
    logger.warning(
        "work_item: failed to index downstream %s -> %s after %d retries",
        parent_id,
        child_id,
        _CAS_LOOP_MAX_RETRIES,
    )


# ── claim / lease / fencing ─────────────────────────────────────────────


def claim_specific(
    engine: Any,
    item_id: str,
    *,
    token: str | None = None,
    now: float | None = None,
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
) -> dict[str, Any] | None:
    """CAS-claim ONE known ``ready`` WorkItem: ``ready -> leased(fencing_token)``.

    Returns ``None`` (never raises) when the item doesn't exist, is
    terminal, is still ``submitted`` (deps unresolved), or has a live lease
    held elsewhere. A stale (expired) lease is reclaimed in place first (the
    crash-recovery path — same reasoning as ``claim_agent_task``'s stale-lease
    re-claim) before the fresh claim is attempted.
    """
    token = token or _default_token()
    now = now if now is not None else _now()

    item = get_work_item(engine, item_id)
    if item is None:
        return None
    status = item.get("status")
    if status in TERMINAL_WORK_ITEM_STATUSES:
        return None
    if status in (WorkItemStatus.LEASED.value, WorkItemStatus.RUNNING.value):
        expires_at = float(item.get("lease_expires_at") or 0.0)
        if expires_at > now:
            return None  # live lease elsewhere
        _reclaim_stale(engine, item_id, item, now=now)
        item = get_work_item(engine, item_id)
        if item is None or item.get("status") != WorkItemStatus.READY.value:
            return None
        status = item.get("status")
    if status != WorkItemStatus.READY.value:
        return None  # submitted (deps unresolved) or a race changed it

    next_retry_at = item.get("next_retry_at")
    if next_retry_at is not None and float(next_retry_at) > now:
        return None  # backoff window not yet elapsed

    prior_epoch = int(item.get("lease_epoch") or 0)
    new_epoch = prior_epoch + 1
    attempt = int(item.get("attempt") or 0) + 1
    won = _cas(
        engine,
        item_id,
        {"status": WorkItemStatus.READY.value, "lease_epoch": prior_epoch},
        {
            "status": WorkItemStatus.LEASED.value,
            "lease_owner": token,
            "lease_epoch": new_epoch,
            "lease_expires_at": now + lease_ttl_s,
            "heartbeat_at": now,
            "attempt": attempt,
            "updated_at": now,
        },
    )
    if not won:
        return None
    return {
        "work_item_id": item_id,
        "kind": item.get("kind"),
        "payload_ref": item.get("payload_ref"),
        "depends_on": list(item.get("depends_on") or []),
        "lease_owner": token,
        "fence_token": new_epoch,
        "attempt": attempt,
        "max_attempts": int(item.get("max_attempts") or DEFAULT_MAX_ATTEMPTS),
    }


def _reclaim_stale(
    engine: Any, item_id: str, item: dict[str, Any], *, now: float
) -> None:
    """Best-effort: requeue an item whose lease has expired back to ``ready``.

    A CAS loss here just means someone else (another claimer or the reaper)
    already reclaimed it — :func:`claim_specific` re-reads afterward either
    way, so a lost race is harmless.
    """
    status = item.get("status")
    prior_epoch = int(item.get("lease_epoch") or 0)
    _cas(
        engine,
        item_id,
        {"status": status, "lease_epoch": prior_epoch},
        {
            "status": WorkItemStatus.READY.value,
            "lease_owner": None,
            "lease_expires_at": None,
            "heartbeat_at": None,
            "updated_at": now,
        },
    )


def _select_ready_candidates(
    engine: Any,
    prio_bucket: int,
    *,
    resource_class: str | None,
    tenant: str | None,
    fairness_group: str | None,
    now: float,
    limit: int,
) -> list[str]:
    rows = (
        engine.query_cypher(
            f"MATCH (w:{_NODE_LABEL} {{status: $status, prio_bucket: $bucket}}) "
            "RETURN w.id AS id, w.created_at AS created_at, w.next_retry_at AS next_retry_at, "
            "w.resource_class AS resource_class, w.tenant AS tenant, "
            "w.fairness_group AS fairness_group "
            f"LIMIT {int(limit) * 4}",
            {"status": WorkItemStatus.READY.value, "bucket": prio_bucket},
        )
        or []
    )
    filtered = []
    for row in rows:
        if resource_class is not None and row.get("resource_class") != resource_class:
            continue
        if tenant is not None and row.get("tenant") != tenant:
            continue
        if fairness_group is not None and row.get("fairness_group") != fairness_group:
            continue
        next_retry_at = row.get("next_retry_at")
        if next_retry_at is not None and float(next_retry_at) > now:
            continue
        filtered.append(row)
    filtered.sort(key=lambda r: float(r.get("created_at") or 0.0))
    return [r["id"] for r in filtered[: int(limit)] if r.get("id")]


def claim_next(
    engine: Any,
    *,
    resource_class: str | None = None,
    tenant: str | None = None,
    fairness_group: str | None = None,
    token: str | None = None,
    now: float | None = None,
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
    max_candidates: int = 25,
) -> dict[str, Any] | None:
    """Claim the next runnable WorkItem: bucket-ascending priority, oldest first.

    Mirrors ``engine_tasks._claim_next_task``'s bucket-ascending-candidate/
    CAS-retry shape; a candidate whose CAS loses just means a peer claimed it
    first — the loop moves to the next candidate.
    """
    now = now if now is not None else _now()
    for bucket in range(4):
        candidates = _select_ready_candidates(
            engine,
            bucket,
            resource_class=resource_class,
            tenant=tenant,
            fairness_group=fairness_group,
            now=now,
            limit=max_candidates,
        )
        for item_id in candidates:
            claim = claim_specific(
                engine, item_id, token=token, now=now, lease_ttl_s=lease_ttl_s
            )
            if claim is not None:
                return claim
    return None


def mark_running(
    engine: Any, item_id: str, claim: dict[str, Any], *, now: float | None = None
) -> bool:
    """CAS ``leased -> running``, fenced on the claim's epoch."""
    now = now if now is not None else _now()
    epoch = claim.get("fence_token")
    return _cas(
        engine,
        item_id,
        {"status": WorkItemStatus.LEASED.value, "lease_epoch": epoch},
        {
            "status": WorkItemStatus.RUNNING.value,
            "heartbeat_at": now,
            "updated_at": now,
        },
    )


def heartbeat(
    engine: Any,
    item_id: str,
    claim: dict[str, Any],
    *,
    now: float | None = None,
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
) -> bool:
    """Renew a ``running`` item's lease, fenced on the claim's epoch."""
    now = now if now is not None else _now()
    epoch = claim.get("fence_token")
    return _cas(
        engine,
        item_id,
        {"status": WorkItemStatus.RUNNING.value, "lease_epoch": epoch},
        {"heartbeat_at": now, "lease_expires_at": now + lease_ttl_s, "updated_at": now},
    )


def claim_and_start(
    engine: Any,
    item_id: str | None = None,
    *,
    resource_class: str | None = None,
    tenant: str | None = None,
    fairness_group: str | None = None,
    token: str | None = None,
    now: float | None = None,
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
) -> dict[str, Any] | None:
    """Convenience: claim (specific id, or the next runnable one) then mark it ``running``."""
    now = now if now is not None else _now()
    claim = (
        claim_specific(engine, item_id, token=token, now=now, lease_ttl_s=lease_ttl_s)
        if item_id
        else claim_next(
            engine,
            resource_class=resource_class,
            tenant=tenant,
            fairness_group=fairness_group,
            token=token,
            now=now,
            lease_ttl_s=lease_ttl_s,
        )
    )
    if claim is None:
        return None
    mark_running(engine, claim["work_item_id"], claim, now=now)
    return claim


# ── lease-expiry reaping (crash recovery) ───────────────────────────────


def reap_expired_leases(
    engine: Any, *, now: float | None = None, limit: int = 200
) -> dict[str, list[str]]:
    """Sweep leased/running items whose lease has expired.

    Requeues to ``ready`` (bumping the fencing epoch so a late-finishing dead
    holder's eventual commit is rejected) unless retries are already
    exhausted (``attempt >= max_attempts``), in which case the item is
    poisoned to ``dead_letter`` instead — the crash-recovery analogue of
    ``engine_tasks._tick_task_reaper``'s ``reaper_resets`` cap, but expressed
    on the SAME ``attempt`` counter :func:`commit_result` uses for app-level
    retries (a crashed worker and a worker that raised both count against the
    one attempt budget).
    """
    now = now if now is not None else _now()
    rows = (
        engine.query_cypher(
            f"MATCH (w:{_NODE_LABEL}) WHERE w.status IN $statuses AND "
            "w.lease_expires_at < $now RETURN w.id AS id "
            f"LIMIT {int(limit)}",
            {
                "statuses": [WorkItemStatus.LEASED.value, WorkItemStatus.RUNNING.value],
                "now": now,
            },
        )
        or []
    )
    reaped_ready: list[str] = []
    reaped_dead_letter: list[str] = []
    for row in rows:
        item_id = row.get("id")
        if not item_id:
            continue
        item = get_work_item(engine, item_id)
        if item is None:
            continue
        status = item.get("status")
        if status not in (WorkItemStatus.LEASED.value, WorkItemStatus.RUNNING.value):
            continue
        expires_at = float(item.get("lease_expires_at") or 0.0)
        if expires_at >= now:
            continue
        prior_epoch = int(item.get("lease_epoch") or 0)
        attempt = int(item.get("attempt") or 0)
        max_attempts = int(item.get("max_attempts") or DEFAULT_MAX_ATTEMPTS)
        if attempt >= max_attempts:
            won = _cas(
                engine,
                item_id,
                {"status": status, "lease_epoch": prior_epoch},
                {
                    "status": WorkItemStatus.DEAD_LETTER.value,
                    "error_ref": f"lease_expired_after_{attempt}_attempts",
                    "lease_owner": None,
                    "updated_at": now,
                    "completed_at": now,
                },
            )
            if won:
                reaped_dead_letter.append(item_id)
        else:
            won = _cas(
                engine,
                item_id,
                {"status": status, "lease_epoch": prior_epoch},
                {
                    "status": WorkItemStatus.READY.value,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "heartbeat_at": None,
                    "lease_epoch": prior_epoch
                    + 1,  # fence out the presumed-dead holder
                    "updated_at": now,
                },
            )
            if won:
                reaped_ready.append(item_id)
    return {"reaped_ready": reaped_ready, "reaped_dead_letter": reaped_dead_letter}


# ── result commit (idempotent, fenced) + atomic dependency release ─────


def _resolve_commit_conflict(engine: Any, item_id: str, epoch: Any) -> str:
    """Classify a lost commit CAS: already-terminal (idempotent no-op) vs fenced."""
    item = get_work_item(engine, item_id)
    if item is None:
        return "missing"
    status = item.get("status")
    if status in TERMINAL_WORK_ITEM_STATUSES:
        return "noop"  # a prior commit (this one or another) already landed
    live_epoch = item.get("lease_epoch")
    if epoch is not None and live_epoch is not None and int(live_epoch) != int(epoch):
        return "fenced"  # reclaimed by a newer holder while this one was finishing
    return "conflict"  # unexpected state change; caller should re-inspect


def commit_result(
    engine: Any,
    item_id: str,
    claim: dict[str, Any],
    *,
    outcome: str,
    result_ref: str | None = None,
    error_ref: str | None = None,
    retryable: bool = True,
    now: float | None = None,
) -> str:
    """Commit a claimed WorkItem's outcome: ``running -> succeeded|failed|dead_letter``,
    or cancel from any non-terminal status.

    ``outcome`` is one of ``"succeeded"``, ``"failed"``, ``"cancelled"``.
    Every branch's CAS precondition includes ``lease_epoch == claim["fence_token"]``
    (where applicable) — a stale holder whose lease was reclaimed gets a
    rejected commit (``"fenced"``), never a corrupted overwrite. A commit that
    finds the item ALREADY terminal (this is a redelivery of an
    already-committed outcome) is a no-op (``"noop"``) — idempotent by
    construction, since the CAS precondition ``status == "running"`` can only
    ever be satisfied once.

    Returns one of: ``"committed"`` (succeeded/cancelled/non-retryable-failed
    landed), ``"retry_scheduled"`` (a retryable failure went back to
    ``ready`` with a bumped fencing epoch and backoff), ``"dead_letter"``
    (retries exhausted), ``"noop"`` (idempotent double-commit), ``"fenced"``
    (stale holder rejected), ``"missing"`` (no such item).
    """
    now = now if now is not None else _now()
    epoch = claim.get("fence_token")
    item = get_work_item(engine, item_id)
    if item is None:
        return "missing"

    if outcome == WorkItemStatus.SUCCEEDED.value:
        won = _cas(
            engine,
            item_id,
            {"status": WorkItemStatus.RUNNING.value, "lease_epoch": epoch},
            {
                "status": WorkItemStatus.SUCCEEDED.value,
                "result_ref": result_ref,
                "error_ref": None,
                "lease_owner": None,
                "lease_expires_at": None,
                "updated_at": now,
                "completed_at": now,
            },
        )
        if won:
            _release_downstream(engine, item_id, now=now)
            return "committed"
        return _resolve_commit_conflict(engine, item_id, epoch)

    if outcome == WorkItemStatus.CANCELLED.value:
        for from_status in (
            WorkItemStatus.RUNNING.value,
            WorkItemStatus.LEASED.value,
            WorkItemStatus.READY.value,
            WorkItemStatus.SUBMITTED.value,
        ):
            conditions: dict[str, Any] = {"status": from_status}
            if from_status in (
                WorkItemStatus.RUNNING.value,
                WorkItemStatus.LEASED.value,
            ):
                conditions["lease_epoch"] = epoch
            won = _cas(
                engine,
                item_id,
                conditions,
                {
                    "status": WorkItemStatus.CANCELLED.value,
                    "error_ref": error_ref,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                    "completed_at": now,
                },
            )
            if won:
                return "committed"
        return _resolve_commit_conflict(engine, item_id, epoch)

    if outcome == WorkItemStatus.FAILED.value:
        if not retryable:
            won = _cas(
                engine,
                item_id,
                {"status": WorkItemStatus.RUNNING.value, "lease_epoch": epoch},
                {
                    "status": WorkItemStatus.FAILED.value,
                    "error_ref": error_ref,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                    "completed_at": now,
                },
            )
            if won:
                return "committed"
            return _resolve_commit_conflict(engine, item_id, epoch)

        attempt = int(item.get("attempt") or 0)
        max_attempts = int(item.get("max_attempts") or DEFAULT_MAX_ATTEMPTS)
        if attempt >= max_attempts:
            won = _cas(
                engine,
                item_id,
                {"status": WorkItemStatus.RUNNING.value, "lease_epoch": epoch},
                {
                    "status": WorkItemStatus.DEAD_LETTER.value,
                    "error_ref": error_ref,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "updated_at": now,
                    "completed_at": now,
                },
            )
            if won:
                return "dead_letter"
            return _resolve_commit_conflict(engine, item_id, epoch)

        backoff = float(item.get("backoff_base_s") or DEFAULT_BACKOFF_BASE_S) * (
            2 ** max(0, attempt - 1)
        )
        new_epoch = (int(epoch) + 1) if epoch is not None else item.get("lease_epoch")
        won = _cas(
            engine,
            item_id,
            {"status": WorkItemStatus.RUNNING.value, "lease_epoch": epoch},
            {
                "status": WorkItemStatus.READY.value,
                "error_ref": error_ref,
                "next_retry_at": now + backoff,
                "lease_owner": None,
                "lease_expires_at": None,
                "heartbeat_at": None,
                "lease_epoch": new_epoch,
                "updated_at": now,
            },
        )
        if won:
            return "retry_scheduled"
        return _resolve_commit_conflict(engine, item_id, epoch)

    raise ValueError(f"unknown work item outcome: {outcome!r}")


def cancel_work_item(
    engine: Any, item_id: str, *, reason: str = "", now: float | None = None
) -> bool:
    """Cancel a not-yet-terminal WorkItem regardless of its current status.

    Idempotent: cancelling an already-cancelled item returns ``True``;
    cancelling any OTHER terminal status returns ``False`` (cancellation
    cannot retroactively override a real succeeded/failed/dead_letter
    outcome).
    """
    now = now if now is not None else _now()
    for _ in range(_CAS_LOOP_MAX_RETRIES):
        item = get_work_item(engine, item_id)
        if item is None:
            return False
        status = item.get("status")
        if status in TERMINAL_WORK_ITEM_STATUSES:
            return status == WorkItemStatus.CANCELLED.value
        won = _cas(
            engine,
            item_id,
            {"status": status},
            {
                "status": WorkItemStatus.CANCELLED.value,
                "error_ref": reason,
                "lease_owner": None,
                "lease_expires_at": None,
                "updated_at": now,
                "completed_at": now,
            },
        )
        if won:
            return True
    return False


def _release_downstream(
    engine: Any, parent_id: str, *, now: float | None = None
) -> list[str]:
    """On a parent's success, atomically release every indexed downstream child."""
    now = now if now is not None else _now()
    parent = get_work_item(engine, parent_id)
    if parent is None:
        return []
    released: list[str] = []
    for child_id in list(parent.get("downstream_ids") or []):
        if _release_one_dependency(engine, child_id, now=now):
            released.append(child_id)
    return released


def _release_one_dependency(engine: Any, child_id: str, *, now: float) -> bool:
    """Atomically decrement one child's ``dep_count``; flip to ``ready`` at zero.

    The decrement and the ``submitted -> ready`` flip happen in the SAME CAS
    call — there is no observable window where the counter is at zero but
    the status hasn't caught up. Idempotent: a child no longer ``submitted``
    (already released by a concurrent parent, or terminal) is left alone.
    """
    for _ in range(_CAS_LOOP_MAX_RETRIES):
        child = get_work_item(engine, child_id)
        if child is None:
            return False
        if child.get("status") != WorkItemStatus.SUBMITTED.value:
            return False  # already released or terminal — nothing to do
        dep_count = int(child.get("dep_count") or 0)
        new_count = max(0, dep_count - 1)
        new_status = (
            WorkItemStatus.READY.value
            if new_count <= 0
            else WorkItemStatus.SUBMITTED.value
        )
        won = _cas(
            engine,
            child_id,
            {"status": WorkItemStatus.SUBMITTED.value, "dep_count": dep_count},
            {"status": new_status, "dep_count": new_count, "updated_at": now},
        )
        if won:
            return new_status == WorkItemStatus.READY.value
    logger.warning(
        "work_item: dependency release retries exhausted for child %s", child_id
    )
    return False


# ── AgentTask bridge — the MIGRATED dispatch path (AU-P1-1) ────────────
#
# Makes WorkItem authoritative for `:AgentTask` claim/execute/writeback when
# `AGENT_CLAIM_BACKEND=workitem` (engine_claim.py). Mirrors the legacy
# `:AgentTask.status` and a companion `:AgentLease` node (never reads them)
# so unmigrated consumers (`fleet_reconciler.fire_ready_agent_tasks`,
# dashboards) keep working unchanged.

_AGENT_TASK_ALREADY_TERMINAL = "completed"


def agent_task_work_item_id(task_id: str) -> str:
    """Deterministic 1:1 WorkItem id for a legacy ``:AgentTask`` id."""
    return f"workitem:agent_task:{task_id}"


def ensure_agent_task_work_item(
    engine: Any, task_id: str, *, now: float | None = None
) -> str | None:
    """Idempotently create/reuse the WorkItem shadowing legacy ``:AgentTask`` ``task_id``.

    Bridges DAGs written by ``TeamComposition.to_durable_task_dag`` (which
    only ever creates ``:AgentTask`` + ``TASK_DEPENDS_ON`` edges, never a
    WorkItem) onto the new state machine: a dependency already at legacy
    status ``'completed'`` needs no shadow (it doesn't block); one that's
    still in flight is recursively shadowed so WorkItem's own dep_count/
    release mechanics take over. Returns ``None`` when the legacy task is
    unknown (no such ``:AgentTask`` node — see the module docstring's
    FOLLOW-UP note: a dependency id that never resolves is a pre-existing
    data-integrity concern, not one this bridge newly introduces) or already
    ``'completed'``.
    """
    now = now if now is not None else _now()
    item_id = agent_task_work_item_id(task_id)
    if get_work_item(engine, item_id) is not None:
        return item_id

    rows = engine.query_cypher(
        "MATCH (t:AgentTask {id: $id}) RETURN t.status AS status, "
        "t.depends_on_task_ids AS depends_on_task_ids, t.dag_id AS dag_id, "
        "t.checkpoint_id AS checkpoint_id",
        {"id": task_id},
    )
    if not rows:
        logger.debug(
            "work_item bridge: no legacy AgentTask node for %s — nothing to shadow",
            task_id,
        )
        return None
    row = rows[0]
    if str(row.get("status") or "") == _AGENT_TASK_ALREADY_TERMINAL:
        return None  # already done — doesn't block, no shadow needed

    dep_task_ids = list(row.get("depends_on_task_ids") or [])
    parent_item_ids: list[str] = []
    for dep_task_id in dep_task_ids:
        dep_item_id = ensure_agent_task_work_item(engine, dep_task_id, now=now)
        if dep_item_id is not None:
            parent_item_ids.append(dep_item_id)

    submit_work_item(
        engine,
        kind="agent_task",
        payload_ref=task_id,
        depends_on=parent_item_ids,
        work_item_id=item_id,
        idempotency_key=item_id,
        now=now,
    )
    return item_id


def claim_agent_task_via_work_item(
    engine: Any,
    task_id: str,
    *,
    token: str | None = None,
    now: float | None = None,
    claim_ttl_s: float = DEFAULT_LEASE_TTL_S,
) -> dict[str, Any] | None:
    """Claim one ``:AgentTask`` through the WorkItem state machine (MIGRATED path).

    Same return contract as
    :func:`agent_utilities.orchestration.agent_dispatch_worker.claim_agent_task`
    (``task_id``/``lease_id``/``dag_id``/``checkpoint_id``/
    ``depends_on_task_ids``/``fence_token``), so every existing downstream
    consumer (``execute_agent_task_turn``, ``_fence_still_valid``) is
    unchanged — plus an internal ``_work_item_id`` key so
    ``_finalize_agent_task`` can additionally commit through
    :func:`commit_result` (dependency release, DLQ, idempotent commit all
    apply). Selected via ``AGENT_CLAIM_BACKEND=workitem``
    (:mod:`~agent_utilities.orchestration.engine_claim`).
    """
    token = token or _default_token()
    now = now if now is not None else _now()

    item_id = ensure_agent_task_work_item(engine, task_id, now=now)
    if item_id is None:
        return None
    claim = claim_specific(
        engine, item_id, token=token, now=now, lease_ttl_s=claim_ttl_s
    )
    if claim is None:
        return None
    mark_running(engine, item_id, claim, now=now)
    item = get_work_item(engine, item_id) or {}

    try:
        engine.add_node(task_id, "AgentTask", properties={"status": "running"})
    except Exception as e:  # noqa: BLE001 — mirror is best-effort
        logger.warning(
            "work_item bridge: legacy AgentTask status mirror failed for %s: %s",
            task_id,
            e,
        )

    lease_id = f"lease:{task_id}:{claim['fence_token']}"
    try:
        engine.add_node(
            lease_id,
            "AgentLease",
            properties={
                "name": f"Lease: {task_id}",
                "owner_token": claim["lease_owner"],
                "resource_id": task_id,
                "acquired_at": now,
                "lease_expires_at": now + claim_ttl_s,
                "lease_epoch": claim["fence_token"],
            },
        )
    except Exception as e:  # noqa: BLE001 — mirror is best-effort
        logger.warning(
            "work_item bridge: AgentLease mirror failed for %s: %s", task_id, e
        )

    dag_id = ""
    checkpoint_id = None
    try:
        legacy_rows = engine.query_cypher(
            "MATCH (t:AgentTask {id: $id}) RETURN t.dag_id AS dag_id, t.checkpoint_id AS checkpoint_id",
            {"id": task_id},
        )
        if legacy_rows:
            dag_id = legacy_rows[0].get("dag_id") or ""
            checkpoint_id = legacy_rows[0].get("checkpoint_id")
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "work_item bridge: legacy dag_id/checkpoint_id read failed for %s: %s",
            task_id,
            e,
        )

    return {
        "task_id": task_id,
        "lease_id": lease_id,
        "dag_id": dag_id,
        "checkpoint_id": checkpoint_id,
        "depends_on_task_ids": list(item.get("depends_on") or []),
        "fence_token": claim["fence_token"],
        "_work_item_id": item_id,
    }


_AGENT_TASK_OUTCOME_TO_WORK_ITEM: dict[str, tuple[str, bool]] = {
    "completed": (WorkItemStatus.SUCCEEDED.value, True),
    "failed": (WorkItemStatus.FAILED.value, True),
    "unroutable": (WorkItemStatus.FAILED.value, False),
    "denied": (WorkItemStatus.FAILED.value, False),
    "cancelled": (WorkItemStatus.CANCELLED.value, False),
    # "blocked" intentionally absent: the legacy task stays non-terminal
    # pending human approval; the WorkItem is left `running` and will
    # naturally re-enter `ready` via `reap_expired_leases` once its lease
    # expires, for a fresh retry after approval.
}


def commit_agent_task_work_item(
    engine: Any, work_item_id: str, claim: dict[str, Any], *, status: str
) -> str | None:
    """Commit an executed ``:AgentTask`` turn's outcome through :func:`commit_result`.

    ``status`` is one of ``execute_agent_task_turn``'s outcome strings
    (``completed``/``failed``/``unroutable``/``denied``/``cancelled``/
    ``blocked``). Returns the :func:`commit_result` outcome, or ``None`` when
    ``status`` has no WorkItem mapping (``blocked`` — see above). Never
    raises (a commit-mirror failure is logged, not propagated — the legacy
    ``:AgentTask`` write already recorded the authoritative-enough outcome
    for unmigrated readers).
    """
    mapping = _AGENT_TASK_OUTCOME_TO_WORK_ITEM.get(status)
    if mapping is None:
        return None
    outcome, retryable = mapping
    task_id = claim.get("task_id")
    try:
        return commit_result(
            engine,
            work_item_id,
            claim,
            outcome=outcome,
            retryable=retryable,
            result_ref=f"outcome:agent_task:{task_id}"
            if outcome == WorkItemStatus.SUCCEEDED.value
            else None,
            error_ref=f"agent_task:{task_id}:{status}"
            if outcome != WorkItemStatus.SUCCEEDED.value
            else None,
        )
    except Exception as e:  # noqa: BLE001 — mirror commit is best-effort
        logger.warning(
            "work_item bridge: commit_result failed for %s: %s", work_item_id, e
        )
        return None


# ── Goal/Loop + ingestion :Task shims (read-only, SHIMMED not migrated) ─


_LOOP_STATUS_TO_WORK_ITEM: dict[str, str] = {
    "pending": WorkItemStatus.READY.value,
    "running": WorkItemStatus.RUNNING.value,
    "completed": WorkItemStatus.SUCCEEDED.value,
    "failed": WorkItemStatus.FAILED.value,
    "cancelled": WorkItemStatus.CANCELLED.value,
    "rejected": WorkItemStatus.DEAD_LETTER.value,
}


def work_item_view_of_loop(engine: Any, loop_id: str) -> dict[str, Any] | None:
    """Read-only :class:`WorkItemStatus` projection of a Loop/goal Concept node.

    SHIM, not authoritative: ``submit_loop``/``claim_loop``/``mark_loop_status``/
    ``run_goal_loop`` are entirely UNCHANGED. This exists so dashboards/
    observability can show ONE status vocabulary across Goal/Task/AgentTask/
    Loop without a full Loop-storage migration (out of scope this session —
    see the module docstring).
    """
    rows = engine.query_cypher(
        "MATCH (c:Concept) WHERE c.id = $id RETURN c.id AS id, c.status AS status, "
        "c.updated_at AS updated_at",
        {"id": loop_id},
    )
    if not rows:
        return None
    row = rows[0]
    raw_status = str(row.get("status") or "pending")
    return {
        "work_item_id": loop_id,
        "kind": "goal_loop",
        "status": _LOOP_STATUS_TO_WORK_ITEM.get(
            raw_status, WorkItemStatus.SUBMITTED.value
        ),
        "native_status": raw_status,
        "updated_at": row.get("updated_at"),
        "shim": True,
    }


_TASK_STATUS_TO_WORK_ITEM: dict[str, str] = {
    "pending": WorkItemStatus.READY.value,
    "scheduled": WorkItemStatus.SUBMITTED.value,
    "blocked": WorkItemStatus.SUBMITTED.value,
    "running": WorkItemStatus.RUNNING.value,
    "completed": WorkItemStatus.SUCCEEDED.value,
    "failed": WorkItemStatus.FAILED.value,
    "cancelled": WorkItemStatus.CANCELLED.value,
    "dead_letter": WorkItemStatus.DEAD_LETTER.value,
}


def work_item_view_of_task(engine: Any, task_id: str) -> dict[str, Any] | None:
    """Read-only :class:`WorkItemStatus` projection of an ingestion ``:Task`` node.

    SHIM, not authoritative: ``engine_tasks.py``'s ``submit_task``/
    ``_claim_next_task``/``_fail_or_retry_task``/``_tick_task_reaper`` (lanes,
    admission gate, reaper, scheduled/blocked promotion sweep) are entirely
    UNCHANGED — that subsystem is a large, separate follow-up (see the
    module docstring).
    """
    rows = engine.query_cypher(
        "MATCH (t:Task {id: $id}) RETURN t.id AS id, t.status AS status",
        {"id": task_id},
    )
    if not rows:
        return None
    raw_status = str(rows[0].get("status") or "pending")
    return {
        "work_item_id": task_id,
        "kind": "ingest_task",
        "status": _TASK_STATUS_TO_WORK_ITEM.get(
            raw_status, WorkItemStatus.SUBMITTED.value
        ),
        "native_status": raw_status,
        "shim": True,
    }
