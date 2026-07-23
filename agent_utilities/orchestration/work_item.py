#!/usr/bin/python
from __future__ import annotations

"""The sole engine-native ``WorkItem`` state machine (AU-P1-1).

CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch — unifies Goal/Task/AgentTask/Loop/dispatch

Every durable unit of work uses this one versioned lifecycle::

    submitted -> ready -> leased(fencing_token) -> running(heartbeat,attempt)
        -> succeeded(result_ref) | failed(error_ref) | cancelled | dead_letter

``WorkItem`` is the only writable work-state record. Definitions and payloads
may live in immutable domain records, but no other node owns status, claim,
lease, retry, dependency, priority, or terminal outcome. Producers submit a
WorkItem at intake; consumers never adopt or infer state from another label.

Legacy status vocabularies (the ingestion ``:Task`` queue, ``:AgentTask``
dispatch, the team-collaboration ``:TaskNode``, Loop/Goal, the dispatch
envelope) are bridged onto this one state machine rather than replaced
outright: :func:`agent_task_work_item_id`/:func:`ensure_agent_task_work_item`/
:func:`claim_agent_task_via_work_item`/:func:`commit_agent_task_work_item`
mirror an ``:AgentTask``'s legacy status for unmigrated readers
(``fleet_reconciler``, dashboards); :func:`team_task_work_item_id`/
:func:`ensure_team_task_work_item`/:func:`start_team_task_work_item`/
:func:`team_task_status_view` do the same for a team ``:TaskNode``; and
:func:`work_item_view_of_task` is a read-only fallback view for a ``:Task``
that has no shadow yet.

Reuses, never reinvents:

* **Atomicity** — production claim/renew/commit/cancel/defer operations are
  dedicated engine-native transactions. Dependency-injected tests use a graph
  CAS harness.
* **Lease/fencing** — every mutation validates tenant, worker, epoch, and
  fencing token in the engine.
* **Idempotent commit** — the native transaction accepts an idempotency key and
  a matching fence; a repeated terminal commit resolves to a no-op.
* **Atomic dependency release** — successful native commit releases indexed
  downstream WorkItems in the same storage transaction.
"""

import contextvars
import logging
import re
import time
import uuid
from collections.abc import Sequence
from typing import Any

from agent_utilities.protocols.epistemic_operations import (
    ClaimWorkItemRequest,
    ClaimWorkItemResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAX_TENANT_IN_FLIGHT",
    "MAX_TENANT_IN_FLIGHT",
    "WORK_ITEM_STATES",
    "TERMINAL_WORK_ITEM_STATUSES",
    "WorkItemBackendUnavailable",
    "NativeWorkItemRequired",
    "TenantQuotaExceeded",
    "new_work_item_id",
    "submit_work_item",
    "get_work_item",
    "claim_specific",
    "claim_next",
    "current_work_item_claim",
    "mark_running",
    "heartbeat",
    "checkpoint_work_item",
    "commit_result",
    "cancel_work_item",
    "defer_work_item",
    "reap_expired_leases",
    "tenant_in_flight_count",
    "execution_work_item_id",
    "claim_execution_work_item",
    "commit_execution_work_item",
    "agent_task_work_item_id",
    "ensure_agent_task_work_item",
    "claim_agent_task_via_work_item",
    "commit_agent_task_work_item",
    "ingest_task_work_item_id",
    "ingest_task_job_id_from_work_item_id",
    "ensure_ingest_task_work_item",
    "claim_ingest_task_work_item",
    "team_task_work_item_id",
    "ensure_team_task_work_item",
    "start_team_task_work_item",
    "team_task_status_view",
    "work_item_view_of_task",
    "orchestrator_work_item_id",
    "submit_orchestrator_work_item",
    "team_work_item_id",
    "submit_team_work_item",
    "start_team_work_item",
    "loop_work_item_id",
    "ensure_loop_work_item",
    "claim_loop_work_item",
    "transition_loop_work_item",
    "set_loop_statechart_instance_id",
    "set_work_item_priority",
    "work_item_view_of_loop",
]

DEFAULT_MAX_TENANT_IN_FLIGHT = 64
MAX_TENANT_IN_FLIGHT = 4096

#: The eight states of the ONE work-item lifecycle (AU-P1-1) — a closed
#: vocabulary of plain strings (W2.5 control-plane migration: no enum, no
#: protection type, just data, agreeing with the eg-statechart-driven Loop
#: chart's own closed vocabulary). Every former ``WorkItemStatus.X.value`` (or
#: bare ``WorkItemStatus.X`` — a ``StrEnum`` member compares equal to its
#: plain string, so both read identically at every call site) is now the
#: literal lowercase string directly.
WORK_ITEM_STATES: tuple[str, ...] = (
    "submitted",
    "ready",
    "leased",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "dead_letter",
)

TERMINAL_WORK_ITEM_STATUSES = frozenset(
    {
        "succeeded",
        "failed",
        "cancelled",
        "dead_letter",
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
_PROCESS_WORKER_TOKEN = f"worker:{uuid.uuid4().hex}"
_loop_claims: contextvars.ContextVar[dict[str, dict[str, Any]] | None] = (
    contextvars.ContextVar("work_item_loop_claims", default=None)
)

#: Fields returned by :func:`get_work_item` (everything but ``id``, which is
#: always requested separately so the row is never empty on a hit).
_FIELDS: tuple[str, ...] = (
    "status",
    "kind",
    "queue",
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
    "fencing_token",
    "lease_expires_at",
    "heartbeat_at",
    "correlation_id",
    "created_at",
    "updated_at",
    "submitted_at",
    "completed_at",
    "result_ref",
    "error_ref",
    "description",
    "dag_id",
    "checkpoint_id",
    "assigned_to",
    "created_by",
    "metadata",
    "loop_statechart_instance_id",
)


class WorkItemBackendUnavailable(RuntimeError):
    """Raised when the connected engine has no atomic ``compare_and_set_node_fields``.

    Fails LOUD rather than silently degrading to a non-atomic read-then-write
    — a WorkItem transition without a real CAS backing it is not a safe
    claim/commit, and AU-P0-3's fail-closed discipline forbids a fabricated
    success in that situation.
    """


class NativeWorkItemRequired(WorkItemBackendUnavailable):
    """Raised when production cannot reach the engine-native WorkItem verbs."""


def _work_tenant(value: str | None = None) -> str:
    """Resolve the non-empty tenant key required by native WorkItem verbs."""
    if value:
        return str(value)
    from agent_utilities.knowledge_graph.core.session import resolve_session

    session = resolve_session(required_scope="kg:write")
    tenant = str(session.tenant or session.graph or "").strip()
    if not tenant:
        raise NativeWorkItemRequired("WorkItem requires a non-empty tenant")
    return tenant


def _authority(engine: Any) -> Any:
    """Resolve exactly one WorkItem authority.

    Host engines expose their native control view as ``_work_item_engine``.
    Already-scoped adapters and generated clients are the authority themselves.
    No content backend or alternate client is searched.
    """
    view = getattr(engine, "_work_item_engine", None)
    return view if view is not None else engine


def _native_method(engine: Any, name: str) -> Any | None:
    """Resolve one generated verb on the sole WorkItem authority."""
    method = getattr(_authority(engine), name, None)
    return method if callable(method) else None


def _native_call(engine: Any, name: str, request: Any) -> Any:
    method = _native_method(engine, name)
    if method is None:
        raise NativeWorkItemRequired(f"engine-native {name}(request) is required")
    try:
        return method(request)
    except NotImplementedError as exc:
        raise NativeWorkItemRequired(
            f"engine-native {name}(request) is unavailable"
        ) from exc


def _claim_request(
    item: dict[str, Any] | None,
    *,
    item_id: str | None,
    queue: str | None,
    tenant: str | None,
    resource_class: str | None,
    fairness_group: str | None,
    token: str,
    now: float,
    lease_ttl_s: float,
) -> ClaimWorkItemRequest:
    row = item or {}
    return ClaimWorkItemRequest(
        schema_version="1",
        work_item_id=item_id,
        tenant_ref=_work_tenant(
            tenant if tenant is not None else str(row.get("tenant") or "")
        ),
        # Exact-id claims intentionally leave the queue filter empty. This
        # permits one-time adoption of pre-queue WorkItems without weakening
        # pool selection, where callers always pass the queue explicitly.
        queue_ref=(queue or None),
        resource_class=(
            resource_class
            if resource_class is not None
            else str(row.get("resource_class") or "default")
        ),
        fairness_group=(
            fairness_group
            if fairness_group is not None
            else str(row.get("fairness_group") or "") or None
        ),
        worker_ref=token,
        now_ms=max(0, int(now * 1000)),
        lease_ms=max(1, int(lease_ttl_s * 1000)),
        max_tenant_in_flight=DEFAULT_MAX_TENANT_IN_FLIGHT,
    )


def _normalize_native_claim(raw: Any) -> dict[str, Any] | None:
    try:
        result = ClaimWorkItemResult.model_validate(raw)
    except (TypeError, ValueError) as exc:
        raise WorkItemBackendUnavailable(
            "native ClaimWorkItem returned an invalid result"
        ) from exc
    if not result.claimed:
        return None
    resolved_id = result.work_item_id
    lease_epoch = result.lease_epoch
    fencing_token = result.fencing_token
    if (
        not resolved_id
        or not result.lease_holder_ref
        or lease_epoch is None
        or fencing_token is None
        or result.attempt is None
        or result.max_attempts is None
    ):
        raise WorkItemBackendUnavailable(
            "native ClaimWorkItem returned incomplete lease authority"
        )
    return {
        "work_item_id": str(resolved_id),
        "kind": result.kind,
        "tenant": None,
        "payload_ref": result.payload_ref,
        "depends_on": [],
        "lease_owner": result.lease_holder_ref,
        "fence_token": lease_epoch,
        "lease_epoch": lease_epoch,
        "fencing_token": fencing_token,
        "attempt": int(result.attempt),
        "max_attempts": int(result.max_attempts),
        "_native": True,
    }


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
        return _PROCESS_WORKER_TOKEN


def _cas(
    engine: Any, node_id: str, conditions: dict[str, Any], updates: dict[str, Any]
) -> bool:
    """Route one atomic metadata CAS to the sole WorkItem authority."""
    authority = _authority(engine)
    fn = getattr(authority, "compare_and_set_node_fields", None)
    if not callable(fn):
        raise WorkItemBackendUnavailable(
            f"{type(authority).__name__} has no compare_and_set_node_fields — "
            "WorkItem scheduling metadata requires an engine-native atomic "
            "CAS backend (AU-P0-3)."
        )
    return bool(fn(node_id, conditions, updates))


def _link(engine: Any, source_id: str, target_id: str, rel_type: str) -> None:
    """Best-effort edge write (never blocks a WorkItem transition on a graph-viz edge)."""
    try:
        linker = getattr(_authority(engine), "link_nodes", None)
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
    rows = _authority(engine).query_cypher(
        f"MATCH (w:{_NODE_LABEL} {{id: $id}}) RETURN w.id AS id, {return_clause}",
        {"id": item_id},
    )
    if not rows:
        return None
    row = dict(rows[0])
    row["depends_on"] = list(row.get("depends_on") or [])
    row["downstream_ids"] = list(row.get("downstream_ids") or [])
    if not isinstance(row.get("metadata"), dict):
        row["metadata"] = {}
    return row


def tenant_in_flight_count(engine: Any, tenant: str) -> int:
    """Count of this tenant's non-terminal WorkItems (the per-tenant quota check)."""
    rows = _authority(engine).query_cypher(
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
    queue: str | None = None,
    payload_ref: str = "",
    tenant: str = "",
    depends_on: Sequence[str] = (),
    priority: Any = 2,
    deadline_unix: float | None = None,
    available_at: float | None = None,
    budget: float | None = None,
    resource_class: str = "default",
    fairness_group: str = "",
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
    idempotency_key: str | None = None,
    correlation_id: str | None = None,
    description: str = "",
    dag_id: str = "",
    checkpoint_id: str | None = None,
    assigned_to: str = "",
    created_by: str = "",
    metadata: dict[str, Any] | None = None,
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
    succeeds in the engine-native commit transaction.
    """
    from agent_utilities.knowledge_graph.core.engine_tasks import _coerce_prio_bucket
    from agent_utilities.models.knowledge_graph import WorkItemNode
    from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

    now = now if now is not None else _now()
    tenant = _work_tenant(tenant)
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
    # Which deps resolved to a tracked WorkItem in this single snapshot — reused
    # by the edge-indexing loop below instead of re-fetching each one, so the
    # created node's dep_count and its dependency edges derive from one
    # consistent read (and N get_work_item() round-trips are removed).
    tracked_dep_ids: set[str] = set()
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
        tracked_dep_ids.add(dep_id)
        if dep.get("status") != "succeeded":
            dep_count += 1

    status = "submitted" if dep_count else "ready"
    privacy_guard = PersistencePrivacyGuard()
    clean_description, _privacy_report = privacy_guard.sanitize_text(description)
    clean_metadata, _metadata_privacy_report = privacy_guard.sanitize(
        dict(metadata or {})
    )
    if not isinstance(clean_metadata, dict):
        raise WorkItemBackendUnavailable(
            "WorkItem metadata privacy normalization failed"
        )

    node = WorkItemNode(
        id=item_id,
        # Human/local payload values do not belong in display fields. The
        # opaque ``payload_ref`` remains a machine reference; ``name`` exposes
        # only the non-identifying work kind.
        name=f"WorkItem: {kind}",
        tenant=tenant,
        kind=kind,
        queue=queue or kind,
        status=status,
        payload_ref=payload_ref,
        idempotency_key=idempotency_key or item_id,
        depends_on=resolved_deps,
        dep_count=dep_count,
        prio_bucket=_coerce_prio_bucket(priority),
        deadline_unix=deadline_unix,
        next_retry_at=available_at,
        budget=budget,
        resource_class=resource_class,
        fairness_group=fairness_group,
        max_attempts=max(1, int(max_attempts)),
        backoff_base_s=float(backoff_base_s),
        correlation_id=correlation_id
        if correlation_id is not None
        else _default_correlation_id(),
        description=clean_description,
        dag_id=dag_id,
        checkpoint_id=checkpoint_id,
        assigned_to=assigned_to,
        created_by=created_by,
        metadata=clean_metadata,
        created_at=now,
        updated_at=now,
        submitted_at=now,
    )
    _authority(engine).add_node(
        item_id, _NODE_LABEL, properties=node.model_dump(exclude={"id", "type"})
    )

    edge_type = _task_depends_on_edge_type()
    for dep_id in resolved_deps:
        if dep_id not in tracked_dep_ids:
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
    """Natively claim one known ``ready`` WorkItem.

    Returns ``None`` when the item doesn't exist, is terminal, is still
    ``submitted`` (deps unresolved), or has a live lease held elsewhere. A
    malformed or unauthorizable item fails closed. Expired-lease recovery is
    part of the engine-native claim transaction.
    """
    token = token or _default_token()
    now = now if now is not None else _now()

    item = get_work_item(engine, item_id)
    if item is None:
        return None
    if not item.get("tenant"):
        raise WorkItemBackendUnavailable(
            f"WorkItem {item_id!r} has no tenant and is not claimable"
        )
    native = _native_call(
        engine,
        "claim_work_item",
        _claim_request(
            item,
            item_id=item_id,
            queue=None,
            tenant=None,
            resource_class=None,
            fairness_group=None,
            token=token,
            now=now,
            lease_ttl_s=lease_ttl_s,
        ),
    )
    # A negative response is authoritative. There is no scan/CAS fallback.
    return _normalize_native_claim(native)


def claim_next(
    engine: Any,
    *,
    resource_class: str | None = None,
    queue: str = "",
    tenant: str | None = None,
    fairness_group: str | None = None,
    token: str | None = None,
    now: float | None = None,
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
    max_candidates: int = 25,
) -> dict[str, Any] | None:
    """Natively claim the next runnable WorkItem.

    ``max_candidates`` remains an API-level bound for callers but native
    selection owns ordering, quota, dependency release, and lease recovery.
    """
    token = token or _default_token()
    now = now if now is not None else _now()
    native = _native_call(
        engine,
        "claim_work_item",
        _claim_request(
            None,
            item_id=None,
            queue=queue,
            tenant=tenant,
            resource_class=resource_class,
            fairness_group=fairness_group,
            token=token,
            now=now,
            lease_ttl_s=lease_ttl_s,
        ),
    )
    return _normalize_native_claim(native)


def mark_running(
    engine: Any, item_id: str, claim: dict[str, Any], *, now: float | None = None
) -> bool:
    """Validate that the claim came from the native WorkItem transaction."""
    if claim.get("_native"):
        # ClaimWorkItem returns a live renewable lease; "leased" and
        # "running" are one engine-native ownership decision.
        return True
    raise NativeWorkItemRequired(
        f"WorkItem {item_id!r} was not claimed by ClaimWorkItem"
    )


def current_work_item_claim(engine: Any, item_id: str) -> dict[str, Any] | None:
    """Return a current lease tuple for an authorized resumed worker call.

    This is not an ownership grant: renew/commit still validates tenant,
    owner, epoch, fencing token, and lease expiry atomically in the engine.
    """
    if _native_method(engine, "claim_work_item") is None:
        raise NativeWorkItemRequired(
            "engine-native claim_work_item(request) is required"
        )
    item = get_work_item(engine, item_id)
    if item is None or item.get("status") not in {
        "leased",
        "running",
    }:
        return None
    required = ("lease_owner", "lease_epoch", "fencing_token")
    if any(item.get(field) is None for field in required):
        return None
    return {
        "work_item_id": item_id,
        "tenant": item.get("tenant"),
        "lease_owner": item.get("lease_owner"),
        "lease_epoch": item.get("lease_epoch"),
        "fence_token": item.get("lease_epoch"),
        "fencing_token": item.get("fencing_token"),
        "attempt": item.get("attempt"),
        "max_attempts": item.get("max_attempts"),
        "_native": True,
    }


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
    epoch = claim.get("lease_epoch", claim.get("fence_token"))
    fencing_token = claim.get("fencing_token", claim.get("fence_token"))
    item = get_work_item(engine, item_id) or {}
    native = _native_call(
        engine,
        "renew_work_item_lease",
        {
            "work_item_id": item_id,
            "tenant": str(claim.get("tenant") or item.get("tenant") or ""),
            "worker_ref": claim.get("lease_owner") or _default_token(),
            "expected_epoch": epoch,
            "fencing_token": fencing_token,
            "now_unix": now,
            "lease_ttl": lease_ttl_s,
            "idempotency_key": f"renew:{item_id}:{epoch}:{int(now)}",
        },
    )
    if isinstance(native, dict):
        return bool(native.get("renewed", native.get("ok", False)))
    return bool(native)


def checkpoint_work_item(
    engine: Any,
    item_id: str,
    claim: dict[str, Any],
    checkpoint_id: str,
    *,
    now: float | None = None,
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
) -> bool:
    """Persist one opaque progress checkpoint under the live native lease.

    The WorkItem remains the sole state authority: this renews the engine-issued
    lease first, then atomically fences the checkpoint update on status, tenant,
    owner, epoch, and fencing token.  No SQLite/Postgres checkpoint sidecar is
    consulted, and a lost lease can never advance progress.
    """

    if not isinstance(checkpoint_id, str) or not re.fullmatch(
        r"checkpoint:[a-z0-9_.:-]{1,240}", checkpoint_id
    ):
        raise ValueError("checkpoint_id must be an opaque checkpoint reference")
    now = now if now is not None else _now()
    if not heartbeat(
        engine,
        item_id,
        claim,
        now=now,
        lease_ttl_s=lease_ttl_s,
    ):
        return False
    item = get_work_item(engine, item_id)
    if item is None or item.get("status") not in {
        "leased",
        "running",
    }:
        return False
    epoch = claim.get("lease_epoch", claim.get("fence_token"))
    fencing_token = claim.get("fencing_token", claim.get("fence_token"))
    conditions = {
        "status": item.get("status"),
        "tenant": item.get("tenant"),
        "lease_owner": claim.get("lease_owner") or _default_token(),
        "lease_epoch": epoch,
        "fencing_token": fencing_token,
    }
    return _cas(
        engine,
        item_id,
        conditions,
        {"checkpoint_id": checkpoint_id, "updated_at": now},
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
    """Confirm native lease recovery is available.

    ``ClaimWorkItem`` reclaims expired leases atomically during selection. A
    Python scan/CAS reaper would be a second work-state authority and is not
    part of the current contract.
    """
    if _native_method(engine, "claim_work_item") is None:
        raise NativeWorkItemRequired(
            "engine-native claim_work_item(request) is required"
        )
    return {"reaped_ready": [], "reaped_dead_letter": []}


# ── result commit (idempotent, fenced) + atomic dependency release ─────


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
    The engine validates the lease epoch, fencing token, tenant, owner, and
    idempotency key in one durable transaction. No Python transition writer or
    dependency-release fallback exists.

    Returns one of: ``"committed"`` (succeeded/cancelled/non-retryable-failed
    landed), ``"retry_scheduled"`` (a retryable failure went back to
    ``ready`` with a bumped fencing epoch and backoff), ``"dead_letter"``
    (retries exhausted), ``"noop"`` (idempotent double-commit), ``"fenced"``
    (stale holder rejected), ``"missing"`` (no such item).
    """
    now = now if now is not None else _now()
    epoch = claim.get("lease_epoch", claim.get("fence_token"))
    fencing_token = claim.get("fencing_token", claim.get("fence_token"))
    item = get_work_item(engine, item_id)
    if item is None:
        return "missing"
    native = _native_call(
        engine,
        "commit_work_item_result",
        {
            "work_item_id": item_id,
            "tenant": str(item.get("tenant") or ""),
            "worker_ref": claim.get("lease_owner") or _default_token(),
            "expected_epoch": epoch,
            "fencing_token": fencing_token,
            "outcome": outcome,
            "result_ref": result_ref,
            "error_ref": error_ref,
            "retryable": bool(retryable),
            "now_unix": now,
            "idempotency_key": (
                f"commit:{item_id}:{epoch}:{outcome}:{result_ref or error_ref or ''}"
            ),
        },
    )
    if not isinstance(native, dict):
        return "committed" if native else "conflict"
    result = str(native.get("status") or native.get("result") or "").lower()
    aliases = {
        "succeeded": "committed",
        "failed": "committed",
        "cancelled": "committed",
        "deadletter": "dead_letter",
        "dead-letter": "dead_letter",
        "requeued": "retry_scheduled",
        "retry": "retry_scheduled",
        "already_committed": "noop",
    }
    result = aliases.get(result, result)
    if result not in {
        "committed",
        "retry_scheduled",
        "dead_letter",
        "noop",
        "fenced",
        "missing",
        "conflict",
    }:
        raise WorkItemBackendUnavailable(
            f"native CommitWorkItemResult returned unknown status {result!r}"
        )
    return result


def cancel_work_item(
    engine: Any,
    item_id: str,
    *,
    reason: str = "",
    claim: dict[str, Any] | None = None,
    now: float | None = None,
) -> bool:
    """Cancel a not-yet-terminal WorkItem regardless of its current status.

    Idempotent: cancelling an already-cancelled item returns ``True``;
    cancelling any OTHER terminal status returns ``False`` (cancellation
    cannot retroactively override a real succeeded/failed/dead_letter
    outcome).
    """
    now = now if now is not None else _now()
    item = get_work_item(engine, item_id)
    if item is None:
        return False
    native = _native_call(
        engine,
        "cancel_work_item",
        {
            "work_item_id": item_id,
            "tenant": str(item.get("tenant") or ""),
            "idempotency_key": f"cancel:{item_id}",
            "reason_ref": reason or None,
            "now_unix": now,
        },
    )
    if not isinstance(native, dict):
        return bool(native)
    status = str(native.get("status") or "").lower()
    if status == "cancelled":
        return True
    if status == "noop":
        latest = get_work_item(engine, item_id)
        return bool(latest and latest.get("status") == "cancelled")
    if status in {"missing", "in_flight", "not_cancellable"}:
        return False
    raise WorkItemBackendUnavailable(
        f"native CancelWorkItem returned unknown status {status!r}"
    )


def defer_work_item(
    engine: Any,
    item_id: str,
    claim: dict[str, Any],
    *,
    next_retry_at: float,
    reason_ref: str = "",
    now: float | None = None,
) -> bool:
    """Release a fenced lease for later polling without consuming an attempt."""
    now = now if now is not None else _now()
    if next_retry_at < now:
        raise ValueError("next_retry_at must not precede now")
    item = get_work_item(engine, item_id)
    if item is None:
        return False
    epoch = claim.get("lease_epoch", claim.get("fence_token"))
    fencing_token = claim.get("fencing_token", claim.get("fence_token"))
    native = _native_call(
        engine,
        "defer_work_item",
        {
            "work_item_id": item_id,
            "tenant": str(claim.get("tenant") or item.get("tenant") or ""),
            "worker_ref": claim.get("lease_owner") or _default_token(),
            "expected_epoch": epoch,
            "fencing_token": fencing_token,
            "idempotency_key": f"defer:{item_id}:{epoch}:{int(next_retry_at)}",
            "next_retry_at": next_retry_at,
            "reason_ref": reason_ref or None,
            "now_unix": now,
        },
    )
    if not isinstance(native, dict):
        return bool(native)
    status = str(native.get("status") or "").lower()
    if status == "deferred":
        return True
    if status in {"missing", "fenced"}:
        return False
    raise WorkItemBackendUnavailable(
        f"native DeferWorkItem returned unknown status {status!r}"
    )


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
    "completed": ("succeeded", True),
    "failed": ("failed", True),
    "unroutable": ("failed", False),
    "denied": ("failed", False),
    "cancelled": ("cancelled", False),
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
            if outcome == "succeeded"
            else None,
            error_ref=f"agent_task:{task_id}:{status}"
            if outcome != "succeeded"
            else None,
        )
    except Exception as e:  # noqa: BLE001 — mirror commit is best-effort
        logger.warning(
            "work_item bridge: commit_result failed for %s: %s", work_item_id, e
        )
        return None


# ── executable WorkItems ──────────────────────────────────────────────


def execution_work_item_id(dag_id: str, step_id: str) -> str:
    """Return the deterministic WorkItem id for one durable DAG step."""
    return f"workitem:execution:{dag_id}:{step_id}"


def claim_execution_work_item(
    engine: Any,
    item_id: str,
    *,
    token: str | None = None,
    now: float | None = None,
    claim_ttl_s: float = DEFAULT_LEASE_TTL_S,
) -> dict[str, Any] | None:
    """Claim an existing executable WorkItem; never infer or adopt state."""
    token = token or _default_token()
    now = now if now is not None else _now()
    item = get_work_item(engine, item_id)
    if item is None or item.get("kind") != "agent_execution":
        return None
    claim = claim_specific(
        engine, item_id, token=token, now=now, lease_ttl_s=claim_ttl_s
    )
    if claim is None:
        return None
    if not mark_running(engine, item_id, claim, now=now):
        return None
    return {
        **claim,
        "lease_id": f"workitem-lease:{item_id}:{claim['fence_token']}",
        "dag_id": str(item.get("dag_id") or ""),
        "checkpoint_id": item.get("checkpoint_id"),
        "depends_on": list(item.get("depends_on") or []),
    }


_EXECUTION_OUTCOME_TO_WORK_ITEM: dict[str, tuple[str, bool]] = {
    "completed": ("succeeded", True),
    "failed": ("failed", True),
    "unroutable": ("failed", False),
    "denied": ("failed", False),
    "cancelled": ("cancelled", False),
}


def commit_execution_work_item(
    engine: Any, work_item_id: str, claim: dict[str, Any], *, status: str
) -> str | None:
    """Commit an executable WorkItem outcome through :func:`commit_result`.

    ``status`` is one of ``execute_work_item_turn``'s outcome strings
    (``completed``/``failed``/``unroutable``/``denied``/``cancelled``/
    ``blocked``). A queued-for-approval result uses the native fenced defer
    operation, releasing its lease for a later retry without consuming a
    failure attempt. A rejected native transition cannot be replaced by any
    caller-side write.
    """
    if status == "blocked":
        return (
            "deferred"
            if defer_work_item(
                engine,
                work_item_id,
                claim,
                next_retry_at=_now() + 60.0,
                reason_ref=f"work_item:{work_item_id}:approval_pending",
            )
            else "fenced"
        )
    mapping = _EXECUTION_OUTCOME_TO_WORK_ITEM.get(status)
    if mapping is None:
        return None
    outcome, retryable = mapping
    return commit_result(
        engine,
        work_item_id,
        claim,
        outcome=outcome,
        retryable=retryable,
        result_ref=f"outcome:work_item:{work_item_id}"
        if outcome == "succeeded"
        else None,
        error_ref=f"work_item:{work_item_id}:{status}"
        if outcome != "succeeded"
        else None,
    )


# ── ingestion :Task bridge — MIGRATED (AU-P1-CL) ───────────────────────
#
# Makes WorkItem authoritative for the ingestion queue's claim/commit/reap
# arbitration (``engine_tasks.TaskManagerMixin``). Mirrors the SAME shape as
# the AgentTask bridge above. New ingestion submissions create the WorkItem
# at intake (the ingestion queue's ``submit_task`` calls
# :func:`ensure_ingest_task_work_item` eagerly);
# :func:`claim_ingest_task_work_item` only claims an already-shadowed item.


_INGEST_TASK_WORK_ITEM_PREFIX = "workitem:ingest_task:"


def ingest_task_work_item_id(job_id: str) -> str:
    """Deterministic 1:1 WorkItem id for an ingestion job id."""
    return f"{_INGEST_TASK_WORK_ITEM_PREFIX}{job_id}"


def ingest_task_job_id_from_work_item_id(work_item_id: str) -> str | None:
    """Invert :func:`ingest_task_work_item_id`, or ``None`` if not one of ours.

    The inverse is used to render the public job id from the sole WorkItem
    (e.g. by the reaper, mapping a :func:`reap_expired_leases` result back
    to the ingestion job it needs to report on).
    """
    if work_item_id.startswith(_INGEST_TASK_WORK_ITEM_PREFIX):
        return work_item_id[len(_INGEST_TASK_WORK_ITEM_PREFIX) :]
    return None


def ensure_ingest_task_work_item(
    engine: Any,
    job_id: str,
    *,
    prio_bucket: int = 2,
    resource_class: str = "default",
    fairness_group: str = "",
    tenant: str = "",
    depends_on: Sequence[str] = (),
    available_at: float | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
    metadata: dict[str, Any] | None = None,
    now: float | None = None,
) -> str:
    """Idempotently create/reuse the sole WorkItem for an ingestion job.

    ``prio_bucket``/``resource_class``/``fairness_group`` mirror the
    ingestion queue's own lane/priority/admission at submission time, so
    they're visible on the authoritative WorkItem too.
    """
    item_id = ingest_task_work_item_id(job_id)
    return submit_work_item(
        engine,
        kind="ingest_task",
        payload_ref=job_id,
        tenant=tenant,
        depends_on=[ingest_task_work_item_id(dep) for dep in depends_on],
        priority=prio_bucket,
        available_at=available_at,
        resource_class=resource_class,
        fairness_group=fairness_group,
        max_attempts=max_attempts,
        backoff_base_s=backoff_base_s,
        metadata=metadata,
        work_item_id=item_id,
        idempotency_key=item_id,
        now=now,
    )


def claim_ingest_task_work_item(
    engine: Any,
    job_id: str,
    *,
    token: str | None = None,
    now: float | None = None,
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
) -> dict[str, Any] | None:
    """Claim an existing ingestion WorkItem; never create or adopt one."""
    now = now if now is not None else _now()
    item_id = ingest_task_work_item_id(job_id)
    item = get_work_item(engine, item_id)
    if item is None or item.get("kind") != "ingest_task":
        return None
    claim = claim_specific(
        engine, item_id, token=token, now=now, lease_ttl_s=lease_ttl_s
    )
    if claim is None:
        return None
    if not mark_running(engine, item_id, claim, now=now):
        return None
    return claim


# ── team-collaboration :TaskNode bridge — MIGRATED (AU-P1-CL) ──────────
#
# Makes WorkItem authoritative for ``capabilities.teams.TeamCapability``'s
# shared ``:TaskNode`` (``pending|in_progress|completed``). Unlike the
# ingestion queue, there is no competitive multi-worker claim here — a team
# task is driven by explicit ``update_task_status`` calls from whichever
# agent is working it — so the bridge exposes the transitions directly
# rather than a claim-pool wrapper.

_TEAM_STATUS_TO_WORK_ITEM_VIEW: dict[str, str] = {
    "submitted": "pending",
    "ready": "pending",
    "leased": "in_progress",
    "running": "in_progress",
    "succeeded": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    "dead_letter": "failed",
}


def team_task_work_item_id(task_id: str) -> str:
    """Deterministic 1:1 WorkItem id for a team-collaboration ``:TaskNode`` id."""
    return f"workitem:team_task:{task_id}"


def ensure_team_task_work_item(
    engine: Any,
    task_id: str,
    *,
    tenant: str = "",
    now: float | None = None,
) -> str:
    """Idempotently create/reuse the WorkItem shadowing team ``:TaskNode`` ``task_id``.

    Submitted immediately ``ready`` (team tasks have no dependency graph);
    ``tenant`` carries the owning ``team_id`` for observability/quota.
    """
    item_id = team_task_work_item_id(task_id)
    return submit_work_item(
        engine,
        kind="team_task",
        payload_ref=task_id,
        tenant=tenant,
        work_item_id=item_id,
        idempotency_key=item_id,
        now=now,
    )


def start_team_task_work_item(
    engine: Any,
    task_id: str,
    *,
    tenant: str = "",
    token: str | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Ensure + claim + run a team task's shadow in one call (``pending -> in_progress``).

    Returns the claim dict on success, or ``None`` when the shadow is not
    presently ``ready`` (already running/terminal elsewhere) — the caller
    (``TeamCapability.update_task_status``) treats a ``None`` as "nothing to
    transition" and still mirrors the caller's requested status onto the
    legacy ``:TaskNode`` field (best-effort, non-blocking).
    """
    now = now if now is not None else _now()
    item_id = ensure_team_task_work_item(engine, task_id, tenant=tenant, now=now)
    claim = claim_specific(engine, item_id, token=token, now=now)
    if claim is None:
        return None
    if not mark_running(engine, item_id, claim, now=now):
        return None
    return claim


def team_task_status_view(engine: Any, task_id: str) -> str | None:
    """Canonical team-vocabulary status (``pending|in_progress|completed|
    failed|cancelled``) for a team task's shadow WorkItem, or ``None`` if no
    shadow exists yet (the task was never transitioned through the bridge —
    the legacy field is authoritative in that case, the non-destructive
    migration shim for pre-existing ``:TaskNode`` rows)."""
    item_id = team_task_work_item_id(task_id)
    item = get_work_item(engine, item_id)
    if item is None:
        return None
    return _TEAM_STATUS_TO_WORK_ITEM_VIEW.get(str(item.get("status")))


# ── ingestion :Task shim (read-only, SHIMMED not migrated) ─────────────


_TASK_STATUS_TO_WORK_ITEM: dict[str, str] = {
    "pending": "ready",
    "scheduled": "submitted",
    "blocked": "submitted",
    "running": "running",
    "completed": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
    "dead_letter": "dead_letter",
}


def work_item_view_of_task(engine: Any, task_id: str) -> dict[str, Any] | None:
    """Read-only WorkItem-status-vocabulary projection of an ingestion ``:Task`` node.

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
        "status": _TASK_STATUS_TO_WORK_ITEM.get(raw_status, "submitted"),
        "native_status": raw_status,
        "shim": True,
    }


# ── queued orchestrator WorkItems ──────────────────────────────────────


def orchestrator_work_item_id(job_id: str) -> str:
    """Deterministic WorkItem id for a queued orchestrator assignment."""
    return f"workitem:orchestrator:{job_id}"


def submit_orchestrator_work_item(
    engine: Any,
    job_id: str,
    *,
    tenant: str = "",
    description: str = "",
    depends_on: Sequence[str] = (),
    now: float | None = None,
) -> str:
    """Submit a queued orchestrator assignment directly as a WorkItem."""
    item_id = orchestrator_work_item_id(job_id)
    return submit_work_item(
        engine,
        kind="orchestrator_task",
        queue="orchestrator_task",
        payload_ref=job_id,
        tenant=tenant,
        description=description,
        depends_on=[orchestrator_work_item_id(dep) for dep in depends_on],
        resource_class="agent_dispatch",
        work_item_id=item_id,
        idempotency_key=item_id,
        now=now,
    )


# ── team WorkItems ──────────────────────────────────────────────────────────────


def team_work_item_id(task_id: str) -> str:
    """Return the deterministic WorkItem id for one team assignment."""
    return f"workitem:team:{task_id}"


def submit_team_work_item(
    engine: Any,
    task_id: str,
    *,
    tenant: str = "",
    description: str = "",
    assigned_to: str = "",
    created_by: str = "",
    now: float | None = None,
) -> str:
    """Submit a team assignment directly as its sole WorkItem record."""
    from agent_utilities.messaging.bus_privacy import bus_reference

    item_id = team_work_item_id(task_id)
    return submit_work_item(
        engine,
        kind="team_assignment",
        payload_ref=task_id,
        tenant=tenant,
        description=description,
        assigned_to=bus_reference("work_assignee", assigned_to, tenant=tenant),
        created_by=bus_reference("work_creator", created_by, tenant=tenant),
        work_item_id=item_id,
        idempotency_key=item_id,
        now=now,
    )


def start_team_work_item(
    engine: Any,
    task_id: str,
    *,
    tenant: str = "",
    token: str | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Claim and start an existing team WorkItem."""
    now = now if now is not None else _now()
    item_id = team_work_item_id(task_id)
    item = get_work_item(engine, item_id)
    if item is None or item.get("kind") != "team_assignment":
        return None
    if tenant and str(item.get("tenant") or "") != _work_tenant(tenant):
        return None
    claim = claim_specific(engine, item_id, token=token, now=now)
    if claim is None:
        return None
    if not mark_running(engine, item_id, claim, now=now):
        return None
    return claim


# ── Goal/Loop authority ─────────────────────────────────────────────


def loop_work_item_id(loop_id: str) -> str:
    """Deterministic WorkItem identity for a Goal/Loop definition."""
    return f"workitem:loop:{loop_id}"


def set_loop_statechart_instance_id(
    engine: Any, item_id: str, instance_id: str
) -> None:
    """Attach the Loop's ``eg-statechart`` instance id to its backing WorkItem (W2.5).

    ``Method::Statechart::Instantiate`` server-generates ``instance_id`` (no
    caller-supplied id support, unlike ``submit_work_item``'s
    ``work_item_id``), so the returned id is stored here and read back by
    ``research.loops.ensure_loop_statechart_instance`` on every subsequent
    call. Best-effort metadata, not part of the lease/fencing authority — a
    plain upsert (mirrors how ``research.loops.submit_loop`` upserts its
    Concept definition), not a CAS-guarded field.
    """
    _authority(engine).add_node(
        item_id, _NODE_LABEL, properties={"loop_statechart_instance_id": instance_id}
    )


def ensure_loop_work_item(
    engine: Any,
    loop_id: str,
    *,
    tenant: str = "",
    priority: int = 2,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    now: float | None = None,
) -> str:
    """Create/reuse the sole writable lifecycle record for ``loop_id``."""
    item_id = loop_work_item_id(loop_id)
    return submit_work_item(
        engine,
        kind="goal_loop",
        payload_ref=loop_id,
        tenant=tenant,
        priority=priority,
        max_attempts=max_attempts,
        work_item_id=item_id,
        idempotency_key=item_id,
        now=now,
    )


def claim_loop_work_item(
    engine: Any,
    loop_id: str,
    *,
    token: str | None = None,
    now: float | None = None,
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
) -> dict[str, Any] | None:
    """Claim an existing Goal/Loop WorkItem; never adopt a bare definition."""
    item_id = loop_work_item_id(loop_id)
    item = get_work_item(engine, item_id)
    if item is None or item.get("kind") != "goal_loop":
        return None
    claim = claim_specific(
        engine, item_id, token=token, now=now, lease_ttl_s=lease_ttl_s
    )
    if claim is None:
        return None
    if not mark_running(engine, item_id, claim, now=now):
        return None
    claims = dict(_loop_claims.get() or {})
    claims[item_id] = claim
    _loop_claims.set(claims)
    return claim


def transition_loop_work_item(
    engine: Any,
    loop_id: str,
    status: str,
    *,
    result_ref: str | None = None,
    error_ref: str | None = None,
    now: float | None = None,
    lease_ttl_s: float = DEFAULT_LEASE_TTL_S,
) -> bool:
    """Apply a Loop lifecycle transition exclusively through WorkItem.

    The caller must own the ambient claim to commit a terminal result. This
    prevents a Concept definition write from impersonating a worker.
    """
    item_id = loop_work_item_id(loop_id)
    item = get_work_item(engine, item_id)
    if item is None or item.get("kind") != "goal_loop":
        return False
    normalized = status.strip().lower()
    claim = (_loop_claims.get() or {}).get(item_id)
    if normalized == "paused":
        if claim is None:
            claim = claim_loop_work_item(
                engine, loop_id, now=now, lease_ttl_s=lease_ttl_s
            )
        if claim is None:
            return False
        deferred = defer_work_item(
            engine,
            item_id,
            claim,
            next_retry_at=(now if now is not None else _now()) + 60.0,
            reason_ref="loop_paused",
            now=now,
        )
        if deferred:
            claims = dict(_loop_claims.get() or {})
            claims.pop(item_id, None)
            _loop_claims.set(claims)
        return deferred
    if normalized in {"running", "pending", "validating"}:
        if claim is None:
            claim = claim_loop_work_item(
                engine, loop_id, now=now, lease_ttl_s=lease_ttl_s
            )
            return claim is not None
        return heartbeat(engine, item_id, claim, now=now, lease_ttl_s=lease_ttl_s)
    if normalized == "orphaned":
        item = get_work_item(engine, item_id)
        return bool(item and item.get("status") in {"submitted", "ready"})
    if normalized in {"cancelled", "canceled"} and claim is None:
        return cancel_work_item(
            engine, item_id, reason=error_ref or "cancelled", now=now
        )
    if claim is None:
        raise WorkItemBackendUnavailable(
            f"Loop {loop_id!r} has no claim in this execution context"
        )
    outcome = {
        "completed": "succeeded",
        "succeeded": "succeeded",
        "failed": "failed",
        "rejected": "failed",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        # Harness-enforced loop-exit terminal statuses (CONCEPT:AU-AHE.harness.
        # loop-exit-conditions). Each commits the WorkItem terminally; the
        # precise, diagnosable reason rides the result_ref/error_ref stamped by
        # ``loops.mark_loop_status`` and is returned verbatim by ``run_loop``.
        # An awaited external signal firing is a SUCCESS; every other enforced
        # exit is an abnormal/exhaustion termination -> FAILED.
        "external_event_satisfied": "succeeded",
        "max_iterations_exceeded": "failed",
        "budget_exceeded": "failed",
        "wall_clock_exceeded": "failed",
        "stalled": "failed",
        "error_threshold_exceeded": "failed",
    }.get(normalized)
    if outcome is None:
        raise ValueError(f"unsupported Loop status transition {status!r}")
    result = commit_result(
        engine,
        item_id,
        claim,
        outcome=outcome,
        result_ref=result_ref,
        error_ref=error_ref,
        retryable=False,
        now=now,
    )
    if result in {"committed", "noop", "dead_letter"}:
        claims = dict(_loop_claims.get() or {})
        claims.pop(item_id, None)
        _loop_claims.set(claims)
        return True
    return False


def set_work_item_priority(
    engine: Any,
    item_id: str,
    priority: Any,
    *,
    now: float | None = None,
) -> bool:
    """Update scheduling priority without granting claim/lease authority."""
    from agent_utilities.knowledge_graph.core.engine_tasks import _coerce_prio_bucket

    now = now if now is not None else _now()
    for _ in range(_CAS_LOOP_MAX_RETRIES):
        item = get_work_item(engine, item_id)
        if item is None or item.get("status") in TERMINAL_WORK_ITEM_STATUSES:
            return False
        current = int(item.get("prio_bucket") or 0)
        if _cas(
            engine,
            item_id,
            {"prio_bucket": current, "status": item.get("status")},
            {
                "prio_bucket": _coerce_prio_bucket(priority),
                "updated_at": now,
            },
        ):
            return True
    return False


def work_item_view_of_loop(engine: Any, loop_id: str) -> dict[str, Any] | None:
    """Read the authoritative WorkItem for a goal definition."""
    item = get_work_item(engine, loop_work_item_id(loop_id))
    if item is None:
        return None
    return {
        "work_item_id": loop_work_item_id(loop_id),
        "kind": "goal_loop",
        "status": str(item.get("status") or "submitted"),
        "native_status": str(item.get("status") or "submitted"),
        "updated_at": item.get("updated_at"),
        "projection": False,
    }
