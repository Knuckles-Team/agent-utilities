#!/usr/bin/python
from __future__ import annotations

"""Stateless agent dispatch worker — the ``agent-dispatch`` consumer fleet.

CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch — Queue-driven agent dispatch with session-keyed partitions
consumed by a stateless dispatch-worker fleet (sibling of the KG-2.57
``kg-ingest`` worker — same skeleton, same delivery contract):

* **Any host** can run ``agent-dispatch-worker`` processes. They consume
  :class:`~agent_utilities.orchestration.agent_dispatch.AgentTurnEnvelope`
  messages from the ``agent_turns`` queue (Kafka consumer group
  ``agent-dispatch``, or the Postgres/SQLite claim equivalents), rehydrate the
  referenced goal/session/job from the shared OS-5.16 state store, and execute
  through the EXISTING execution paths — ``core.sessions.run_goal_loop`` for
  goal runs, the orchestration manager's agent execution for orchestrator
  jobs. Nothing is duplicated; the worker only relocates WHERE those bodies
  run.
* **At-least-once + idempotent claims.** The queue ack/offset-commit happens
  strictly AFTER a turn finishes (or is durably marked failed). A worker crash
  redelivers the envelope (Kafka rebalance / Postgres visibility timeout /
  SQLite head-until-ack); the claim check then skips terminal jobs and
  re-claims jobs whose previous claim went stale — crash recovery without a
  separate scheduler (the reaper pattern, folded into the claim).
* **Per-session mutual exclusion.** Claims and execution run inside
  :func:`~agent_utilities.orchestration.agent_dispatch.session_execution_guard`
  (process-local lock + fleet-wide Postgres advisory lock), so even a
  redelivery racing the original consumer can never execute one session
  twice concurrently — the correctness contract for turn coherence.
* **Engine clients.** Like the ingest workers, dispatch workers force
  ``KG_DAEMON_ROLE=client`` (CONCEPT:AU-OS.identity.authenticated-identity-enforcement auth applies) and never contend
  for the KG host flock. ``main()`` binds a verified process actor/
  GraphSession (``acquire_process_identity_token`` -> ``mint_actor_from_token_sync``
  -> ``mint_graph_session``) BEFORE the first protected engine call, mirrors
  :func:`~agent_utilities.knowledge_graph.ingest_worker.main`'s identical
  bootstrap, and re-binds that SAME authority inside every pool worker
  thread (``_authorized_background_thread`` — a bare ``ContextVar`` does not
  cross a thread boundary), so a process identity may lease/execute work but
  never inherits or synthesizes a user's application permissions.
* **Delivery safety.** Ack follows durable terminal WorkItem state, never
  precedes it (:func:`_ack_after_durable_outcome` is the sole broker-ack
  chokepoint). An unparseable envelope is dead-lettered
  (:func:`_dead_letter_poison_envelope`, keyed by delivery digest for
  idempotent redelivery) before it may be acked; a turn-execution exception
  is durably committed as ``failed`` before its message may be acked.
* **Wire tenant is untrusted until re-checked at claim time.**
  ``enqueue_agent_turn`` verifies ``envelope.tenant`` against the caller's
  authenticated ``GraphSession`` at admission (fail closed, ``PermissionError``
  on mismatch) — but the ``agent_turns`` queue transport carries no signed
  per-message carrier yet (:class:`TenantMismatchError`'s docstring; GOC-15's
  envelope-carrier contract is deferred). Before claiming, the consumer loop
  re-reads the durable WorkItem this ``job_id`` was admitted under and rejects
  (dead-letters, never silently executes) a delivery whose wire tenant
  disagrees with it.

Run::

    python -m agent_utilities.orchestration.agent_dispatch_worker [--workers N]
    # or the console script:
    agent-dispatch-worker
"""

import hashlib
import json
import logging
import os
import secrets
import threading
import time
from collections.abc import Callable
from typing import Any

from agent_utilities.orchestration.agent_dispatch import (
    DISPATCH_GROUP,
    KIND_GOAL_LOOP,
    KIND_ORCHESTRATOR_TASK,
    AgentTurnEnvelope,
    get_dispatch_queue,
    session_execution_guard,
)

logger = logging.getLogger(__name__)

_PROCESS_WORKER_TOKEN = f"worker:{secrets.token_hex(16)}"


def worker_token() -> str:
    """Opaque process-lifetime identity for claims and heartbeats.

    Host names, operating-system user names, and filesystem locations are not
    lease authority and are never persisted in WorkItem ownership fields.
    """
    return _PROCESS_WORKER_TOKEN


def _claim_ttl_seconds(value: float | None = None) -> float:
    """Resolve the typed dispatch lease TTL; explicit test/caller values win."""
    if value is None:
        from agent_utilities.core.config import config

        value = config.agent_dispatch_claim_ttl_s
    ttl = float(value)
    if ttl <= 0:
        raise ValueError("dispatch claim TTL must be positive")
    return ttl


def _renew_interval_seconds(lease_ttl_s: float, value: float | None = None) -> float:
    """Resolve a renewal interval that is always strictly below the lease TTL."""
    if value is None:
        from agent_utilities.core.config import config

        value = config.agent_dispatch_renew_interval_s
    interval = min(float(value), float(lease_ttl_s) / 3.0)
    if not 0 < interval < lease_ttl_s:
        raise ValueError("dispatch renew interval must be below the lease TTL")
    return interval


# ── claims (idempotent, stale-claim aware) ─────────────────────────────────


def load_goal_run(
    goal_id: str,
    *,
    token: str | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Load one goal definition after checking its authoritative WorkItem.

    Reads the goal's KG Loop node (CONCEPT:AU-KG.research.these-properties-carry) plus the ``goal_spec``
    persisted in the session's metadata (the envelope carried only the reference).
    This is read-only rehydration. The parent dispatch WorkItem and the loop's
    own WorkItem provide all lifecycle ownership and fencing.
    """
    from agent_utilities.core import sessions as _sessions

    token = token or worker_token()
    now = now if now is not None else time.time()

    engine = _sessions._goal_engine()
    if engine is None:
        logger.warning("No active KG engine — cannot claim goal %s.", goal_id)
        return None
    try:
        rows = engine.query_cypher(
            "MATCH (c:Concept) WHERE c.id = $id RETURN c.id AS id, c.id AS goal_id, "
            "c.session_id AS session_id, c.objective AS objective, "
            "c.validation_cmd AS validation_cmd, c.max_iterations AS max_iterations, "
            "c.updated_at AS updated_at",
            {"id": goal_id},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Goal claim query failed (%s)", type(e).__name__)
        return None
    row = next(
        (r for r in (rows or []) if isinstance(r, dict) and r.get("goal_id")), None
    )
    if not row:
        logger.warning("Dispatch envelope for unknown goal %s skipped.", goal_id)
        return None
    from agent_utilities.orchestration.work_item import (
        TERMINAL_WORK_ITEM_STATUSES,
        work_item_view_of_loop,
    )

    work_view = work_item_view_of_loop(engine, goal_id)
    if work_view and work_view.get("status") in TERMINAL_WORK_ITEM_STATUSES:
        logger.debug("Duplicate delivery of terminal goal %s skipped.", goal_id)
        return None

    session_id = str(row.get("session_id") or "")
    spec: dict[str, Any] = {
        "goal_id": goal_id,
        "session_id": session_id,
        "objective": str(row.get("objective") or ""),
        "validation_cmd": str(row.get("validation_cmd") or ""),
        "max_iterations": int(row.get("max_iterations") or 20),
        "constraints": [],
    }
    try:
        conn = _sessions._connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT metadata_json FROM sessions WHERE id = ?", (session_id,))
        sess = cursor.fetchone()
        conn.close()
        if sess:
            stored = (json.loads(sess["metadata_json"] or "{}") or {}).get(
                "goal_spec"
            ) or {}
            for key in ("objective", "validation_cmd", "max_iterations"):
                if stored.get(key):
                    spec[key] = stored[key]
            if stored.get("constraints"):
                spec["constraints"] = list(stored["constraints"])
    except Exception as e:  # noqa: BLE001 — session goal_spec is a fallback
        logger.debug("session goal_spec fallback failed: %s", e)

    # Read-only rehydration. LoopController claims the goal's WorkItem; this
    # dispatch layer owns only the parent agent-turn WorkItem.
    return spec


def claim_orchestrator_work_item(
    engine: Any,
    job_id: str,
    *,
    token: str | None = None,
    now: float | None = None,
    claim_ttl_s: float | None = None,
) -> dict[str, Any] | None:
    """Claim one existing orchestrator WorkItem and return its payload."""
    token = token or worker_token()
    now = now if now is not None else time.time()
    claim_ttl_s = _claim_ttl_seconds(claim_ttl_s)

    from agent_utilities.orchestration import work_item as _wi

    view = getattr(engine, "_work_item_engine", engine)
    item_id = _wi.orchestrator_work_item_id(job_id)
    item = _wi.get_work_item(view, item_id)
    if item is None or item.get("kind") != "orchestrator_task":
        logger.warning("Dispatch envelope for unknown WorkItem %s skipped.", item_id)
        return None
    claim = _wi.claim_specific(
        view, item_id, token=token, now=now, lease_ttl_s=claim_ttl_s
    )
    if claim is None or not _wi.mark_running(view, item_id, claim, now=now):
        return None
    return {"job_id": job_id, "description": item.get("description") or "", **claim}


def _work_item_fence_still_valid(
    engine: Any,
    work_item_id: str,
    claim: dict[str, Any],
    *,
    lease_ttl_s: float | None = None,
) -> bool:
    """Renew and verify the sole WorkItem lease; fail closed on any mismatch.

    A claim without its deterministic WorkItem id is untrusted and cannot
    authorize a commit. Distinct from the AgentLease-based
    :func:`_fence_still_valid` below (a different node/claim shape entirely —
    kept as two functions, never overloaded under one name).
    """
    if claim.get("work_item_id") != work_item_id or engine is None:
        logger.warning(
            "WorkItem %s has no verifiable lease; commit rejected",
            work_item_id,
        )
        return False

    from agent_utilities.orchestration.work_item import heartbeat

    try:
        lease_ttl_s = _claim_ttl_seconds(lease_ttl_s)
        return heartbeat(
            engine,
            str(work_item_id),
            claim,
            lease_ttl_s=lease_ttl_s,
        )
    except Exception as exc:  # noqa: BLE001 — authority check fails closed
        logger.warning(
            "WorkItem fence renewal failed for %s; commit rejected (error_type=%s)",
            work_item_id,
            type(exc).__name__,
        )
        return False


class WorkItemLeaseLost(RuntimeError):
    """The current worker no longer owns a renewable WorkItem lease."""


class TenantMismatchError(RuntimeError):
    """A delivered envelope's wire ``tenant`` disagrees with the WorkItem it
    was admitted under (CONCEPT: GOC-18 consumer-side defense in depth).

    ``enqueue_agent_turn`` already verifies ``envelope.tenant`` against an
    authenticated ``GraphSession`` at ADMISSION time
    (``agent_dispatch.py``'s ``PermissionError`` gate) — but the
    ``agent_turns`` queue transport itself carries no signed per-message
    carrier yet (GOC-15's envelope-carrier contract is still deferred; see
    this module's docstring). A delivery is therefore untrusted wire data on
    the CONSUMER side even for a syntactically well-formed envelope: a
    tampered/forged broker message reusing a legitimate ``job_id`` but
    asserting a different ``tenant`` must never be silently trusted or
    silently executed.
    """


class WorkItemLeaseGuard:
    """Periodically renew one native WorkItem lease during long execution.

    Executors receive this guard as their second argument. Every mutating
    operation must run through :meth:`side_effect`, which performs a synchronous
    renewal immediately before invoking it. The background renewal only keeps
    long computation alive; it never turns a stale fencing token into authority.
    """

    def __init__(
        self,
        engine: Any,
        work_item_id: str,
        claim: dict[str, Any],
        *,
        lease_ttl_s: float,
        heartbeat_interval_s: float | None = None,
    ) -> None:
        if lease_ttl_s <= 0:
            raise ValueError("lease_ttl_s must be positive")
        self.engine = engine
        self.work_item_id = work_item_id
        self.claim = claim
        self.lease_ttl_s = float(lease_ttl_s)
        self.heartbeat_interval_s = _renew_interval_seconds(
            self.lease_ttl_s, heartbeat_interval_s
        )
        if not 0 < self.heartbeat_interval_s < self.lease_ttl_s:
            raise ValueError("heartbeat interval must be positive and below lease TTL")
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._renew_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def require_current(self) -> None:
        """Synchronously renew or raise before an authoritative side effect."""
        if self._lost.is_set():
            raise WorkItemLeaseLost("WorkItem lease was lost")
        with self._renew_lock:
            if self._lost.is_set() or not _work_item_fence_still_valid(
                self.engine,
                self.work_item_id,
                self.claim,
                lease_ttl_s=self.lease_ttl_s,
            ):
                self._lost.set()
                raise WorkItemLeaseLost("WorkItem lease renewal was rejected")

    def side_effect(
        self, operation: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """Renew immediately, then invoke one bounded mutating operation."""
        self.require_current()
        return operation(*args, **kwargs)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval_s):
            try:
                self.require_current()
            except WorkItemLeaseLost:
                return

    def start(self) -> WorkItemLeaseGuard:
        """Validate once and start periodic renewal."""
        self.require_current()
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="WorkItemLeaseHeartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(1.0, self.heartbeat_interval_s + 0.1))

    def __enter__(self) -> WorkItemLeaseGuard:
        return self.start()

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()


#: Marker `claim["_claim_backend"]` value the (now-retired, No-Legacy-deleted —
#: see `orchestration/engine_claim.py`'s module docstring) engine-native probe
#: backend used to stamp. No live claim producer sets this value anymore
#: (`engine_claim.claim_agent_task` has one backend, `workitem`, whose claims
#: carry no `_claim_backend` marker at all) — kept as a bare string literal so
#: `_fence_still_valid`'s dead-but-harmless "engine-native" branch below still
#: type-checks/tests against the exact historical marker shape.
_CLAIM_BACKEND_ENGINE_NATIVE = "engine"


def _fence_still_valid(
    engine: Any, task_id: str, claim: dict[str, Any], *, token: str
) -> bool:
    """CAS gate at commit time: reject a stale holder whose lease was reclaimed.

    Re-reads the LIVE ``:AgentLease`` for ``task_id`` and compares its
    ``lease_epoch`` against the epoch this ``claim`` was issued under
    (``claim["fence_token"]``). A live epoch strictly greater than the
    claimed one means a newer claim now holds the resource (the original
    holder's lease expired and was re-claimed while it kept executing) — that
    stale holder's commit must be rejected, never allowed to overwrite the
    newer holder's work.

    Posture (AU-P0-3/L15) depends on which backend produced ``claim``
    (``claim["_claim_backend"]``, historically stamped by :func:`claim_agent_task`
    as ``"kg"`` and by the engine-native probe backend as ``"engine"`` — both
    retired by ``engine_claim.py``'s No-Legacy cleanup; no live path stamps
    either marker today, so this function's ``is_engine_native`` branch is
    unreachable in practice and kept only for the historical claim shape):

    * **KG best-effort path** (``_claim_backend != "engine"``, including
      claims with no marker at all — e.g. hand-built test fixtures) — fails
      OPEN (returns ``True``) when there is nothing to fence against: no
      engine, no ``fence_token`` on the claim, no live lease row, or a lease
      row that predates this fencing scheme (no ``lease_epoch`` recorded).
      Same best-effort posture as :func:`resolve_capability_grant` (an
      audit-read hiccup must never block a legitimate commit on this path).
    * **Engine-native path** (``_claim_backend == "engine"``) — fails CLOSED
      (returns ``False``, rejecting the commit) whenever the fence cannot be
      confirmed: no engine to query, or the fence-check query itself raises.
      A worker on this path that cannot confirm it still holds the lease
      must NOT commit — silently allowing the commit through on a query
      error would let a stale holder overwrite a newer holder's work with no
      way to detect it after the fact.
    """
    is_engine_native = claim.get("_claim_backend") == _CLAIM_BACKEND_ENGINE_NATIVE
    if engine is None:
        if is_engine_native:
            logger.warning(
                "Fence check for engine-native claim %s has no engine client "
                "to verify against — failing CLOSED (commit rejected).",
                task_id,
            )
            return False
        return True
    claimed_epoch = claim.get("fence_token")
    if claimed_epoch is None:
        return True
    try:
        rows = engine.query_cypher(
            "MATCH (l:AgentLease {resource_id: $rid}) RETURN l.id AS id, "
            "l.owner_token AS owner_token, "
            "l.lease_epoch AS lease_epoch ORDER BY l.acquired_at DESC LIMIT 1",
            {"rid": task_id},
        )
    except Exception as e:  # noqa: BLE001 — see posture note above: KG path only
        if is_engine_native:
            logger.warning(
                "Fence check query failed for engine-native claim %s — "
                "cannot confirm the lease is still held, failing CLOSED "
                "(commit rejected): %s",
                task_id,
                e,
            )
            return False
        logger.debug("Fence check query failed for %s: %s", task_id, e)
        return True
    if not rows:
        return True
    live_epoch = rows[0].get("lease_epoch")
    if live_epoch is None:
        return True
    if int(live_epoch) > int(claimed_epoch):
        return False
    return True


# ── capability grants (Gap-6) ─────────────────────────────────────────
#
# The write/read pair completing the ``AUTHORIZED_FOR`` edge team-synthesis
# (``orchestration/engine.py``) already queries but that, until now, nothing
# in this codebase actually wrote. See ``AgentCapabilityGrantNode`` for the
# reuse-audit against ``AgentIdentityNode.capabilities``/``AgentCapabilityNode``.


def resolve_capability_grant(
    engine: Any,
    agent_id: str,
    capability: str,
    *,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Look up the most recent live (non-revoked, non-expired) grant for ``(agent_id, capability)``.

    Best-effort — a query failure or no engine returns ``None`` (never
    raises), same posture as every other durable-accounting read in this
    codebase (e.g. ``action_policy._recent_decisions``).
    """
    if engine is None or not agent_id or not capability:
        return None
    from agent_utilities.messaging.bus_privacy import bus_reference

    agent_id = bus_reference("agent", agent_id)
    now = now if now is not None else time.time()
    try:
        rows = engine.query_cypher(
            "MATCH (g:AgentCapabilityGrant {agent_id: $agent_id, "
            "capability: $capability}) "
            "RETURN g.id AS id, g.issuer AS issuer, g.granted_at AS granted_at, "
            "g.expires_at AS expires_at, g.revoked AS revoked "
            "ORDER BY g.granted_at DESC LIMIT 1",
            {"agent_id": agent_id, "capability": capability},
        )
    except Exception as e:  # noqa: BLE001 — resolution is best-effort
        logger.debug("capability grant query failed (%s)", type(e).__name__)
        return None
    if not rows:
        return None
    row = rows[0]
    if not row.get("id") or row.get("revoked"):
        return None
    expires_at = row.get("expires_at")
    if expires_at is not None and float(expires_at) <= now:
        return None
    return dict(row)


def grant_capability(
    engine: Any,
    agent_id: str,
    capability: str,
    *,
    issuer: str = "system",
    ttl_seconds: float | None = None,
    now: float | None = None,
) -> str | None:
    """Issue and persist one ``:AgentCapabilityGrant``, linked ``Agent -[:AUTHORIZED_FOR]-> grant``.

    Best-effort (never raises); returns the new grant id, or ``None`` on a
    missing engine / write failure.
    """
    if engine is None:
        return None
    from agent_utilities.messaging.bus_privacy import bus_reference

    agent_id = bus_reference("agent", agent_id)
    issuer = bus_reference("capability_issuer", issuer)
    now = now if now is not None else time.time()
    grant_id = f"capability_grant:{agent_id}:{capability}:{secrets.token_hex(16)}"
    expires_at = (now + ttl_seconds) if ttl_seconds else None
    try:
        engine.add_node(
            grant_id,
            "AgentCapabilityGrant",
            properties={
                "name": f"Grant: {capability} -> {agent_id}",
                "agent_id": agent_id,
                "capability": capability,
                "issuer": issuer,
                "granted_at": now,
                "expires_at": expires_at,
                "revoked": False,
            },
        )
        add_edge = getattr(engine, "add_edge", None)
        if callable(add_edge):
            add_edge(agent_id, grant_id, "AUTHORIZED_FOR")
    except Exception as e:  # noqa: BLE001 — grant issuance is best-effort
        logger.warning("grant_capability write failed (%s)", type(e).__name__)
        return None
    return grant_id


class NoExecutorBoundError(RuntimeError):
    """Raised when no concrete executor is bound to an executable WorkItem.

    Distinguishes an UNROUTABLE task (this) from a real executor failure (any
    other exception raised by a bound executor) while guaranteeing both are
    recorded as unsuccessful — AU-P0-3: unrun work must never be marked
    ``completed`` with ``reward=1.0``.
    """


def _default_work_item_executor(
    claim: dict[str, Any], _lease: WorkItemLeaseGuard
) -> str:
    """Structural default executor: FAILS CLOSED — no executor bound means no
    work ran, so this must never be recorded as a successful completion.

    WorkItem producers should pass a real ``executor=`` callable to
    :func:`execute_work_item_turn`. Raising :class:`NoExecutorBoundError`
    routes the item through the failure path (``status=
    "unroutable"``, ``reward=0.0``), the same no-fabrication discipline
    :class:`~agent_utilities.models.evidence_bundle.EvidenceBundle` documents
    for the identical reason.
    """
    raise NoExecutorBoundError(
        f"no executor bound for WorkItem {claim.get('work_item_id')}"
    )


def _write_work_item_provenance(
    engine: Any,
    *,
    work_item_id: str,
    claim: dict[str, Any],
    agent_id: str,
    status: str,
    result: Any,
    evidence: Any,
    policy_decision_node: Any,
    grant_id: str | None,
) -> str:
    """Persist one WorkItem's Observation/Claim/Action/Trace as ONE atomic batch.

    BUG-015/GOC-20 (B8, ``decisions/GOC-20-atomic-outcome-provenance.md``):
    previously this wrote four nodes with four separate, non-atomic
    ``engine.add_node`` calls and returned ``None`` on both total success and
    total failure -- the caller (:func:`execute_work_item_turn`) never
    inspected the return value at all, so a WorkItem could report
    ``"completed"`` with zero provenance nodes. It now returns
    ``"written"``/``"unavailable"``/``"failed"`` -- never ``None`` -- mirroring
    ``agent_runner._persist_execution_provenance_batch``'s established
    contract, and commits all four nodes through the engine's native
    all-or-nothing typed batch instead of four independent writes. There is
    no serial per-node fallback: an all-or-nothing failure must never be
    reinterpreted as a partial write.
    """
    if engine is None:
        return "unavailable"
    batch_write = getattr(engine, "batch_typed_mutations", None)
    if not callable(batch_write):
        return "unavailable"

    from agent_utilities.messaging.bus_privacy import bus_reference
    from agent_utilities.models.knowledge_graph import (
        ActionNode,
        ClaimNode,
        ObservationNode,
        TraceNode,
    )
    from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

    privacy = PersistencePrivacyGuard()
    clean_result, _result_privacy = privacy.sanitize_text(str(result))
    clean_reason, _reason_privacy = privacy.sanitize_text(
        str(policy_decision_node.reason)
    )
    agent_ref = bus_reference("agent", agent_id)

    obs_id = f"observation:work_item:{work_item_id}:{secrets.token_hex(16)}"
    claim_node_id = f"claim:work_item:{work_item_id}:{secrets.token_hex(16)}"
    action_id = f"action:work_item:{work_item_id}:{secrets.token_hex(16)}"
    trace_id = f"trace:work_item:{work_item_id}:{secrets.token_hex(16)}"
    lease_id = claim.get("lease_id", "")
    confidence = getattr(evidence, "confidence", None)
    confidence = confidence if confidence is not None else 1.0

    try:
        observation = ObservationNode(
            id=obs_id,
            name=f"Observation: {work_item_id}",
            content=(
                f"WorkItem {work_item_id} claimed via lease {lease_id} "
                f"(dag={claim.get('dag_id') or 'n/a'})"
            ),
            confidence=confidence,
            source="agent-dispatch",
        )
        obs_props = observation.to_graph_properties(exclude={"id"})
        obs_props["work_item_id"] = work_item_id
        obs_props["lease_id"] = lease_id

        policy_claim = ClaimNode(
            id=claim_node_id,
            name=f"Claim: {work_item_id} policy decision",
            claim_text=(
                f"{policy_decision_node.kind}({policy_decision_node.target}) -> "
                f"{policy_decision_node.decision} ({clean_reason})"
            ),
            claim_type="decision",
            is_verified=policy_decision_node.allowed,
        )
        claim_props = policy_claim.to_graph_properties(exclude={"id"})
        claim_props["work_item_id"] = work_item_id
        claim_props["policy_decision_id"] = policy_decision_node.id

        action = ActionNode(
            id=action_id,
            name=f"Action: execute {work_item_id}",
            action_type="work_item.execute",
            status=status,
            result=clean_result[:4000],
        )
        action_props = action.to_graph_properties(exclude={"id"})
        action_props["work_item_id"] = work_item_id
        action_props["lease_id"] = lease_id
        action_props["policy_decision_id"] = policy_decision_node.id
        action_props["capability_grant_id"] = grant_id or ""
        action_props["agent_id"] = agent_ref

        trace = TraceNode(
            id=trace_id,
            name=f"Trace: work_item {work_item_id}",
            agent=agent_ref or None,
            task_id=work_item_id,
            status="ok" if status == "completed" else "error",
            outcome=status,
        )
        trace_props = trace.to_graph_properties(exclude={"id"})
        trace_props["lease_id"] = lease_id

        mutations = [
            {
                "kind": "node",
                "id": obs_id,
                "node_type": "Observation",
                "properties": obs_props,
            },
            {
                "kind": "node",
                "id": claim_node_id,
                "node_type": "Claim",
                "properties": claim_props,
            },
            {
                "kind": "node",
                "id": action_id,
                "node_type": "Action",
                "properties": action_props,
            },
            {
                "kind": "node",
                "id": trace_id,
                "node_type": "Trace",
                "properties": trace_props,
            },
        ]
        ok = batch_write(mutations)
    except Exception as e:  # noqa: BLE001 — an all-or-nothing batch failure, never
        # reinterpreted as a partial write (BUG-015/GOC-20 B8).
        logger.warning("work-item provenance batch failed (%s)", type(e).__name__)
        return "failed"
    return "written" if ok else "unavailable"


def _finalize_work_item(
    engine: Any,
    work_item_id: str,
    claim: dict[str, Any],
    *,
    status: str,
    reward: float,
    feedback_text: str,
) -> str:
    """Commit the WorkItem, then durably append its OutcomeEvaluation.

    BUG-015/GOC-20 (B7, ``decisions/GOC-20-atomic-outcome-provenance.md``): the
    WorkItem status CAS (:func:`~agent_utilities.orchestration.work_item
    .commit_execution_work_item`) is the existing atomic, fenced B6 boundary —
    one redb transaction with its outbox row, CAS'd on lease epoch/fencing
    token — and is unchanged here. The OutcomeEvaluation append that follows
    it is no longer best-effort-and-silent: previously a failed append was
    caught, logged at ``warning``, and never surfaced, so this function
    returned ``"committed"`` (or the terminal status) whether or not the
    OutcomeEvaluation node actually landed. It now returns ``"degraded"``
    instead whenever that append does not durably land (no native typed-batch
    capability, or the batch itself fails) — a caller can no longer read the
    WorkItem's own committed status as proof its OutcomeEvaluation exists.
    There is no serial fallback: a batch that cannot commit atomically is
    reported, never silently reattempted node-by-node.
    """
    if engine is None:
        return "missing"
    from agent_utilities.models.knowledge_graph import OutcomeEvaluationNode
    from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

    if claim.get("work_item_id") != work_item_id:
        return "missing"
    from agent_utilities.orchestration.work_item import commit_execution_work_item

    try:
        committed = commit_execution_work_item(
            engine, work_item_id, claim, status=status
        )
    except Exception as exc:  # noqa: BLE001 — execution API is fail-closed/no-raise
        logger.warning("WorkItem commit failed (%s)", type(exc).__name__)
        return "conflict"
    if committed in {"fenced", "missing", "conflict"}:
        return str(committed)

    outcome_id = f"outcome:work_item:{work_item_id}:{secrets.token_hex(16)}"
    batch_write = getattr(engine, "batch_typed_mutations", None)
    outcome_written = False
    if callable(batch_write):
        try:
            clean_feedback, _privacy_report = PersistencePrivacyGuard().sanitize_text(
                feedback_text
            )
            outcome = OutcomeEvaluationNode(
                id=outcome_id,
                name=f"Outcome: {work_item_id}",
                reward=reward,
                feedback_text=clean_feedback,
                lease_id=claim.get("lease_id", ""),
                dag_id=claim.get("dag_id", ""),
            )
            outcome_written = bool(
                batch_write(
                    [
                        {
                            "kind": "node",
                            "id": outcome_id,
                            "node_type": "OutcomeEvaluation",
                            "properties": outcome.to_graph_properties(exclude={"id"}),
                        }
                    ]
                )
            )
        except Exception as e:  # noqa: BLE001 — a failed OutcomeEvaluation write must
            # DEGRADE the reported status, never be swallowed as if it landed
            # (BUG-015/GOC-20 B7).
            logger.warning("work-item outcome append failed (%s)", type(e).__name__)
            outcome_written = False
    if not outcome_written:
        return "degraded"
    return str(committed or "blocked")


def execute_work_item_turn(
    engine: Any,
    work_item_id: str,
    *,
    agent_id: str = "",
    capability: str = "work_item.execute",
    executor: Callable[[dict[str, Any], WorkItemLeaseGuard], Any] | None = None,
    evidence: Any = None,
    token: str | None = None,
    now: float | None = None,
    claim_ttl_s: float | None = None,
) -> str:
    """Claim, execute, and commit one durable executable WorkItem.

    The full chain is::

        ClaimWorkItem -> EvidenceBundle -> policy frame
        (AgentPolicyDecisionNode over action_policy.decide()) -> capability
        grant (AgentCapabilityGrantNode over AUTHORIZED_FOR) -> execute ->
        Observation/Claim/Action + AgentTrace + AgentOutcome
        OutcomeEvaluationNode -> fenced WorkItem commit and native dependency
        release.

    Outcomes: ``"skipped"`` (duplicate delivery / live claim elsewhere, from
    the native claim), ``"blocked"`` (action_policy queued the action
        for human approval — its WorkItem lease is fenced and deferred so a
        fresh claim after approval retries it), ``"denied"`` (action_policy forbade the action
    outright — terminal), ``"unroutable"`` (no executor was bound — terminal,
    ``reward=0.0``, AU-P0-3 fail-closed), ``"fenced"`` (this holder's lease
    was reclaimed by a newer holder before it could commit — the commit is
    rejected, no writeback happens, AU-P0-3 fencing), ``"completed"`` |
    ``"failed"`` (the executor ran; writeback recorded), ``"degraded"``
    (BUG-015/GOC-20: the WorkItem status itself committed, but its required
    OutcomeEvaluation and/or Observation/Claim/Action/Trace provenance did
    NOT durably land — the caller must not treat this as a clean
    ``"completed"``/``"failed"``/``"blocked"``/``"denied"`` outcome; see
    ``decisions/GOC-20-atomic-outcome-provenance.md``). Never raises — an
    executor exception is caught and recorded as a failed outcome, mirroring
    :func:`_execute_orchestrator_turn`'s durable failure path.

    The executor signature is ``executor(claim, lease_guard)``. Mutating calls
    must use ``lease_guard.side_effect(...)`` so authority is renewed directly
    before the effect; long computation is covered by periodic renewal.

    A negative native claim is final. No graph projection or alternate claim
    backend is consulted.
    """
    token = token or worker_token()
    now = now if now is not None else time.time()
    claim_ttl_s = _claim_ttl_seconds(claim_ttl_s)

    from agent_utilities.orchestration.work_item import claim_execution_work_item

    claim = claim_execution_work_item(
        engine, work_item_id, token=token, now=now, claim_ttl_s=claim_ttl_s
    )
    if claim is None:
        return "skipped"
    from agent_utilities.messaging.bus_privacy import bus_reference

    agent_id = bus_reference("agent", agent_id, tenant=str(claim.get("tenant") or ""))

    # EvidenceBundle (C1) — minimal, honest envelope: what is known about this
    # claim before executing. Callers with a real retrieval surface should
    # pass `evidence=` instead of relying on this placeholder.
    if evidence is None:
        from agent_utilities.models.evidence_bundle import EvidenceBundle

        evidence = EvidenceBundle(
            reasoning_trace=[{"step": "work_item_claim", **claim}]
        )

    # Policy frame (AgentPolicyDecision) — the SAME action_policy gate every
    # other autonomous mutating action goes through.
    from agent_utilities.models.knowledge_graph import AgentPolicyDecisionNode
    from agent_utilities.orchestration.action_policy import (
        DECISION_QUEUE,
        ActionRequest,
        get_action_policy,
    )

    lease = WorkItemLeaseGuard(
        engine,
        work_item_id,
        claim,
        lease_ttl_s=claim_ttl_s,
    )
    try:
        lease.start()
        policy_decision = lease.side_effect(
            get_action_policy(engine).decide,
            ActionRequest(
                kind="work_item.execute",
                target=work_item_id,
                source="agent-dispatch",
                actor_id=agent_id,
            ),
        )
    except WorkItemLeaseLost:
        lease.close()
        return "fenced"
    except Exception:
        lease.close()
        raise
    policy_decision_node = AgentPolicyDecisionNode.from_action_decision(
        policy_decision, agent_id=agent_id
    )

    if not policy_decision.allowed:
        status = "blocked" if policy_decision.decision == DECISION_QUEUE else "denied"
        result = (
            f"policy {policy_decision.decision} ({policy_decision.tier}): "
            f"{policy_decision.reason}"
        )
        try:
            finalization = lease.side_effect(
                _finalize_work_item,
                engine,
                work_item_id,
                claim,
                status=status,
                reward=0.0,
                feedback_text=result[:2000],
            )
        except WorkItemLeaseLost:
            return "fenced"
        finally:
            lease.close()
        if finalization in {"fenced", "missing", "conflict"}:
            return "fenced"
        provenance_status = _write_work_item_provenance(
            engine,
            work_item_id=work_item_id,
            claim=claim,
            agent_id=agent_id,
            status=status,
            result=result,
            evidence=evidence,
            policy_decision_node=policy_decision_node,
            grant_id=None,
        )
        # BUG-015/GOC-20 (B7/B8): the WorkItem's OutcomeEvaluation and its
        # Observation/Claim/Action/Trace provenance are both REQUIRED — a
        # terminal report may not claim a clean outcome while either is
        # missing. See decisions/GOC-20-atomic-outcome-provenance.md.
        if finalization == "degraded" or provenance_status != "written":
            return "degraded"
        return "blocked" if status == "blocked" else "denied"

    # Capability grant — resolve an existing grant, or self-issue a bootstrap
    # one so there is always SOME AUTHORIZED_FOR audit trail for the
    # execution (advisory today: action_policy above is the hard gate; this
    # is the per-grant record team-synthesis already reads).
    grant_id: str | None = None
    try:
        if agent_id:
            existing = resolve_capability_grant(engine, agent_id, capability, now=now)
            grant_id = existing.get("id") if existing else None
            if grant_id is None:
                grant_id = lease.side_effect(
                    grant_capability,
                    engine,
                    agent_id,
                    capability,
                    issuer="agent-dispatch",
                    ttl_seconds=claim_ttl_s,
                    now=now,
                )

        # Execute — pluggable body; the default FAILS CLOSED. The lease guard
        # renews periodically, and the executor receives the current-only
        # side-effect fencing surface.
        try:
            lease.require_current()
            # BUG-070: bind WorkItem/lease/agent identity for the exact
            # duration of the executor call so any generic engine mutation
            # it triggers (e.g. lifecycle.batch_update) is attributable from
            # logs alone -- see work_item_context's module docstring for the
            # BUG-064 incident this closes the gap for.
            from agent_utilities.orchestration.work_item_context import (
                bind_work_item_context,
            )

            with bind_work_item_context(
                work_item_id=work_item_id,
                agent_id=agent_id,
                lease_id=str(claim.get("lease_id", "")),
                capability=capability,
            ):
                result = (executor or _default_work_item_executor)(claim, lease)
            status = "completed"
            reward = 1.0
        except WorkItemLeaseLost:
            raise
        except NoExecutorBoundError as e:
            result = str(e)
            status = "unroutable"
            reward = 0.0
        except Exception as e:  # noqa: BLE001 — durably record, never raise
            result = str(e)
            status = "failed"
            reward = 0.0

        finalization = lease.side_effect(
            _finalize_work_item,
            engine,
            work_item_id,
            claim,
            status=status,
            reward=reward,
            feedback_text=str(result)[:2000],
        )
    except WorkItemLeaseLost:
        logger.warning(
            "WorkItem %s: renewable fence was lost; result discarded",
            work_item_id,
        )
        return "fenced"
    finally:
        lease.close()
    if finalization in {"fenced", "missing", "conflict"}:
        return "fenced"
    provenance_status = _write_work_item_provenance(
        engine,
        work_item_id=work_item_id,
        claim=claim,
        agent_id=agent_id,
        status=status,
        result=result,
        evidence=evidence,
        policy_decision_node=policy_decision_node,
        grant_id=grant_id,
    )
    # BUG-015/GOC-20 (B7/B8): a WorkItem may not report "completed"/"failed"
    # while its OutcomeEvaluation or Observation/Claim/Action/Trace provenance
    # did not durably land. See decisions/GOC-20-atomic-outcome-provenance.md.
    if finalization == "degraded" or provenance_status != "written":
        return "degraded"
    return status


def _default_agent_task_executor(claim: dict[str, Any]) -> str:
    """Structural default executor: FAILS CLOSED — no executor bound means no
    work ran (mirrors :func:`_default_work_item_executor`'s discipline for
    the AgentTask-specific, single-argument executor contract below).
    """
    raise NoExecutorBoundError(
        f"no executor bound for AgentTask {claim.get('task_id')}"
    )


def _finalize_agent_task(
    engine: Any,
    work_item_id: str,
    claim: dict[str, Any],
    *,
    status: str,
) -> str | None:
    """Commit an executed ``:AgentTask`` turn's WorkItem shadow.

    Mirrors :func:`_finalize_work_item`'s pattern for the AgentTask-specific
    bridge: commits through :func:`~agent_utilities.orchestration.work_item
    .commit_agent_task_work_item` (native dependency release, DLQ, idempotent
    commit), then mirrors the TERMINAL legacy ``:AgentTask.status`` for
    unmigrated readers (``fleet_reconciler``, dashboards) —
    ``claim_agent_task_via_work_item`` already mirrored ``"running"`` at
    claim time.
    """
    if engine is None:
        return None
    if claim.get("work_item_id") != work_item_id:
        return None

    from agent_utilities.orchestration.work_item import commit_agent_task_work_item

    try:
        committed = commit_agent_task_work_item(
            engine, work_item_id, claim, status=status
        )
    except Exception as exc:  # noqa: BLE001 — commit API is fail-closed/no-raise
        logger.warning("AgentTask WorkItem commit failed (%s)", type(exc).__name__)
        return "conflict"

    task_id = str(claim.get("task_id") or "")
    if task_id:
        try:
            engine.add_node(task_id, "AgentTask", properties={"status": status})
        except Exception as e:  # noqa: BLE001 — mirror is best-effort
            logger.warning(
                "legacy AgentTask status mirror failed for %s: %s", task_id, e
            )

    return committed


def execute_agent_task_turn(
    engine: Any,
    task_id: str,
    *,
    agent_id: str = "",
    executor: Callable[[dict[str, Any]], Any] | None = None,
    token: str | None = None,
    now: float | None = None,
    claim_ttl_s: float | None = None,
) -> str:
    """Claim, execute, and commit one durable ``:AgentTask`` turn (AU-P1-1 MIGRATED path).

    The bridge counterpart of :func:`execute_work_item_turn` for the legacy
    ``:AgentTask``/``TASK_DEPENDS_ON`` DAG shape (``TeamComposition
    .to_durable_task_dag``): claims through
    :func:`~agent_utilities.orchestration.engine_claim.claim_agent_task`
    (which shadows the task 1:1 onto a WorkItem — see
    :func:`~agent_utilities.orchestration.work_item.ensure_agent_task_work_item`
    — and mirrors the legacy ``:AgentTask``/``:AgentLease`` nodes for
    unmigrated readers), runs the bound ``executor(claim)``, then commits
    through the SAME fenced WorkItem authority :func:`execute_work_item_turn`
    uses, with native cross-task dependency release.

    Outcomes: ``"skipped"`` (duplicate delivery / live claim elsewhere),
    ``"unroutable"`` (no executor was bound — terminal, AU-P0-3 fail-closed),
    ``"fenced"`` (this holder's lease was reclaimed by a newer holder before
    it could commit — no writeback happens), ``"completed"`` | ``"failed"``
    (the executor ran; writeback recorded, dependents released on success).
    Never raises — an executor exception is caught and recorded as a failed
    outcome.

    The executor signature is ``executor(claim)`` — simpler than
    :func:`execute_work_item_turn`'s ``executor(claim, lease_guard)``, since
    an AgentTask body is typically a short, already-fenced unit of work
    (``TeamComposition``-authored tasks) rather than the long-running,
    self-renewing computation that seam is for; use
    :func:`execute_work_item_turn` directly if a task body needs its own
    ``lease_guard.side_effect`` renewal points.

    Unlike :func:`execute_work_item_turn`, this does not (yet) route through
    ``action_policy``'s governance gate or issue a capability grant — no
    caller of the legacy ``:AgentTask`` DAG shape requires per-action policy
    review today. A future caller that does should compose
    :func:`execute_work_item_turn` directly against the same WorkItem shadow
    rather than extending this function's contract.
    """
    token = token or worker_token()
    now = now if now is not None else time.time()
    claim_ttl_s = _claim_ttl_seconds(claim_ttl_s)

    from agent_utilities.orchestration.engine_claim import claim_agent_task

    claim = claim_agent_task(
        engine, task_id, token=token, now=now, claim_ttl_s=claim_ttl_s
    )
    if claim is None:
        return "skipped"

    work_item_id = str(claim.get("work_item_id") or "")
    if not work_item_id:
        # No live claim backend produces a claim without one; fail closed
        # rather than commit against an unverifiable authority.
        logger.warning(
            "AgentTask %s claim carries no work_item_id; commit rejected", task_id
        )
        return "fenced"

    lease = WorkItemLeaseGuard(engine, work_item_id, claim, lease_ttl_s=claim_ttl_s)
    try:
        lease.start()
        try:
            lease.require_current()
            # BUG-070: same WorkItem-identity binding as execute_work_item_turn
            # -- the AgentTask bridge shadows onto a WorkItem 1:1, so it is the
            # same claimed-WorkItem execution seam a generic engine mutation
            # (e.g. lifecycle.batch_update) can reach through.
            from agent_utilities.orchestration.work_item_context import (
                bind_work_item_context,
            )

            with bind_work_item_context(
                work_item_id=work_item_id,
                agent_id=agent_id,
                lease_id=str(claim.get("lease_id", "")),
                task_id=task_id,
            ):
                result = (executor or _default_agent_task_executor)(claim)
            status = "completed"
        except WorkItemLeaseLost:
            raise
        except NoExecutorBoundError as e:
            result = str(e)
            status = "unroutable"
        except Exception as e:  # noqa: BLE001 — durably record, never raise
            result = str(e)
            status = "failed"

        finalization = lease.side_effect(
            _finalize_agent_task,
            engine,
            work_item_id,
            claim,
            status=status,
        )
    except WorkItemLeaseLost:
        logger.warning(
            "AgentTask %s: renewable fence was lost; result discarded", task_id
        )
        return "fenced"
    finally:
        lease.close()
    if finalization in {"fenced", "missing", "conflict", None}:
        return "fenced"
    from agent_utilities.security.persistence_privacy import persistence_reference

    logger.debug(
        "AgentTask %s turn finished: %s (agent=%s, result=%s)",
        task_id,
        status,
        persistence_reference("agent", agent_id, namespace="agent-dispatch-worker"),
        result,
    )
    return status


# ── execution (the existing bodies, relocated) ─────────────────────────────


def _execute_goal_turn(spec: dict[str, Any]) -> str:
    """Run the claimed goal via the EXISTING ``run_goal_loop`` body."""
    import asyncio

    from agent_utilities.core.resource_priority import (
        PriorityClass,
        priority_scope,
    )
    from agent_utilities.core.sessions import run_goal_loop

    # CONCEPT:AU-ORCH.scheduling.resource-priority-edict — the autonomous goal loop is
    # BACKGROUND work: run_goal_loop → LoopController.run_loop issues per-tick engine
    # writes (work_items.claim / lifecycle.batch_update) against the single-writer engine
    # every ~2s. UNTAGGED, that context resolves to ORCHESTRATION (high, never yields), so
    # the loop saturates the write lock and starves interactive delegations' RAG reads.
    # Tag the whole loop BACKGROUND_INGESTION so every engine call it makes stamps the
    # yielding priority claim (session.engine_verified_context → the engine's reserved-read
    # QoS lane keys off it) and its shared-LLM calls yield the reserved headroom. asyncio.run
    # copies this context into the loop coroutine; mirrors engine_tasks.py's per-task
    # priority_scope for background KG work.
    with priority_scope(PriorityClass.BACKGROUND_INGESTION):
        asyncio.run(
            run_goal_loop(
                session_id=spec["session_id"],
                goal_id=spec["goal_id"],
                objective=spec["objective"],
                validation_cmd=spec.get("validation_cmd", ""),
                max_iterations=int(spec.get("max_iterations", 20)),
                constraints=list(spec.get("constraints", [])),
            )
        )
    return "completed"


def _execute_orchestrator_turn(
    engine: Any,
    envelope: AgentTurnEnvelope,
    claim: dict[str, Any],
    *,
    claim_ttl_s: float,
) -> str:
    """Run the claimed orchestrator job via the existing agent execution path.

    The native WorkItem lease and idempotent fenced commit are the sole durable
    outcome authority. A redelivery can execute only after the engine reclaims
    an expired lease; no second checkpoint database can race the WorkItem.
    """
    import asyncio

    from agent_utilities.orchestration.manager import Orchestrator

    orch = Orchestrator(engine)

    async def _invoke() -> Any:
        # D-25-4: pin run_id to the orchestrator job_id (never leave it to
        # execute_agent's own new_run_id() default) so the :RunTrace this run
        # writes is deterministically findable by the SAME id a caller already
        # holds (the WorkItem/job_id) -- see manager.get_run_trace's own
        # id derivation (observability.trace_ontology.trace_id). Without this,
        # the real output execute_agent returns was computed and then
        # discarded: only an opaque result_ref marker survived past this
        # function, and no caller had any id to look the real output up by.
        return await orch.execute_agent(
            agent_name=envelope.agent_name,
            task=claim["description"],
            session_id=envelope.session_id,
            run_id=envelope.job_id,
        )

    from agent_utilities.orchestration import work_item as _wi

    work_engine = getattr(engine, "_work_item_engine", engine)
    item_id = str(claim["work_item_id"])
    lease = WorkItemLeaseGuard(
        work_engine,
        item_id,
        claim,
        lease_ttl_s=claim_ttl_s,
    )
    try:
        lease.start()
        try:
            lease.require_current()
            output = asyncio.run(_invoke())
        except WorkItemLeaseLost:
            raise
        except Exception as e:  # noqa: BLE001 — durably mark failed, then ack
            committed = lease.side_effect(
                _wi.commit_result,
                work_engine,
                item_id,
                claim,
                outcome="failed",
                error_ref=f"orchestrator:{envelope.job_id}:failed",
                retryable=False,
            )
            if committed not in {"committed", "noop"}:
                raise _wi.WorkItemBackendUnavailable(
                    f"orchestrator WorkItem failure commit rejected ({committed})"
                ) from e
            return "failed"

        committed = lease.side_effect(
            _wi.commit_result,
            work_engine,
            item_id,
            claim,
            outcome="succeeded",
            result_ref=f"orchestrator:{envelope.job_id}:completed",
            retryable=False,
        )
        if committed not in {"committed", "noop"}:
            raise _wi.WorkItemBackendUnavailable(
                f"orchestrator WorkItem success commit rejected ({committed})"
            )
    finally:
        lease.close()

    from agent_utilities.messaging.bus_privacy import bus_reference

    logger.info(
        "Orchestrator WorkItem %s completed (result_chars=%d)",
        bus_reference("work_item", item_id),
        len(str(output)),
    )
    return "completed"


def _fail_expired(envelope: AgentTurnEnvelope, engine: Any) -> None:
    """Cancel the payload WorkItem for a past-deadline dispatch."""
    if envelope.kind == KIND_GOAL_LOOP:
        from agent_utilities.core import sessions as _sessions

        gid = envelope.payload_ref
        goal_engine = _sessions._goal_engine()
        if goal_engine is None:
            logger.error("No KG engine — cannot expire goal %s.", gid)
            return
        try:
            from agent_utilities.orchestration import work_item as _wi

            item_id = _wi.loop_work_item_id(gid)
            # A goal past its dispatch deadline before ever being claimed has
            # no loop WorkItem yet — create_goal registers only the dispatch
            # envelope's own WorkItem eagerly (submit_work_item above); the
            # goal's OWN loop WorkItem is otherwise created lazily by
            # run_goal_loop's submit_loop() on first execution, which this
            # goal never reached. cancel_work_item on a nonexistent item is a
            # silent no-op, so the goal's status (the WorkItem is the sole
            # authority, AU-P1-1) would incorrectly stay "submitted" forever
            # instead of the "cancelled" terminal outcome this expiry
            # represents. ensure_loop_work_item is idempotent (upserts), so
            # this is a no-op when the item already exists (e.g. a goal that
            # WAS claimed at least once before this deadline check).
            _wi.ensure_loop_work_item(goal_engine, gid)
            if _wi.cancel_work_item(
                goal_engine,
                item_id,
                reason="dispatch_deadline_expired",
            ):
                logger.info("Cancelled expired goal WorkItem %s", item_id)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to expire goal: %s", e)
    elif envelope.kind == KIND_ORCHESTRATOR_TASK and engine is not None:
        try:
            from agent_utilities.orchestration import work_item as _wi

            view = getattr(engine, "_work_item_engine", engine)
            item_id = _wi.orchestrator_work_item_id(envelope.payload_ref)
            if not _wi.cancel_work_item(
                view, item_id, reason="dispatch_deadline_expired", now=time.time()
            ):
                return
            logger.info("Cancelled expired orchestrator WorkItem %s", item_id)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to expire task %s: %s", envelope.payload_ref, e)


def execute_agent_turn(
    envelope: AgentTurnEnvelope,
    engine: Any = None,
    *,
    token: str | None = None,
    now: float | None = None,
    claim_ttl_s: float | None = None,
) -> str:
    """Claim + execute + write back ONE dispatched turn; return the outcome.

    Outcomes: ``completed`` | ``failed`` | ``skipped`` (duplicate delivery /
    live claim elsewhere) | ``expired`` (deadline passed). The whole cycle
    holds the per-session guard — one executor per session, fleet-wide.
    """
    token = token or worker_token()
    claim_ttl_s = _claim_ttl_seconds(claim_ttl_s)
    if engine is None and envelope.kind == KIND_GOAL_LOOP:
        from agent_utilities.core import sessions as _sessions

        engine = _sessions._goal_engine()
    if engine is None:
        raise RuntimeError("agent turn dispatch requires the process graph authority")

    from agent_utilities.orchestration import work_item as _wi

    dispatch_item_id = f"workitem:dispatch:{envelope.job_id}"
    with session_execution_guard(envelope.session_id):
        if envelope.deadline_unix and (now or time.time()) > envelope.deadline_unix:
            _wi.cancel_work_item(
                engine, dispatch_item_id, reason="dispatch_deadline_expired", now=now
            )
            _fail_expired(envelope, engine)
            return "expired"
        dispatch_claim = _wi.claim_specific(
            engine,
            dispatch_item_id,
            token=token,
            now=now,
            lease_ttl_s=claim_ttl_s,
        )
        if dispatch_claim is None:
            return "skipped"  # authoritative negative; no legacy claim fallback
        if not _wi.mark_running(engine, dispatch_item_id, dispatch_claim, now=now):
            return "skipped"
        lease = WorkItemLeaseGuard(
            engine,
            dispatch_item_id,
            dispatch_claim,
            lease_ttl_s=claim_ttl_s,
        )
        try:
            lease.start()
            outcome = "failed"
            error_detail = ""
            try:
                if envelope.kind == KIND_GOAL_LOOP:
                    spec = load_goal_run(
                        envelope.payload_ref,
                        token=token,
                        now=now,
                    )
                    if spec is not None:
                        lease.require_current()
                        outcome = _execute_goal_turn(spec)
                elif envelope.kind == KIND_ORCHESTRATOR_TASK:
                    claim = lease.side_effect(
                        claim_orchestrator_work_item,
                        engine,
                        envelope.payload_ref,
                        token=token,
                        now=now,
                        claim_ttl_s=claim_ttl_s,
                    )
                    if claim is not None:
                        outcome = _execute_orchestrator_turn(
                            engine,
                            envelope,
                            claim,
                            claim_ttl_s=claim_ttl_s,
                        )
                else:
                    from agent_utilities.messaging.bus_privacy import bus_reference

                    logger.error(
                        "Unknown dispatch kind %r (job_ref=%s)",
                        envelope.kind,
                        bus_reference(
                            "dispatch_job", envelope.job_id, tenant=envelope.tenant
                        ),
                    )
            except WorkItemLeaseLost:
                raise
            except Exception as e:  # noqa: BLE001 — BUG-003: a turn-execution
                # exception must land on the SAME durable commit_result call
                # below, never escape past this point. Before this guard, an
                # exception from `_execute_goal_turn` (no inner try existed
                # for the goal_loop branch) propagated straight out of
                # `execute_agent_turn` with the dispatch WorkItem still
                # non-terminal (`leased`/`running`) — the outer consumer
                # loop's catch-all then logged it and acked the broker
                # message anyway (CONCEPT:AU-OS.governance.verified-write-state-advance:
                # the ack is the "advance", and it must never precede its
                # write's confirmed result).
                logger.error(
                    "agent-dispatch turn execution error (%s)", type(e).__name__
                )
                outcome = "failed"
                error_detail = f"{type(e).__name__}: {e}"[:500]
            committed = lease.side_effect(
                _wi.commit_result,
                engine,
                dispatch_item_id,
                dispatch_claim,
                outcome="succeeded" if outcome == "completed" else "failed",
                result_ref=f"dispatch:{envelope.job_id}:completed"
                if outcome == "completed"
                else None,
                error_ref=f"dispatch:{envelope.job_id}:{error_detail or outcome}"
                if outcome != "completed"
                else None,
                retryable=False,
            )
            if committed not in {"committed", "noop"}:
                raise _wi.WorkItemBackendUnavailable(
                    f"agent turn commit was rejected ({committed})"
                )
            return outcome
        except WorkItemLeaseLost:
            return "fenced"
        finally:
            lease.close()


# ── consumer loop / pool ───────────────────────────────────────────────────


#: Seconds between fleet-registry heartbeats (and metric gauge refreshes).
HEARTBEAT_INTERVAL_S = 30.0


def _heartbeat(queue: Any, worker_id: str, active_sessions: list[str]) -> None:
    """Register liveness + refresh the ORCH-1.45 gauges (never load-bearing)."""
    from agent_utilities.orchestration.agent_dispatch import (
        dispatch_queue_depth,
        list_dispatch_workers,
        record_dispatch_worker_heartbeat,
    )

    backend = type(queue).__name__
    try:
        record_dispatch_worker_heartbeat(
            worker_id,
            capacity=1,
            active_sessions=active_sessions,
            queue_backend=backend,
        )
    except Exception as e:  # noqa: BLE001 — docstring: "(never load-bearing)"; run_dispatch_consumer_loop's claim->execute->ack cycle never reads this heartbeat, it only affects fleet-registry liveness visibility
        logger.debug("dispatch worker heartbeat failed: %s", e)
        return
    try:
        from agent_utilities.observability.gateway_metrics import (
            DISPATCH_QUEUE_DEPTH,
            DISPATCH_WORKERS,
        )

        DISPATCH_QUEUE_DEPTH.labels(backend=backend).set(
            float(dispatch_queue_depth(queue))
        )
        DISPATCH_WORKERS.set(float(len(list_dispatch_workers())))
    except Exception as e:  # noqa: BLE001 — same heartbeat contract as above; only refreshes DISPATCH_QUEUE_DEPTH/DISPATCH_WORKERS Prometheus gauges, no dispatch-correctness dependency
        logger.debug("dispatch metrics refresh failed: %s", e)


def _poison_delivery_digest(payload: Any) -> str:
    """Stable digest over one raw, unparseable queue payload.

    The DLQ idempotency key for a poison envelope (BUG-003): an identical
    redelivery of the same malformed message must resolve to the SAME
    durable dead-letter record instead of accumulating duplicates.
    """
    try:
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    except (TypeError, ValueError):
        blob = repr(payload).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def poison_work_item_id(payload: Any) -> str:
    """Deterministic durable-DLQ WorkItem id for one poison payload.

    Public so callers/tests can predict or look up the record without
    re-deriving the digest algorithm.
    """
    return f"workitem:dispatch:poison:{_poison_delivery_digest(payload)}"


def _dead_letter_poison_envelope(
    engine: Any, payload: Any, *, error: BaseException
) -> str | None:
    """Durably record an unparseable dispatch envelope BEFORE it may be acked.

    CONCEPT: BUG-003 remediation — reuses the SAME WorkItem dead-letter
    machinery every other durable failure in this codebase drains through
    (:mod:`agent_utilities.knowledge_graph.ingestion.dead_letter` lists any
    WorkItem whose ``status == "dead_letter"``, regardless of ``kind`` — a
    poison record submitted with ``kind="dispatch_poison"`` is visible there
    for free). Keyed by :func:`poison_work_item_id` so a redelivered poison
    payload resolves to the SAME durable record. ``max_attempts=1`` plus
    ``retryable=True`` drives the FIRST commit straight through the engine's
    existing retry-exhaustion path into ``dead_letter`` — no separate DLQ
    write path, no second state authority.

    Returns the dead-lettered WorkItem id, or ``None`` if durability could
    not be confirmed. The caller MUST then withhold the ack: a poison record
    that isn't confirmed durable is not a safe ack gate, and this function is
    itself idempotent on retry (redelivery will call it again).
    """
    if engine is None:
        return None
    from agent_utilities.orchestration import work_item as _wi

    work_item_id = poison_work_item_id(payload)
    try:
        existing = _wi.get_work_item(engine, work_item_id)
        if existing is not None:
            return (
                work_item_id
                if existing.get("status") in _wi.TERMINAL_WORK_ITEM_STATUSES
                else None
            )
        token = worker_token()
        _wi.submit_work_item(
            engine,
            kind="dispatch_poison",
            payload_ref=f"poison:{_poison_delivery_digest(payload)}",
            tenant="__system__",
            resource_class="agent_dispatch",
            work_item_id=work_item_id,
            idempotency_key=work_item_id,
            max_attempts=1,
            description=f"poison agent-dispatch envelope ({type(error).__name__})",
            metadata={"error_type": type(error).__name__},
        )
        claim = _wi.claim_specific(engine, work_item_id, token=token)
        if claim is None:
            # A racing redelivery already claimed/finished it — re-read: the
            # record is durable either way, or genuinely isn't yet, in which
            # case the caller withholds ack and this retries on redelivery.
            current = _wi.get_work_item(engine, work_item_id)
            return (
                work_item_id
                if current is not None
                and current.get("status") in _wi.TERMINAL_WORK_ITEM_STATUSES
                else None
            )
        _wi.mark_running(engine, work_item_id, claim)
        committed = _wi.commit_result(
            engine,
            work_item_id,
            claim,
            outcome="failed",
            error_ref=f"poison:{type(error).__name__}: {str(error)[:200]}",
            # max_attempts=1 + retryable=True -> the engine's existing
            # retry-exhaustion path lands this straight in "dead_letter".
            retryable=True,
        )
        if committed not in {"committed", "dead_letter", "noop"}:
            return None
        return work_item_id
    except Exception as exc:  # noqa: BLE001 — durability confirmation fails closed
        logger.error("agent-dispatch poison DLQ write failed (%s)", type(exc).__name__)
        return None


def _ack_after_durable_outcome(
    queue: Any,
    item_id: Any,
    engine: Any,
    work_item_id: str | None,
) -> bool:
    """The ONE broker-ack chokepoint (CONCEPT: BUG-003 remediation).

    Re-reads the authoritative WorkItem status and acks the transport ONLY
    when durable terminal state (``succeeded``/``failed``/``cancelled``/
    ``dead_letter``) is confirmed present — never on a local ``outcome``
    variable alone, which a crash/exception between execute and commit can
    leave disagreeing with reality. Returns whether the ack fired.

    A withheld ack is always safe: at-least-once redelivery retries the
    message, and every durable write this worker makes (claim, commit,
    poison-DLQ) is itself idempotent on redelivery — see
    :func:`_dead_letter_poison_envelope` and ``work_item.commit_result``'s
    idempotency-key docstring.
    """
    from agent_utilities.orchestration import work_item as _wi

    if work_item_id is None:
        logger.warning("agent-dispatch ack withheld: no durable WorkItem id")
        return False
    item = _wi.get_work_item(engine, work_item_id) if engine is not None else None
    if item is None:
        logger.warning(
            "agent-dispatch ack withheld: no durable WorkItem for %s", work_item_id
        )
        return False
    if item.get("status") not in _wi.TERMINAL_WORK_ITEM_STATUSES:
        logger.warning(
            "agent-dispatch ack withheld: WorkItem %s is %r, not terminal",
            work_item_id,
            item.get("status"),
        )
        return False
    try:
        queue.ack(item_id)
    except Exception as e:  # noqa: BLE001 — redelivery is safe (idempotent)
        logger.warning(
            "agent-dispatch ack failed (%s); redelivery is safe.", type(e).__name__
        )
        return False
    return True


def run_dispatch_consumer_loop(
    queue: Any,
    stop_event: threading.Event,
    engine: Any = None,
    *,
    worker_id: str | None = None,
    idle_sleep_s: float = 0.5,
    heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S,
) -> None:
    """Drain ``agent_turns`` until ``stop_event``: claim → execute → ack.

    The ack/commit happens strictly AFTER the turn is processed or durably
    marked failed (at-least-once); a poisonous envelope is acked after its
    failure is recorded so it never wedges the loop, exactly like the
    ingest consumer (KG-2.57). Between turns the worker heartbeats into the
    fleet registry, so ``/api/fleet/topology`` shows it (placement is
    queue-pull: workers claim work when they have capacity — no central
    placer to fail or rebalance; see ``orchestration/agent_dispatch.py``).
    """
    if engine is None:
        # Mirror execute_agent_turn's own auto-resolve courtesy (it falls
        # back to the process-wide engine for goal_loop turns when a caller
        # leaves ``engine`` unset) -- this loop's poison/dead-letter branch
        # calls _dead_letter_poison_envelope(engine, ...) directly, BEFORE
        # ever reaching execute_agent_turn's own resolution, so without this
        # a caller that (like execute_agent_turn's callers) relies on the
        # already-active process engine would silently dead-letter nothing
        # and withhold every ack forever.
        from agent_utilities.core import sessions as _sessions

        engine = _sessions._goal_engine()
    token = worker_id or worker_token()
    active: list[str] = []
    next_heartbeat = 0.0
    while not stop_event.is_set():
        if time.monotonic() >= next_heartbeat:
            _heartbeat(queue, token, active)
            next_heartbeat = time.monotonic() + heartbeat_interval_s

        try:
            item = queue.get()
        except Exception as e:  # noqa: BLE001 — transport hiccup: back off, retry
            logger.warning("agent-dispatch poll error (%s)", type(e).__name__)
            time.sleep(2.0)
            continue
        if item is None:
            time.sleep(idle_sleep_s)
            continue

        item_id, payload = item
        try:
            envelope = AgentTurnEnvelope.from_item(payload)
        except Exception as e:  # noqa: BLE001 — BUG-003: poison envelope. A
            # durable dead-letter record MUST exist before this message may
            # ever be acked — the prior behavior (log + unconditional ack)
            # dropped the message and every trace of its failure together.
            logger.error("agent-dispatch poison envelope (%s)", type(e).__name__)
            poison_id = _dead_letter_poison_envelope(engine, payload, error=e)
            _record_turn_outcome("poison" if poison_id else "poison_unrecorded")
            if not _ack_after_durable_outcome(queue, item_id, engine, poison_id):
                time.sleep(idle_sleep_s)
            continue

        dispatch_item_id = f"workitem:dispatch:{envelope.job_id}"

        # CONCEPT: GOC-18 defense in depth — reject a wire tenant that
        # disagrees with the tenant this WorkItem was durably admitted under,
        # BEFORE ever claiming or executing it. Reads the WorkItem directly
        # (never the claim response — ``work_item.claim_specific``'s
        # ``_normalize_native_claim`` does not yet surface tenant on the
        # claim itself; that gap is tracked separately) so this check is
        # unaffected by that surface. A missing WorkItem or a missing/blank
        # envelope tenant is not a mismatch — the existing claim/skip and
        # producer-side admission checks own those cases.
        if engine is not None and envelope.tenant:
            from agent_utilities.orchestration import work_item as _wi

            admitted_item = _wi.get_work_item(engine, dispatch_item_id)
            admitted_tenant = admitted_item.get("tenant") if admitted_item else None
            if admitted_tenant and envelope.tenant != admitted_tenant:
                logger.error(
                    "agent-dispatch tenant mismatch for %s: wire tenant "
                    "disagrees with the admitted WorkItem — rejecting delivery",
                    dispatch_item_id,
                )
                mismatch = TenantMismatchError(
                    f"envelope tenant does not match the WorkItem {dispatch_item_id} "
                    "was admitted under"
                )
                poison_id = _dead_letter_poison_envelope(engine, payload, error=mismatch)
                _record_turn_outcome(
                    "tenant_mismatch" if poison_id else "tenant_mismatch_unrecorded"
                )
                if not _ack_after_durable_outcome(queue, item_id, engine, poison_id):
                    time.sleep(idle_sleep_s)
                continue

        outcome = "failed"
        try:
            active[:] = [envelope.session_id]
            _heartbeat(queue, token, active)
            next_heartbeat = time.monotonic() + heartbeat_interval_s
            outcome = execute_agent_turn(envelope, engine, token=token)
        except Exception as e:  # noqa: BLE001 — record + keep consuming; the
            # ack gate below withholds ack unless a durable terminal state
            # is confirmed, so this catch-all can no longer mask data loss.
            logger.error("agent-dispatch worker error (%s)", type(e).__name__)
            outcome = "failed"
        finally:
            active.clear()
        _record_turn_outcome(outcome)
        if outcome == "fenced":
            # The message remains unacknowledged so Kafka/Postgres can redeliver
            # after the current claim is replaced or expires. Acknowledging a
            # stale execution would turn lease loss into data loss.
            time.sleep(idle_sleep_s)
            continue
        if outcome == "skipped":
            # No new durable state was produced by THIS delivery attempt (a
            # duplicate of an already-terminal item, or a live claim held
            # elsewhere) — nothing new to protect, so ack directly. Mirrors
            # the ingest worker's identical idempotent-skip-then-ack pattern.
            try:
                queue.ack(item_id)
            except Exception as e:  # noqa: BLE001 — redelivery is safe (idempotent)
                logger.warning(
                    "agent-dispatch ack failed (%s); redelivery is safe.",
                    type(e).__name__,
                )
            continue
        # The ONE ack chokepoint (CONCEPT: BUG-003): re-reads the durable
        # WorkItem before acking, regardless of what the local `outcome`
        # variable claims — see `_ack_after_durable_outcome`'s docstring.
        if not _ack_after_durable_outcome(queue, item_id, engine, dispatch_item_id):
            time.sleep(idle_sleep_s)


def _record_turn_outcome(outcome: str) -> None:
    """Count one processed turn on the OS-5.23 metrics registry."""
    try:
        from agent_utilities.observability.gateway_metrics import DISPATCH_TURNS

        DISPATCH_TURNS.labels(outcome=outcome).inc()
    except Exception:  # noqa: BLE001 — metrics are never load-bearing
        pass


def start_dispatch_worker_pool(
    queue: Any = None,
    *,
    worker_count: int = 1,
    stop_event: threading.Event | None = None,
    engine: Any = None,
    background_session: Any = None,
) -> list[threading.Thread]:
    """Start ``worker_count`` dispatch consumer threads against ``queue``.

    CONCEPT: BUG-002 remediation — mirrors
    :func:`~agent_utilities.knowledge_graph.ingest_worker.start_ingest_consumer_pool`
    exactly: a verified process actor/GraphSession is captured (or accepted
    explicitly via ``background_session``) BEFORE any thread is spawned —
    ``ContextVar``-scoped identity does not cross a bare
    :class:`threading.Thread` boundary, so each worker thread is started
    through :func:`~agent_utilities.knowledge_graph.core.engine_tasks
    ._authorized_background_thread`, which re-binds that SAME captured
    authority inside the new thread rather than running unauthenticated.
    Absence of a verified actor/session is a hard failure here (fail
    closed), never a reason to spawn an anonymous worker.

    A shared Kafka backend allocates one thread-local consumer per worker and
    binds each acknowledgement receipt to that owner (confluent consumers are
    not thread-safe). SQLite/Postgres backends remain safe to share.
    """
    from agent_utilities.knowledge_graph.core.engine_tasks import (
        _authorized_background_thread,
        _capture_verified_background_session,
    )

    worker_session = background_session or _capture_verified_background_session()

    stop = stop_event or threading.Event()
    threads: list[threading.Thread] = []
    for i in range(max(1, worker_count)):
        q = queue if queue is not None else get_dispatch_queue()

        def _runner(q: Any = q, idx: int = i) -> None:
            run_dispatch_consumer_loop(
                q, stop, engine, worker_id=f"{worker_token()}:{idx}"
            )

        t = _authorized_background_thread(
            worker_session, _runner, name=f"AgentDispatchWorker-{i}"
        )
        t.start()
        threads.append(t)
    logger.info(
        "agent-dispatch worker pool started: %d workers, group=%s",
        len(threads),
        DISPATCH_GROUP,
    )
    return threads


def main(argv: list[str] | None = None) -> int:
    """Entry point: a standalone, host-role-free agent dispatch worker."""
    import argparse
    import signal

    parser = argparse.ArgumentParser(
        prog="agent-dispatch-worker",
        description=(
            "Stateless agent dispatch worker (CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch): consumes "
            f"session-keyed agent turns (group '{DISPATCH_GROUP}') and "
            "executes them as an engine client — no KG host role required."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Consumer threads on this host (default: 1; turns are LLM-bound).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    # Engine-client posture (CONCEPT:EG-KG.storage.nonblocking-checkpoint/OS-5.9): never contend for the host
    # flock, never spawn the consolidated daemon — this process only consumes.
    os.environ.setdefault("KG_DAEMON_ROLE", "client")

    # CONCEPT: BUG-002 remediation — bind a verified process actor/GraphSession
    # BEFORE the first protected engine call, exactly like
    # knowledge_graph.ingest_worker.main() already does. Before this fix the
    # dispatch worker went straight from KG_DAEMON_ROLE=client to
    # IntelligenceGraphEngine.get_or_create() with no
    # acquire_process_identity_token / mint_actor_from_token_sync /
    # mint_graph_session call anywhere — it dispatched (and every claim/commit
    # it made) without ever presenting an authenticated identity.
    from agent_utilities.core.config import config
    from agent_utilities.knowledge_graph.core.session import use_session
    from agent_utilities.security.brain_context import use_actor
    from agent_utilities.security.request_identity import (
        acquire_process_identity_token,
        mint_actor_from_token_sync,
        mint_graph_session,
    )

    token = acquire_process_identity_token(config)
    actor = mint_actor_from_token_sync(token)
    session = mint_graph_session(actor)
    session.engine_verified_context()

    with use_actor(session.actor), use_session(session):
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine.get_or_create()

        # Verify the client/auth path (CONCEPT:AU-OS.identity.authenticated-identity-enforcement) BEFORE consuming: a worker
        # that cannot reach the engine must fail loud, not claim turns and drop them.
        try:
            from agent_utilities.orchestration import work_item as _wi

            if not callable(getattr(engine, "claim_work_item", None)):
                raise _wi.NativeWorkItemRequired(
                    "engine does not expose native ClaimWorkItem"
                )
            engine.query_cypher("MATCH (w:WorkItem) RETURN count(w) AS c")
        except Exception as e:  # noqa: BLE001
            parser.exit(
                2,
                "Cannot reach the epistemic-graph engine as a client: "
                f"error_type={type(e).__name__}\nCheck GRAPH_SERVICE_ENDPOINTS "
                "(external) or the packaged-local "
                "transport and shared HMAC secret "
                "(GRAPH_SERVICE_AUTH_SECRET or the host's data_dir()/engine_secret "
                "— CONCEPT:AU-OS.identity.authenticated-identity-enforcement).\n",
            )

        stop = threading.Event()

        def _shutdown(signum: int, _frame: Any) -> None:
            logger.info("Signal %s received — draining and stopping workers.", signum)
            stop.set()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        threads = start_dispatch_worker_pool(
            worker_count=args.workers,
            stop_event=stop,
            engine=engine,
            background_session=session,
        )
        while any(t.is_alive() for t in threads) and not stop.is_set():
            time.sleep(1.0)
        for t in threads:
            t.join(timeout=10.0)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
