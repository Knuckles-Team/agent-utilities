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
  for the KG host flock.

Run::

    python -m agent_utilities.orchestration.agent_dispatch_worker [--workers N]
    # or the console script:
    agent-dispatch-worker
"""

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
            "MATCH (c:Concept) WHERE c.id = $id RETURN c.id AS goal_id, "
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
            "MATCH (l:AgentLease {resource_id: $rid}) RETURN l.owner_token AS owner_token, "
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
) -> None:
    """Write provenance for one executed WorkItem.

    Best-effort (mirrors ``action_policy._audit``'s posture: an audit-write
    failure never unwinds the decision/execution that already happened).
    """
    if engine is None:
        return
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
        obs_props = observation.model_dump(exclude={"id", "type"})
        obs_props["work_item_id"] = work_item_id
        obs_props["lease_id"] = lease_id
        engine.add_node(obs_id, "Observation", properties=obs_props)

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
        claim_props = policy_claim.model_dump(exclude={"id", "type"})
        claim_props["work_item_id"] = work_item_id
        claim_props["policy_decision_id"] = policy_decision_node.id
        engine.add_node(claim_node_id, "Claim", properties=claim_props)

        action = ActionNode(
            id=action_id,
            name=f"Action: execute {work_item_id}",
            action_type="work_item.execute",
            status=status,
            result=clean_result[:4000],
        )
        action_props = action.model_dump(exclude={"id", "type"})
        action_props["work_item_id"] = work_item_id
        action_props["lease_id"] = lease_id
        action_props["policy_decision_id"] = policy_decision_node.id
        action_props["capability_grant_id"] = grant_id or ""
        action_props["agent_id"] = agent_ref
        engine.add_node(action_id, "Action", properties=action_props)

        trace = TraceNode(
            id=trace_id,
            name=f"Trace: work_item {work_item_id}",
            agent=agent_ref or None,
            task_id=work_item_id,
            status="ok" if status == "completed" else "error",
            outcome=status,
        )
        trace_props = trace.model_dump(exclude={"id", "type"})
        trace_props["lease_id"] = lease_id
        engine.add_node(trace_id, "Trace", properties=trace_props)
    except Exception as e:  # noqa: BLE001 — provenance is audit, never blocks the outcome
        logger.warning("work-item provenance write failed (%s)", type(e).__name__)


def _finalize_work_item(
    engine: Any,
    work_item_id: str,
    claim: dict[str, Any],
    *,
    status: str,
    reward: float,
    feedback_text: str,
) -> str:
    """Commit the WorkItem, then append non-authoritative outcome evidence."""
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
        engine.add_node(
            outcome_id,
            "OutcomeEvaluation",
            properties=outcome.model_dump(exclude={"id", "type"}),
        )
    except Exception as e:  # noqa: BLE001 — writeback is durable-best-effort
        logger.warning("work-item outcome append failed (%s)", type(e).__name__)
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
    ``"failed"`` (the executor ran; writeback recorded). Never raises — an
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
        _write_work_item_provenance(
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
    _write_work_item_provenance(
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
    return status


# ── execution (the existing bodies, relocated) ─────────────────────────────


def _execute_goal_turn(spec: dict[str, Any]) -> str:
    """Run the claimed goal via the EXISTING ``run_goal_loop`` body."""
    import asyncio

    from agent_utilities.core.sessions import run_goal_loop

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
        return await orch.execute_agent(
            agent_name=envelope.agent_name,
            task=claim["description"],
            session_id=envelope.session_id,
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
                outcome=_wi.WorkItemStatus.FAILED.value,
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
            outcome=_wi.WorkItemStatus.SUCCEEDED.value,
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
            if _wi.cancel_work_item(
                goal_engine,
                item_id,
                reason="dispatch_deadline_expired",
            ):
                logger.info("Cancelled expired goal WorkItem %s", item_id)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to expire goal (%s)", type(e).__name__)
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
            committed = lease.side_effect(
                _wi.commit_result,
                engine,
                dispatch_item_id,
                dispatch_claim,
                outcome="succeeded" if outcome == "completed" else "failed",
                result_ref=f"dispatch:{envelope.job_id}:completed"
                if outcome == "completed"
                else None,
                error_ref=f"dispatch:{envelope.job_id}:{outcome}"
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
    except Exception as e:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
        logger.debug("dispatch metrics refresh failed: %s", e)


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
        outcome = "failed"
        try:
            envelope = AgentTurnEnvelope.from_item(payload)
            active[:] = [envelope.session_id]
            _heartbeat(queue, token, active)
            next_heartbeat = time.monotonic() + heartbeat_interval_s
            outcome = execute_agent_turn(envelope, engine, token=token)
        except Exception as e:  # noqa: BLE001 — record + keep consuming
            logger.error("agent-dispatch worker error (%s)", type(e).__name__)
        finally:
            active.clear()
        _record_turn_outcome(outcome)
        if outcome == "fenced":
            # The message remains unacknowledged so Kafka/Postgres can redeliver
            # after the current claim is replaced or expires. Acknowledging a
            # stale execution would turn lease loss into data loss.
            time.sleep(idle_sleep_s)
            continue
        try:
            queue.ack(item_id)
        except Exception as e:  # noqa: BLE001 — redelivery is safe (idempotent)
            logger.warning(
                "agent-dispatch ack failed (%s); redelivery is safe.",
                type(e).__name__,
            )


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
) -> list[threading.Thread]:
    """Start ``worker_count`` dispatch consumer threads against ``queue``.

    A shared Kafka backend allocates one thread-local consumer per worker and
    binds each acknowledgement receipt to that owner (confluent consumers are
    not thread-safe). SQLite/Postgres backends remain safe to share.
    """
    stop = stop_event or threading.Event()
    threads: list[threading.Thread] = []
    for i in range(max(1, worker_count)):
        q = queue if queue is not None else get_dispatch_queue()

        def _runner(q: Any = q, idx: int = i) -> None:
            run_dispatch_consumer_loop(
                q, stop, engine, worker_id=f"{worker_token()}:{idx}"
            )

        t = threading.Thread(
            target=_runner, name=f"AgentDispatchWorker-{i}", daemon=True
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
        worker_count=args.workers, stop_event=stop, engine=engine
    )
    while any(t.is_alive() for t in threads) and not stop.is_set():
        time.sleep(1.0)
    for t in threads:
        t.join(timeout=10.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
