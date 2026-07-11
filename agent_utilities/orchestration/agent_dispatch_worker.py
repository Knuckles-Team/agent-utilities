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
import socket
import threading
import time
import uuid
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

#: A 'running' claim older than this is presumed dead (its worker crashed
#: between claim and writeback) and may be re-claimed on redelivery. Mirrors
#: the ingest reaper's runtime-cap reasoning, folded into the claim check.
CLAIM_TTL_S = 3600.0

_GOAL_TERMINAL = ("completed", "failed", "cancelled", "paused")
_TASK_TERMINAL = ("completed", "failed", "cancelled")
#: Terminal statuses for a durable ``:AgentTask`` (C3/Phase 3a) — same three
#: outcomes as ``_TASK_TERMINAL`` plus ``unroutable`` (AU-P0-3: a task with no
#: bound executor — terminal because redelivery would just hit the identical
#: unroutable outcome again); kept as its own tuple because the two node
#: kinds are independent schemas that may diverge later.
_AGENT_TASK_TERMINAL = ("completed", "failed", "cancelled", "unroutable")


def worker_token() -> str:
    """Stable identity for claims/heartbeats: ``hostname:pid:agent-dispatch``."""
    return f"{socket.gethostname()}:{os.getpid()}:agent-dispatch"


def _turn_correlation_id() -> str:
    """Correlation id stamped on the executed Task node (CONCEPT:AU-OS.observability.run-wide-correlation-id).

    Makes ``/api/fleet/touched`` able to resolve which agent turn touched a task.
    """
    try:
        from agent_utilities.observability import correlation

        return correlation.ensure_correlation_id()
    except Exception:  # noqa: BLE001 — best-effort context
        return ""


# ── claims (idempotent, stale-claim aware) ─────────────────────────────────


def claim_goal_run(
    goal_id: str,
    *,
    token: str | None = None,
    now: float | None = None,
    claim_ttl_s: float = CLAIM_TTL_S,
) -> dict[str, Any] | None:
    """Claim one goal run; return its rehydrated spec, or ``None`` to skip.

    Reads the goal's KG Loop node (CONCEPT:AU-KG.research.these-properties-carry) plus the ``goal_spec``
    persisted in the session's metadata (the envelope carried only the reference).
    Skips terminal/paused goals (duplicate delivery) and goals whose 'running'
    claim is FRESH (a live worker owns them); re-claims stale 'running' and
    'orphaned' goals — that re-claim IS the crash-recovery path. The exactly-once
    effect is now guaranteed at the iteration level by ``DurableExecutionManager``
    (OS-5.16), so this node claim is best-effort owner dedup.
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
            "c.session_id AS session_id, c.status AS status, c.objective AS objective, "
            "c.validation_cmd AS validation_cmd, c.max_iterations AS max_iterations, "
            "c.updated_at AS updated_at",
            {"id": goal_id},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Goal %s claim query failed: %s", goal_id, e)
        return None
    row = next(
        (r for r in (rows or []) if isinstance(r, dict) and r.get("goal_id")), None
    )
    if not row:
        logger.warning("Dispatch envelope for unknown goal %s skipped.", goal_id)
        return None
    status = str(row.get("status") or "")
    if status in _GOAL_TERMINAL:
        logger.debug("Duplicate delivery of goal %s (%s) skipped.", goal_id, status)
        return None
    if status == "running":
        age = now - float(row.get("updated_at") or 0)
        if age < claim_ttl_s:
            logger.debug(
                "Goal %s is running with a fresh claim (%.0fs) — skipping.",
                goal_id,
                age,
            )
            return None
        logger.warning(
            "Re-claiming goal %s: previous claim is stale (%.0fs > %.0fs).",
            goal_id,
            age,
            claim_ttl_s,
        )

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

    # Claim: stamp running + owner onto the Loop node (best-effort owner dedup).
    try:
        engine.add_node(
            goal_id,
            "Concept",
            properties={"status": "running", "owner_host": token, "updated_at": now},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Goal %s claim write failed: %s", goal_id, e)
    return spec


def claim_orchestrator_task(
    engine: Any,
    job_id: str,
    *,
    token: str | None = None,
    now: float | None = None,
    claim_ttl_s: float = CLAIM_TTL_S,
) -> dict[str, Any] | None:
    """Claim one orchestrator ``:Task`` node; return its payload, or ``None``.

    Same idempotency contract as the ingest claim (KG-2.57): terminal statuses
    are duplicate deliveries; a 'running' node with a fresh ``claim_unix`` is
    owned by a live worker; a stale one is re-claimed (crash recovery)."""
    token = token or worker_token()
    now = now if now is not None else time.time()

    rows = engine.query_cypher(
        "MATCH (t:Task {id: $id}) RETURN t.status AS s, t.description AS d, "
        "t.claim_unix AS cu",
        {"id": job_id},
    )
    if not rows:
        logger.warning("Dispatch envelope for unknown task %s skipped.", job_id)
        return None
    row = rows[0]
    status = row.get("s")
    if status in _TASK_TERMINAL:
        logger.debug("Duplicate delivery of task %s (%s) skipped.", job_id, status)
        return None
    if status == "running":
        age = now - float(row.get("cu") or 0)
        if age < claim_ttl_s:
            logger.debug("Task %s running with a fresh claim — skipping.", job_id)
            return None
        logger.warning("Re-claiming task %s: stale claim (%.0fs).", job_id, age)

    engine.add_node(
        job_id,
        "Task",
        properties={
            "status": "running",
            "description": row.get("d") or "",
            "claimed_by": token,
            "claim_unix": now,
            "dispatch_host": socket.gethostname(),
        },
    )
    return {"job_id": job_id, "description": row.get("d") or ""}


def claim_agent_task(
    engine: Any,
    task_id: str,
    *,
    token: str | None = None,
    now: float | None = None,
    claim_ttl_s: float = CLAIM_TTL_S,
) -> dict[str, Any] | None:
    """Claim one durable ``:AgentTask`` node; return its payload, or ``None`` to skip.

    CONCEPT:AU-OS.state.cognitive-scheduler-preemption — Graph-Native Agent-OS Objects (C3/Phase 3a)

    Generalizes :func:`claim_goal_run`/:func:`claim_orchestrator_task`
    (same stale-claim-aware idempotency contract, same ``CLAIM_TTL_S`` /
    :func:`worker_token`) from stamping ownership inline on the claimed node
    to a dedicated ``:AgentLease`` node keyed by ``resource_id == task_id``
    — the reusable claim primitive C3/Phase 3a introduces so any resource
    (not only dispatch turns) can be leased identically. Terminal task
    statuses are duplicate-delivery skips; a lease with a fresh
    ``lease_expires_at`` is owned by a live worker (skip); a lease past
    that deadline is re-claimed — the same crash-recovery path as the two
    claims above, now over ``:AgentLease`` instead of an inline stamp.
    """
    token = token or worker_token()
    now = now if now is not None else time.time()

    rows = engine.query_cypher(
        "MATCH (t:AgentTask {id: $id}) RETURN t.status AS status, "
        "t.depends_on_task_ids AS depends_on_task_ids, t.dag_id AS dag_id, "
        "t.checkpoint_id AS checkpoint_id",
        {"id": task_id},
    )
    if not rows:
        logger.warning("Dispatch envelope for unknown agent task %s skipped.", task_id)
        return None
    row = rows[0]
    status = str(row.get("status") or "")
    if status in _AGENT_TASK_TERMINAL:
        logger.debug(
            "Duplicate delivery of agent task %s (%s) skipped.", task_id, status
        )
        return None

    lease_rows = engine.query_cypher(
        "MATCH (l:AgentLease {resource_id: $rid}) RETURN l.owner_token AS owner_token, "
        "l.lease_expires_at AS lease_expires_at, l.lease_epoch AS lease_epoch "
        "ORDER BY l.acquired_at DESC LIMIT 1",
        {"rid": task_id},
    )
    prior_epoch = 0
    if lease_rows:
        lease = lease_rows[0]
        prior_epoch = int(lease.get("lease_epoch") or 0)
        expires_at = float(lease.get("lease_expires_at") or 0.0)
        if expires_at > now:
            logger.debug(
                "Agent task %s has a fresh lease (owner=%s, %.0fs remaining) — skipping.",
                task_id,
                lease.get("owner_token"),
                expires_at - now,
            )
            return None
        logger.warning(
            "Re-claiming agent task %s: previous lease expired %.0fs ago.",
            task_id,
            now - expires_at,
        )

    # Monotonic fencing token (AU-P0-3): every (re)claim strictly increments
    # the epoch. A holder whose lease later expires and gets re-claimed by
    # someone else is left carrying a STALE (lower) epoch; the finalize-time
    # CAS check (`_fence_still_valid`) rejects any commit from that stale
    # holder even if its execution eventually finishes — the fencing-token
    # guarantee this module's docstring calls out as missing.
    fence_token = prior_epoch + 1
    lease_id = f"lease:{task_id}:{uuid.uuid4().hex[:8]}"
    try:
        engine.add_node(
            lease_id,
            "AgentLease",
            properties={
                "name": f"Lease: {task_id}",
                "owner_token": token,
                "resource_id": task_id,
                "acquired_at": now,
                "lease_expires_at": now + claim_ttl_s,
                "lease_epoch": fence_token,
            },
        )
        engine.add_node(task_id, "AgentTask", properties={"status": "running"})
    except Exception as e:  # noqa: BLE001
        logger.warning("Agent task %s claim write failed: %s", task_id, e)

    return {
        "task_id": task_id,
        "lease_id": lease_id,
        "dag_id": row.get("dag_id") or "",
        "checkpoint_id": row.get("checkpoint_id"),
        "depends_on_task_ids": list(row.get("depends_on_task_ids") or []),
        "fence_token": fence_token,
        # AU-P0-3/L15: marks this claim as the KG best-effort path, so
        # `_fence_still_valid` knows fail-OPEN is defensible for it (see that
        # function's docstring). `engine_claim._try_engine_claim` stamps
        # `"engine"` instead for the engine-native path, which must fail
        # CLOSED on a fence-check error.
        "_claim_backend": "kg",
    }


#: Marker `claim["_claim_backend"]` value stamped by the engine-native claim
#: path (`orchestration.engine_claim._try_engine_claim`). Kept as a bare
#: string literal (not imported from `engine_claim`) to avoid the import
#: cycle documented at the top of `engine_claim.py`; must stay in sync with
#: `engine_claim.AGENT_CLAIM_BACKEND_ENGINE`.
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
    (``claim["_claim_backend"]``, stamped by :func:`claim_agent_task` as
    ``"kg"`` and by ``engine_claim._try_engine_claim`` as ``"engine"``):

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


# ── capability grants (Codex Gap-6) ─────────────────────────────────────────
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
    if engine is None:
        return None
    now = now if now is not None else time.time()
    try:
        rows = engine.query_cypher(
            "MATCH (a:Agent {agent_id: $agent_id})-[:AUTHORIZED_FOR]->"
            "(g:AgentCapabilityGrant {capability: $capability}) "
            "RETURN g.id AS id, g.issuer AS issuer, g.granted_at AS granted_at, "
            "g.expires_at AS expires_at, g.revoked AS revoked "
            "ORDER BY g.granted_at DESC LIMIT 1",
            {"agent_id": agent_id, "capability": capability},
        )
    except Exception as e:  # noqa: BLE001 — resolution is best-effort
        logger.debug(
            "resolve_capability_grant: query failed for %s/%s: %s",
            agent_id,
            capability,
            e,
        )
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
    now = now if now is not None else time.time()
    grant_id = f"capability_grant:{agent_id}:{capability}:{uuid.uuid4().hex[:8]}"
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
        logger.warning(
            "grant_capability: write failed for %s/%s: %s", agent_id, capability, e
        )
        return None
    return grant_id


class NoExecutorBoundError(RuntimeError):
    """Raised by :func:`_default_agent_task_executor`: no concrete executor was
    bound for this ``:AgentTask``, so nothing actually ran.

    Distinguishes an UNROUTABLE task (this) from a real executor failure (any
    other exception raised by a bound executor) while guaranteeing both are
    recorded as unsuccessful — AU-P0-3: unrun work must never be marked
    ``completed`` with ``reward=1.0``.
    """


def _default_agent_task_executor(claim: dict[str, Any]) -> str:
    """Structural default executor: FAILS CLOSED — no executor bound means no
    work ran, so this must never be recorded as a successful completion.

    Concrete ``:AgentTask`` producers (e.g. ``TeamComposition.to_durable_task_dag()``
    callers) should pass a real ``executor=`` callable to
    :func:`execute_agent_task_turn`. Previously this default returned an
    "acknowledged" string and the caller unconditionally set
    ``status="completed"; reward=1.0`` — unrun work was rewarded as if it had
    succeeded. Raising :class:`NoExecutorBoundError` instead routes the task
    through :func:`execute_agent_task_turn`'s failure path (``status=
    "unroutable"``, ``reward=0.0``), the same no-fabrication discipline
    :class:`~agent_utilities.models.evidence_bundle.EvidenceBundle` documents
    for the identical reason.
    """
    raise NoExecutorBoundError(
        f"no executor bound for task {claim.get('task_id')} — task is unroutable"
    )


def _write_agent_task_provenance(
    engine: Any,
    *,
    task_id: str,
    claim: dict[str, Any],
    agent_id: str,
    status: str,
    result: Any,
    evidence: Any,
    policy_decision_node: Any,
    grant_id: str | None,
) -> None:
    """Write the Observation/Claim/Action/AgentTrace provenance for one executed ``:AgentTask``.

    Best-effort (mirrors ``action_policy._audit``'s posture: an audit-write
    failure never unwinds the decision/execution that already happened).
    """
    if engine is None:
        return
    from agent_utilities.models.knowledge_graph import (
        ActionNode,
        AgentTraceNode,
        ClaimNode,
        ObservationNode,
    )

    obs_id = f"observation:agent_task:{task_id}:{uuid.uuid4().hex[:8]}"
    claim_node_id = f"claim:agent_task:{task_id}:{uuid.uuid4().hex[:8]}"
    action_id = f"action:agent_task:{task_id}:{uuid.uuid4().hex[:8]}"
    trace_id = f"trace:agent_task:{task_id}:{uuid.uuid4().hex[:8]}"
    lease_id = claim.get("lease_id", "")
    confidence = getattr(evidence, "confidence", None)
    confidence = confidence if confidence is not None else 1.0

    try:
        observation = ObservationNode(
            id=obs_id,
            name=f"Observation: {task_id}",
            content=(
                f"AgentTask {task_id} claimed via lease {lease_id} "
                f"(dag={claim.get('dag_id') or 'n/a'})"
            ),
            confidence=confidence,
            source="agent-dispatch",
        )
        obs_props = observation.model_dump(exclude={"id", "type"})
        obs_props["task_id"] = task_id
        obs_props["lease_id"] = lease_id
        engine.add_node(obs_id, "Observation", properties=obs_props)

        policy_claim = ClaimNode(
            id=claim_node_id,
            name=f"Claim: {task_id} policy decision",
            claim_text=(
                f"{policy_decision_node.kind}({policy_decision_node.target}) -> "
                f"{policy_decision_node.decision} ({policy_decision_node.reason})"
            ),
            claim_type="decision",
            is_verified=policy_decision_node.allowed,
        )
        claim_props = policy_claim.model_dump(exclude={"id", "type"})
        claim_props["task_id"] = task_id
        claim_props["policy_decision_id"] = policy_decision_node.id
        engine.add_node(claim_node_id, "Claim", properties=claim_props)

        action = ActionNode(
            id=action_id,
            name=f"Action: execute {task_id}",
            action_type="agent_task.execute",
            status=status,
            result=str(result)[:4000],
        )
        action_props = action.model_dump(exclude={"id", "type"})
        action_props["task_id"] = task_id
        action_props["lease_id"] = lease_id
        action_props["policy_decision_id"] = policy_decision_node.id
        action_props["capability_grant_id"] = grant_id or ""
        action_props["agent_id"] = agent_id
        engine.add_node(action_id, "Action", properties=action_props)

        trace = AgentTraceNode(
            id=trace_id,
            name=f"Trace: agent_task {task_id}",
            agent=agent_id or None,
            task_id=task_id,
            status="ok" if status == "completed" else "error",
            outcome=status,
        )
        trace_props = trace.model_dump(exclude={"id", "type"})
        trace_props["lease_id"] = lease_id
        engine.add_node(trace_id, "Trace", properties=trace_props)
    except Exception as e:  # noqa: BLE001 — provenance is audit, never blocks the outcome
        logger.warning("agent_task provenance write failed for %s: %s", task_id, e)


def _finalize_agent_task(
    engine: Any,
    task_id: str,
    claim: dict[str, Any],
    *,
    status: str,
    reward: float,
    feedback_text: str,
) -> None:
    """Writeback: the ``AgentOutcome`` (``OutcomeEvaluationNode``) + the ``:AgentTask`` status flip.

    ``OutcomeEvaluationNode.lease_id``/``dag_id`` were already wired for
    exactly this C3/Phase 3a purpose. The status flip is what
    ``fire_ready_agent_tasks``/the fleet reconciler already polls to wake
    ``TASK_DEPENDS_ON`` dependents (D23/C3) — untouched here, just triggered
    by this write like every other ``:AgentTask`` status transition.

    AU-P1-1: when ``claim`` carries a ``_work_item_id`` (the
    ``AGENT_CLAIM_BACKEND=workitem`` path — see
    ``orchestration.work_item.claim_agent_task_via_work_item``), this ALSO
    commits the outcome through :func:`~agent_utilities.orchestration.
    work_item.commit_agent_task_work_item` so the unified state machine's
    atomic dependency release / DLQ / idempotent-commit mechanics fire. The
    legacy ``:AgentTask`` write above still happens unconditionally — it is
    what unmigrated readers (``fire_ready_agent_tasks``, dashboards) still
    consult, so WorkItem's dependency release and this legacy poll-sweep are
    each other's belt-and-suspenders during the opt-in rollout, not a
    replacement for one another yet.
    """
    if engine is None:
        return
    from agent_utilities.models.knowledge_graph import OutcomeEvaluationNode

    outcome_id = f"outcome:agent_task:{task_id}:{uuid.uuid4().hex[:8]}"
    try:
        outcome = OutcomeEvaluationNode(
            id=outcome_id,
            name=f"Outcome: {task_id}",
            reward=reward,
            feedback_text=feedback_text,
            lease_id=claim.get("lease_id", ""),
            dag_id=claim.get("dag_id", ""),
        )
        engine.add_node(
            outcome_id,
            "OutcomeEvaluation",
            properties=outcome.model_dump(exclude={"id", "type"}),
        )
        engine.add_node(task_id, "AgentTask", properties={"status": status})
    except Exception as e:  # noqa: BLE001 — writeback is durable-best-effort
        logger.warning("agent_task finalize failed for %s: %s", task_id, e)

    work_item_id = claim.get("_work_item_id")
    if work_item_id:
        from agent_utilities.orchestration.work_item import (
            commit_agent_task_work_item,
        )

        commit_agent_task_work_item(engine, work_item_id, claim, status=status)


def execute_agent_task_turn(
    engine: Any,
    task_id: str,
    *,
    agent_id: str = "",
    capability: str = "agent_task.execute",
    executor: Callable[[dict[str, Any]], Any] | None = None,
    evidence: Any = None,
    token: str | None = None,
    now: float | None = None,
    claim_ttl_s: float = CLAIM_TTL_S,
) -> str:
    """Claim -> execute -> writeback ONE durable ``:AgentTask`` (Codex Gap-6 orchestration flow).

    Generalizes :func:`execute_agent_turn`'s claim/execute/writeback shape
    (see :func:`_execute_goal_turn`/:func:`_execute_orchestrator_turn`) to the
    C3 durable ``:AgentTask`` DAG primitive, making the full chain explicit
    end to end::

        ClaimNext (claim_agent_task) -> EvidenceBundle (C1) -> policy frame
        (AgentPolicyDecisionNode over action_policy.decide()) -> capability
        grant (AgentCapabilityGrantNode over AUTHORIZED_FOR) -> execute ->
        Observation/Claim/Action + AgentTrace + AgentOutcome
        (OutcomeEvaluationNode, lease_id/dag_id already wired C3/Phase 3a) ->
        :AgentTask status flip (already polled by fire_ready_agent_tasks /
        the fleet reconciler to wake TASK_DEPENDS_ON dependents — D23/C3,
        untouched here).

    Outcomes: ``"skipped"`` (duplicate delivery / live claim elsewhere, from
    :func:`claim_agent_task`), ``"blocked"`` (action_policy queued the action
    for human approval — the task is left non-terminal so a fresh claim after
    approval retries it), ``"denied"`` (action_policy forbade the action
    outright — terminal), ``"unroutable"`` (no executor was bound — terminal,
    ``reward=0.0``, AU-P0-3 fail-closed), ``"fenced"`` (this holder's lease
    was reclaimed by a newer holder before it could commit — the commit is
    rejected, no writeback happens, AU-P0-3 fencing), ``"completed"`` |
    ``"failed"`` (the executor ran; writeback recorded). Never raises — an
    executor exception is caught and recorded as a failed outcome, mirroring
    :func:`_execute_orchestrator_turn`'s durable failure path.

    The claim itself is routed through :func:`~agent_utilities.orchestration.
    engine_claim.claim_agent_task` (AU-P0-3), which is the KG path here
    (this module's :func:`claim_agent_task`) by default and switches to the
    engine-native ``ClaimNext``/lease/CAS primitive when
    ``AGENT_CLAIM_BACKEND=engine`` resolves live — imported lazily to avoid
    the import cycle (``engine_claim`` imports this module's
    :func:`claim_agent_task` as ITS kg fallback).
    """
    token = token or worker_token()
    now = now if now is not None else time.time()

    from agent_utilities.orchestration import engine_claim

    claim = engine_claim.claim_agent_task(
        engine, task_id, token=token, now=now, claim_ttl_s=claim_ttl_s
    )
    if claim is None:
        return "skipped"

    # EvidenceBundle (C1) — minimal, honest envelope: what is known about this
    # claim before executing. Callers with a real retrieval surface should
    # pass `evidence=` instead of relying on this placeholder.
    if evidence is None:
        from agent_utilities.models.evidence_bundle import EvidenceBundle

        evidence = EvidenceBundle(
            reasoning_trace=[{"step": "agent_task_claim", **claim}]
        )

    # Policy frame (AgentPolicyDecision) — the SAME action_policy gate every
    # other autonomous mutating action goes through.
    from agent_utilities.models.knowledge_graph import AgentPolicyDecisionNode
    from agent_utilities.orchestration.action_policy import (
        DECISION_QUEUE,
        ActionRequest,
        get_action_policy,
    )

    policy_decision = get_action_policy(engine).decide(
        ActionRequest(
            kind="agent_task.execute",
            target=task_id,
            source="agent-dispatch",
            actor_id=agent_id,
        )
    )
    policy_decision_node = AgentPolicyDecisionNode.from_action_decision(
        policy_decision, agent_id=agent_id
    )

    if not policy_decision.allowed:
        status = "blocked" if policy_decision.decision == DECISION_QUEUE else "failed"
        result = (
            f"policy {policy_decision.decision} ({policy_decision.tier}): "
            f"{policy_decision.reason}"
        )
        if not _fence_still_valid(engine, task_id, claim, token=token):
            logger.warning(
                "Agent task %s: fencing token stale (lease reclaimed by a "
                "newer holder) — policy-decision commit rejected.",
                task_id,
            )
            return "fenced"
        _write_agent_task_provenance(
            engine,
            task_id=task_id,
            claim=claim,
            agent_id=agent_id,
            status=status,
            result=result,
            evidence=evidence,
            policy_decision_node=policy_decision_node,
            grant_id=None,
        )
        _finalize_agent_task(
            engine,
            task_id,
            claim,
            status=status,
            reward=0.0,
            feedback_text=result[:2000],
        )
        return "blocked" if status == "blocked" else "denied"

    # Capability grant — resolve an existing grant, or self-issue a bootstrap
    # one so there is always SOME AUTHORIZED_FOR audit trail for the
    # execution (advisory today: action_policy above is the hard gate; this
    # is the per-grant record team-synthesis already reads).
    grant_id: str | None = None
    if agent_id:
        existing = resolve_capability_grant(engine, agent_id, capability, now=now)
        grant_id = existing.get("id") if existing else None
        if grant_id is None:
            grant_id = grant_capability(
                engine,
                agent_id,
                capability,
                issuer="agent-dispatch",
                ttl_seconds=claim_ttl_s,
                now=now,
            )

    # Execute — pluggable body; the default FAILS CLOSED (no fabricated
    # success — AU-P0-3). Concrete task kinds plug their own executor in,
    # same shape as _execute_goal_turn/_execute_orchestrator_turn.
    try:
        result = (executor or _default_agent_task_executor)(claim)
        status = "completed"
        reward = 1.0
    except NoExecutorBoundError as e:
        # No executor bound ⇒ nothing ran. Never "completed"/reward=1.0.
        result = str(e)
        status = "unroutable"
        reward = 0.0
    except Exception as e:  # noqa: BLE001 — durably record, never raise
        result = str(e)
        status = "failed"
        reward = 0.0

    if not _fence_still_valid(engine, task_id, claim, token=token):
        logger.warning(
            "Agent task %s: fencing token stale (lease reclaimed by a newer "
            "holder) — execution commit rejected (result discarded: %s).",
            task_id,
            status,
        )
        return "fenced"

    _write_agent_task_provenance(
        engine,
        task_id=task_id,
        claim=claim,
        agent_id=agent_id,
        status=status,
        result=result,
        evidence=evidence,
        policy_decision_node=policy_decision_node,
        grant_id=grant_id,
    )
    _finalize_agent_task(
        engine,
        task_id,
        claim,
        status=status,
        reward=reward,
        feedback_text=str(result)[:2000],
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
    engine: Any, envelope: AgentTurnEnvelope, claim: dict[str, Any]
) -> str:
    """Run the claimed orchestrator job via the existing agent execution path.

    The agent invocation is wrapped in a durable action keyed by ``job_id``
    (CONCEPT:AU-OS.state.unified-durable-state-externalization): the queue gives at-least-once delivery, so a redelivery
    of the same turn returns the recorded result instead of re-running the
    agent (exactly-once effect), complementing the stale-claim guard above.
    """
    import asyncio

    from agent_utilities.orchestration.durable_execution import (
        DurableExecutionManager,
    )
    from agent_utilities.orchestration.manager import Orchestrator

    orch = Orchestrator(engine)
    durable = DurableExecutionManager(session_id=envelope.session_id)

    async def _invoke() -> Any:
        return await orch.execute_agent(
            agent_name=envelope.agent_name,
            task=claim["description"],
            session_id=envelope.session_id,
        )

    try:
        output = asyncio.run(
            durable.arun_durable_action(
                node_id=f"orchestrator_task:{envelope.job_id}",
                action=_invoke,
                idempotency_key=envelope.job_id,
            )
        )
    except Exception as e:  # noqa: BLE001 — durably mark failed, then ack
        engine._update_task_status(
            envelope.payload_ref,
            "failed",
            {
                "error": str(e),
                "executed_by": worker_token(),
                "correlation_id": _turn_correlation_id(),
            },
        )
        return "failed"
    engine._update_task_status(
        envelope.payload_ref,
        "completed",
        {
            "result": str(output)[:4000],
            "executed_by": worker_token(),
            "correlation_id": _turn_correlation_id(),
        },
    )
    return "completed"


def _fail_expired(envelope: AgentTurnEnvelope, engine: Any) -> None:
    """Durably mark a past-deadline turn failed without executing it."""
    reason = f"Dispatch deadline {envelope.deadline_unix} expired before execution."
    if envelope.kind == KIND_GOAL_LOOP:
        from agent_utilities.core import sessions as _sessions

        gid = envelope.payload_ref
        goal_engine = _sessions._goal_engine()
        if goal_engine is None:
            logger.error("No KG engine — cannot expire goal %s.", gid)
            return
        try:
            from agent_utilities.knowledge_graph.research.loops import TERMINAL_STATUS

            entry = _sessions._load_goal_entry(goal_engine, gid)
            if entry and str(entry.get("status") or "") not in TERMINAL_STATUS:
                goal_engine.add_node(
                    gid,
                    "Concept",
                    properties={
                        "status": "failed",
                        "error": reason,
                        "updated_at": time.time(),
                    },
                )
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to expire goal %s: %s", gid, e)
    elif engine is not None:
        try:
            engine._update_task_status(
                envelope.payload_ref, "failed", {"error": reason}
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to expire task %s: %s", envelope.payload_ref, e)


def execute_agent_turn(
    envelope: AgentTurnEnvelope,
    engine: Any = None,
    *,
    token: str | None = None,
    now: float | None = None,
    claim_ttl_s: float = CLAIM_TTL_S,
) -> str:
    """Claim + execute + write back ONE dispatched turn; return the outcome.

    Outcomes: ``completed`` | ``failed`` | ``skipped`` (duplicate delivery /
    live claim elsewhere) | ``expired`` (deadline passed). The whole cycle
    holds the per-session guard — one executor per session, fleet-wide.
    """
    token = token or worker_token()
    with session_execution_guard(envelope.session_id):
        if envelope.deadline_unix and (now or time.time()) > envelope.deadline_unix:
            _fail_expired(envelope, engine)
            return "expired"
        if envelope.kind == KIND_GOAL_LOOP:
            spec = claim_goal_run(
                envelope.payload_ref, token=token, now=now, claim_ttl_s=claim_ttl_s
            )
            if spec is None:
                return "skipped"
            return _execute_goal_turn(spec)
        if envelope.kind == KIND_ORCHESTRATOR_TASK:
            if engine is None:
                raise RuntimeError(
                    "orchestrator_task dispatch requires an engine client"
                )
            claim = claim_orchestrator_task(
                engine,
                envelope.payload_ref,
                token=token,
                now=now,
                claim_ttl_s=claim_ttl_s,
            )
            if claim is None:
                return "skipped"
            return _execute_orchestrator_turn(engine, envelope, claim)
        logger.error(
            "Unknown dispatch kind %r (job %s).", envelope.kind, envelope.job_id
        )
        return "failed"


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
            logger.warning("agent-dispatch poll error: %s", e)
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
            logger.error("agent-dispatch worker error: %s", e)
        finally:
            active.clear()
        _record_turn_outcome(outcome)
        try:
            queue.ack(item_id)
        except Exception as e:  # noqa: BLE001 — redelivery is safe (idempotent)
            logger.warning("agent-dispatch ack failed (%s); redelivery is safe.", e)


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

    With the Kafka transport each thread should own its own consumer-backed
    queue (confluent consumers are not thread-safe); the SQLite/Postgres
    backends are internally locked, so one shared queue object is fine.
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

    engine = IntelligenceGraphEngine()

    # Verify the client/auth path (CONCEPT:AU-OS.identity.authenticated-identity-enforcement) BEFORE consuming: a worker
    # that cannot reach the engine must fail loud, not claim turns and drop them.
    try:
        engine.query_cypher("MATCH (t:Task) RETURN count(t) AS c")
    except Exception as e:  # noqa: BLE001
        parser.exit(
            2,
            "Cannot reach the epistemic-graph engine as a client: "
            f"{e}\nCheck GRAPH_SERVICE_ENDPOINTS / GRAPH_SERVICE_TCP_ADDR / "
            "GRAPH_SERVICE_SOCKET and the shared HMAC secret "
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
