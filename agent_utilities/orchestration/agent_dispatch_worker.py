#!/usr/bin/python
from __future__ import annotations

"""Stateless agent dispatch worker — the ``agent-dispatch`` consumer fleet.

CONCEPT:ORCH-1.45 — Queue-driven agent dispatch with session-keyed partitions
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
  ``KG_DAEMON_ROLE=client`` (CONCEPT:OS-5.14 auth applies) and never contend
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


def worker_token() -> str:
    """Stable identity for claims/heartbeats: ``hostname:pid:agent-dispatch``."""
    return f"{socket.gethostname()}:{os.getpid()}:agent-dispatch"


def _turn_correlation_id() -> str:
    """Correlation id stamped on the executed Task node (CONCEPT:OS-5.11).

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

    Reads the goal's KG Loop node (CONCEPT:KG-2.78) plus the ``goal_spec``
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
    (CONCEPT:OS-5.16): the queue gives at-least-once delivery, so a redelivery
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
            "Stateless agent dispatch worker (CONCEPT:ORCH-1.45): consumes "
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

    # Engine-client posture (CONCEPT:KG-2.8/OS-5.9): never contend for the host
    # flock, never spawn the consolidated daemon — this process only consumes.
    os.environ.setdefault("KG_DAEMON_ROLE", "client")

    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine()

    # Verify the client/auth path (CONCEPT:OS-5.14) BEFORE consuming: a worker
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
            "— CONCEPT:OS-5.14).\n",
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
