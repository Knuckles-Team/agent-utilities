#!/usr/bin/python
from __future__ import annotations

"""Queue-driven agent dispatch — the enqueue side.

CONCEPT:ORCH-1.45 — Queue-driven agent dispatch with session-keyed partitions
consumed by a stateless dispatch-worker fleet

The in-process asyncio scheduler (``core/cognitive_scheduler.py``) caps agent
concurrency at one process on one host: sessions are pinned to the host that
created them, and a busy gateway cannot hand work to an idle peer. This module
externalizes the *dispatch* of an agent turn (a goal-loop run or an
orchestrator job) onto the SAME durable task-queue stack the KG ingest plane
already scales on (CONCEPT:KG-2.55/2.56/2.57):

* the queue carries a small typed :class:`AgentTurnEnvelope` — REFERENCES only
  (``payload_ref`` points at the goal/Task record in the shared state store or
  graph); large bodies never ride the queue;
* the partition key is the **session id** (see
  :func:`~agent_utilities.knowledge_graph.core.kafka_queue_backend.partition_key_for`)
  so all turns of one session land on one partition and execute serially —
  per-session ordering is a turn-coherence REQUIREMENT, stronger than the
  ingest plane's per-tenant ordering;
* any host running an ``agent-dispatch-worker``
  (:mod:`agent_utilities.orchestration.agent_dispatch_worker`) claims turns,
  rehydrates session state from the OS-5.16 shared state store, executes
  through the EXISTING goal/agent execution paths, and writes results back
  durably — the scheduler tier becomes horizontally scalable and sessions are
  no longer pinned to their birth host.

Placement is deliberately **queue-pull** (workers claim work when they have
capacity) rather than a central placer pushing turns at workers: with
session-keyed partitions the broker/queue already provides per-session
serialization and uniform load spreading, so a placer would add a coordination
point and a failure mode without adding correctness. Affinity-aware placement
(HRW on warm caches) is future work layered on the same envelope.

Selection is one AgentConfig flag — ``agent_dispatch_backend``
(``AGENT_DISPATCH_BACKEND``): ``inline`` (default — the existing in-process
behavior, byte-for-byte) or ``queue`` (dispatch returns a job handle and a
worker fleet executes). The queue *transport* reuses the KG-2.55 resolution
(``TASK_QUEUE_BACKEND``/auto): Kafka topic ``agent_turns``, Postgres SKIP
LOCKED table ``agent_dispatch_queue``, or the zero-infra per-host SQLite file.
"""

import logging
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

#: Kafka topic / consumer group for dispatched agent turns. Postgres and SQLite
#: transports use an equivalently-named table / db file instead of a topic.
AGENT_TURNS_TOPIC = "agent_turns"
DISPATCH_GROUP = "agent-dispatch"

#: Accepted ``AGENT_DISPATCH_BACKEND`` values.
AGENT_DISPATCH_BACKENDS = ("inline", "queue")

#: Envelope kinds the dispatch workers know how to execute.
KIND_GOAL_LOOP = "goal_loop"
KIND_ORCHESTRATOR_TASK = "orchestrator_task"


class AgentTurnEnvelope(BaseModel):
    """One dispatched agent turn on the ``agent_turns`` queue.

    CONCEPT:ORCH-1.45 — the queue carries references, not bodies: the durable
    record (the ``goals``/``sessions`` rows for a goal run, the ``:Task`` graph
    node for an orchestrator job) is the payload's source of truth, addressed
    by ``payload_ref``. ``job_id`` doubles as the idempotency key — redelivery
    of an already-claimed/finished job is skipped by the worker's claim check
    (at-least-once delivery, idempotent claims).
    """

    job_id: str = Field(default_factory=lambda: f"dispatch-{uuid.uuid4().hex[:8]}")
    session_id: str
    kind: str = KIND_GOAL_LOOP
    payload_ref: str = ""
    agent_name: str = ""
    tenant: str = ""
    priority: str = "normal"
    deadline_unix: float | None = None
    attempt: int = 0
    enqueued_at: float = Field(default_factory=time.time)

    def to_item(self) -> dict[str, Any]:
        """Serialize for the queue. ``session_id`` stays top-level so
        ``partition_key_for`` keys the message without decoding metadata."""
        return self.model_dump()

    @classmethod
    def from_item(cls, item: dict[str, Any]) -> AgentTurnEnvelope:
        return cls.model_validate(item)


def resolve_dispatch_backend(config: Any = None) -> str:
    """Resolve ``agent_dispatch_backend`` to ``inline`` or ``queue``."""
    if config is None:
        from agent_utilities.core.config import config as _cfg

        config = _cfg
    raw = str(getattr(config, "agent_dispatch_backend", "inline") or "inline")
    choice = raw.strip().lower()
    if choice not in AGENT_DISPATCH_BACKENDS:
        raise ValueError(
            f"AGENT_DISPATCH_BACKEND={choice!r} is not one of {AGENT_DISPATCH_BACKENDS}"
        )
    return choice


def dispatch_queue_enabled(config: Any = None) -> bool:
    """True when agent dispatch is queue-backed (CONCEPT:ORCH-1.45)."""
    return resolve_dispatch_backend(config) == "queue"


# ── queue construction ─────────────────────────────────────────────────────

_queue_lock = threading.Lock()
_queue: Any = None


def create_dispatch_queue(config: Any = None) -> Any:
    """Build the ``agent_turns`` queue on the KG-2.55-selected transport.

    Composes the existing task-queue stack rather than introducing a second
    queue technology: the SAME ``TASK_QUEUE_BACKEND``/auto resolution picks
    kafka (keyed ``agent_turns`` topic), postgres (SKIP LOCKED claims on the
    ``agent_dispatch_queue`` table of the shared state store), or the per-host
    SQLite file. Fail-loud semantics follow the ingest queue: an explicitly
    selected kafka/postgres transport that is unreachable raises
    :class:`~agent_utilities.knowledge_graph.core.queue_backend.TaskQueueUnavailable`.
    """
    from agent_utilities.knowledge_graph.core.queue_backend import (
        resolve_task_queue_backend,
    )

    if config is None:
        from agent_utilities.core.config import config as _cfg

        config = _cfg

    choice, explicit = resolve_task_queue_backend(config)
    fallback_db_path = _sqlite_queue_path()

    if choice == "kafka":
        from agent_utilities.knowledge_graph.core.kafka_queue_backend import (
            KafkaQueueBackend,
        )

        return KafkaQueueBackend(
            fallback_db_path=None if explicit else fallback_db_path,
            bootstrap_servers=getattr(config, "kafka_bootstrap_servers", None),
            fail_loud=explicit,
            partitions=int(getattr(config, "agent_turns_partitions", 6) or 6),
            tasks_topic=AGENT_TURNS_TOPIC,
            consumer_group=DISPATCH_GROUP,
        )

    if choice == "postgres":
        from agent_utilities.knowledge_graph.core.queue_backend import (
            TaskQueueUnavailable,
        )

        try:
            from agent_utilities.knowledge_graph.core.postgres_queue_backend import (
                PostgresTaskQueue,
            )

            return PostgresTaskQueue(queue_table="agent_dispatch_queue")
        except Exception as e:  # noqa: BLE001 — explicit ⇒ fail loud, auto ⇒ degrade
            if explicit:
                raise TaskQueueUnavailable(
                    "TASK_QUEUE_BACKEND=postgres is explicitly selected but the "
                    "agent dispatch queue cannot reach the state-store Postgres "
                    f"({getattr(config, 'state_db_uri', None)!r}): {e}. Fix "
                    "STATE_DB_URI / the database, or unset TASK_QUEUE_BACKEND."
                ) from e
            logger.warning(
                "STATE_DB_URI set but the Postgres dispatch queue is unavailable "
                "(%s) — falling back to the per-host SQLite dispatch queue.",
                e,
            )

    from agent_utilities.knowledge_graph.core.engine_tasks import SQLiteTaskQueue

    return SQLiteTaskQueue(fallback_db_path)


def _sqlite_queue_path() -> str:
    from agent_utilities.core.paths import data_dir

    return str(data_dir() / "agent_dispatch_queue.db")


def get_dispatch_queue(config: Any = None) -> Any:
    """Process-wide cached dispatch queue (lazily constructed)."""
    global _queue
    with _queue_lock:
        if _queue is None:
            _queue = create_dispatch_queue(config)
        return _queue


def reset_dispatch_queue_for_tests(queue: Any = None) -> None:
    """Swap/clear the cached dispatch queue (test isolation seam)."""
    global _queue
    with _queue_lock:
        _queue = queue


def dispatch_queue_depth(queue: Any = None) -> int:
    """Unclaimed ``agent_turns`` depth (Kafka = consumer-group lag)."""
    q = queue if queue is not None else get_dispatch_queue()
    try:
        return int(q.get_queue_size())
    except Exception:  # noqa: BLE001 — depth probe is best-effort
        return 0


# ── enqueue ────────────────────────────────────────────────────────────────


def enqueue_agent_turn(
    envelope: AgentTurnEnvelope, queue: Any = None
) -> dict[str, Any]:
    """Publish one agent turn and return its job handle.

    The handle is what queue-mode dispatch returns to the caller instead of
    executing in-process: poll the existing ``graph_orchestrate action=status``
    / ``/api/graph/orchestrate/job/{job_id}`` surface (orchestrator jobs) or
    the goals API (goal runs) for progress and the executing worker/host.
    """
    q = queue if queue is not None else get_dispatch_queue()
    q.put(envelope.to_item())
    logger.info(
        "Agent turn enqueued: job=%s session=%s kind=%s",
        envelope.job_id,
        envelope.session_id,
        envelope.kind,
    )
    return {
        "job_id": envelope.job_id,
        "session_id": envelope.session_id,
        "kind": envelope.kind,
        "dispatch": "queued",
        "status": "pending",
    }


# ── per-session mutual exclusion ───────────────────────────────────────────

_session_locks_lock = threading.Lock()
_session_locks: dict[str, threading.Lock] = {}


def _session_lock(session_id: str) -> threading.Lock:
    with _session_locks_lock:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _session_locks[session_id] = lock
        return lock


@contextmanager
def session_execution_guard(session_id: str) -> Iterator[None]:
    """One executing worker per session at a time (CONCEPT:ORCH-1.45).

    At-least-once delivery means two workers can briefly hold the same
    session's turns (e.g. a redelivery racing the original consumer). Turn
    coherence requires per-session mutual exclusion, layered:

    * a process-local per-session lock serializes worker threads in ONE
      process (covers the SQLite/single-host transports);
    * ``state_claim_guard`` extends the critical section fleet-wide via a
      Postgres advisory lock when durable state is externalized
      (CONCEPT:OS-5.16) — two hosts can never execute one session at once.

    A crashed holder releases both automatically (process death drops the
    advisory lock server-side), so crash recovery is redelivery + re-claim,
    never a stuck session.
    """
    from agent_utilities.core.state_store import state_claim_guard

    with _session_lock(session_id), state_claim_guard(f"agent-session:{session_id}"):
        yield


# ── fleet-visible worker registry ──────────────────────────────────────────

#: A worker whose last heartbeat is older than this is presumed gone and is
#: excluded from topology/metrics (its in-flight claim recovers separately via
#: the stale-claim re-claim path).
WORKER_HEARTBEAT_TTL_S = 90.0


def record_dispatch_worker_heartbeat(
    worker_id: str,
    *,
    host: str = "",
    capacity: int = 1,
    active_sessions: list[str] | tuple[str, ...] = (),
    queue_backend: str = "",
) -> None:
    """Upsert this worker's liveness row in the fleet registry.

    CONCEPT:ORCH-1.45 — the registry lives in the SAME sessions store the
    OS-5.18 supervisory plane already reads (per-host SQLite, or the shared
    Postgres under ``STATE_DB_URI`` — where every gateway sees every host's
    workers). ``/api/fleet/topology`` surfaces these rows.
    """
    import json as _json
    import socket as _socket

    from agent_utilities.core import sessions as _sessions

    now = time.time()
    conn = _sessions._connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO dispatch_workers
                (worker_id, host, capacity, active_sessions, queue_backend,
                 started_at, last_heartbeat)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
                host = excluded.host,
                capacity = excluded.capacity,
                active_sessions = excluded.active_sessions,
                queue_backend = excluded.queue_backend,
                last_heartbeat = excluded.last_heartbeat
            """,
            (
                worker_id,
                host or _socket.gethostname(),
                int(capacity),
                _json.dumps(list(active_sessions)),
                queue_backend,
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_dispatch_workers(
    ttl_s: float = WORKER_HEARTBEAT_TTL_S,
) -> list[dict[str, Any]]:
    """Live dispatch workers (heartbeat within ``ttl_s``), newest first."""
    import json as _json

    from agent_utilities.core import sessions as _sessions

    cutoff = time.time() - ttl_s
    conn = _sessions._connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT worker_id, host, capacity, active_sessions, queue_backend, "
            "started_at, last_heartbeat FROM dispatch_workers "
            "WHERE last_heartbeat >= ? ORDER BY last_heartbeat DESC",
            (cutoff,),
        )
        workers: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            entry = dict(row)
            try:
                entry["active_sessions"] = _json.loads(
                    entry.get("active_sessions") or "[]"
                )
            except (TypeError, ValueError):
                entry["active_sessions"] = []
            workers.append(entry)
        return workers
    finally:
        conn.close()
