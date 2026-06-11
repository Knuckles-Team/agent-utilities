#!/usr/bin/python
from __future__ import annotations

"""Decoupled KG ingest worker — the ``kg-ingest`` Kafka consumer group.

CONCEPT:KG-2.57 — Decoupled kg-ingest consumer group with idempotent at-least-once task claims
and lag-visible backpressure: ingest workers no longer
have to live as daemon threads inside the host engine process. With the Kafka
task queue selected (``TASK_QUEUE_BACKEND=kafka``, CONCEPT:KG-2.55) any number
of worker processes — on any host — join the ``kg-ingest`` consumer group,
consume keyed task messages from ``kg_tasks`` (CONCEPT:KG-2.56), and process
them with the SAME worker body the in-process workers use
(:meth:`TaskManagerMixin._execute_claimed_task` — extracted, not duplicated).

Workers are engine **clients**: they talk to the single Rust epistemic-graph
daemon over UDS/TCP with the OS-5.14 HMAC secret (``GRAPH_SERVICE_AUTH_SECRET``
or the shared ``data_dir()/engine_secret``) and never take the KG host flock —
``KG_DAEMON_ROLE=client`` is forced so a worker process spawns no daemon
threads of its own.

Delivery semantics (documented contract):

* **At-least-once.** Offsets are committed only AFTER a task finishes (or is
  durably marked failed); a worker crash redelivers the message to another
  group member.
* **Idempotent claims.** The task's ``job_id`` is its idempotency key: the
  claim MERGEs the ``:Task`` node and skips any job already
  ``running``/``completed``/``failed``/``cancelled`` (guarded cross-host by the
  KG-2.54 ``state_claim_guard`` advisory lock when ``STATE_DB_URI`` is set).
  Graph writes themselves are MERGE-based, so a rare duplicate execution
  converges instead of corrupting.
* **Per-key ordering.** Kafka orders within a partition; the KG-2.56 key
  hierarchy (tenant → repo/corpus → task type) therefore gives per-tenant /
  per-repo ordering without global serialization. There is no cross-partition
  priority lane (unlike the graph-polling mode's high-priority poll).

Run::

    python -m agent_utilities.knowledge_graph.ingest_worker [--workers N]
    # or the console script:
    kg-ingest-worker --bootstrap-servers kafka:9092

Per-host concurrency autosizes with the same CPU/memory sizer the in-process
pool uses (:func:`compute_ingest_worker_count`); each worker thread owns its
own consumer, so partitions spread across threads AND processes uniformly.
"""

import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .core.engine_tasks import (
    _decode_metadata,
    _encode_metadata,
    compute_ingest_worker_count,
)
from .core.kafka_queue_backend import INGEST_GROUP, TASKS_TOPIC

logger = logging.getLogger(__name__)

#: Long-running parse/LLM tasks must not trip a group rebalance mid-task.
_MAX_POLL_INTERVAL_MS = 3_600_000  # 1h — above the reaper's runtime cap default
_TERMINAL_OR_ACTIVE = {"running", "completed", "failed", "cancelled"}


def claim_task_envelope(
    engine: Any, envelope: dict[str, Any]
) -> tuple[str, Path, bool, str] | None:
    """Idempotently claim ONE consumed task envelope (CONCEPT:KG-2.57).

    MERGEs the ``:Task`` node as ``running`` with this process's ownership
    stamp (the same ``claimed_by``/``claim_unix`` contract the zombie reaper
    audits) and returns ``(job_id, target, is_codebase, task_type)`` — or
    ``None`` when the message is a duplicate delivery of an already
    claimed/finished job (at-least-once dedup) or malformed.
    """
    from agent_utilities.core.state_store import state_claim_guard

    job_id = envelope.get("job_id")
    if not job_id:
        logger.warning("Ingest message without job_id skipped: %.120s", envelope)
        return None
    props = dict(envelope.get("props") or {})
    meta = _decode_metadata(props.get("metadata"))
    target = meta.get("target")
    task_type = meta.get("type", "document")

    # Cross-host claim atomicity (CONCEPT:KG-2.54): partition assignment already
    # routes a message to exactly one group member; the advisory-lock guard
    # additionally serializes the status-check/claim against redeliveries.
    with state_claim_guard("kg-task-claim"):
        rows = engine.query_cypher(
            "MATCH (t:Task {id: $id}) RETURN t.status as s", {"id": job_id}
        )
        status = rows[0].get("s") if rows else None
        if status in _TERMINAL_OR_ACTIVE:
            logger.debug(
                "Duplicate delivery of %s (status=%s) skipped.", job_id, status
            )
            return None
        meta["started_at"] = datetime.now(UTC).isoformat()
        meta["claimed_by"] = engine._get_host_token()
        meta["claim_unix"] = time.time()
        props["status"] = "running"
        props["metadata"] = _encode_metadata(meta)
        engine.add_node(job_id, "Task", properties=props)

    if not target:
        logger.error("Task %s has no target in metadata, failing.", job_id)
        engine._update_task_status(
            job_id,
            "failed",
            {"error": "Missing target in task metadata", "type": "unknown"},
        )
        return None
    return job_id, Path(target), task_type == "codebase", task_type


def run_ingest_consumer_loop(
    engine: Any, consumer: Any, stop_event: threading.Event
) -> None:
    """Consume ``kg_tasks`` until ``stop_event``: claim → process → commit.

    Commit happens strictly AFTER the task is processed (or durably marked
    failed) — at-least-once. One poisonous message never kills the loop.
    """
    while not stop_event.is_set():
        try:
            msg = consumer.poll(1.0)
        except Exception as e:  # noqa: BLE001 — broker hiccup: back off, retry
            logger.warning("kg-ingest poll error: %s", e)
            time.sleep(2.0)
            continue
        if msg is None:
            continue
        if getattr(msg, "error", lambda: None)():
            logger.debug("kg-ingest message error: %s", msg.error())
            continue

        job_id = None
        try:
            envelope = json.loads(msg.value().decode("utf-8"))
            claimed = claim_task_envelope(engine, envelope)
            if claimed is not None:
                job_id, target, is_codebase, task_type = claimed
                engine._execute_claimed_task(job_id, target, is_codebase, task_type)
        except Exception as e:  # noqa: BLE001 — mark failed, keep consuming
            logger.error("kg-ingest worker error: %s", e)
            if job_id:
                try:
                    engine._update_task_status(job_id, "failed", {"error": str(e)})
                except Exception as inner:  # noqa: BLE001
                    logger.error("Failed to mark %s failed: %s", job_id, inner)
        try:
            consumer.commit(message=msg, asynchronous=False)
        except Exception as e:  # noqa: BLE001 — redelivery is safe (idempotent)
            logger.warning("kg-ingest offset commit failed (%s); redelivery is safe.", e)


def _default_consumer_factory(bootstrap_servers: str) -> Any:
    from confluent_kafka import Consumer

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": INGEST_GROUP,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "max.poll.interval.ms": _MAX_POLL_INTERVAL_MS,
        }
    )
    consumer.subscribe([TASKS_TOPIC])
    return consumer


def _resolve_bootstrap_servers(engine: Any, override: str | None = None) -> str:
    if override:
        return override
    q = getattr(engine, "_submission_queue", None)
    servers = getattr(q, "bootstrap_servers", None)
    if servers:
        return str(servers)
    from agent_utilities.core.config import config

    return getattr(config, "kafka_bootstrap_servers", None) or "localhost:9092"


def start_ingest_consumer_pool(
    engine: Any,
    *,
    worker_count: int | None = None,
    bootstrap_servers: str | None = None,
    stop_event: threading.Event | None = None,
    consumer_factory: Any = None,
) -> list[threading.Thread]:
    """Start ``worker_count`` ``kg-ingest`` consumer threads (CONCEPT:KG-2.57).

    Used by BOTH the host engine in Kafka mode (its in-process pool becomes
    ordinary group members) and the standalone ``kg-ingest-worker`` process —
    the group spans them, and Kafka splits partitions across every member.
    Each thread owns its own consumer (confluent consumers are not
    thread-safe). ``consumer_factory`` is the test seam.
    """
    count = worker_count or compute_ingest_worker_count()
    stop = stop_event or threading.Event()
    servers = _resolve_bootstrap_servers(engine, bootstrap_servers)
    make_consumer = consumer_factory or (lambda: _default_consumer_factory(servers))

    threads: list[threading.Thread] = []
    for i in range(count):

        def _runner() -> None:
            consumer = make_consumer()
            try:
                run_ingest_consumer_loop(engine, consumer, stop)
            finally:
                close = getattr(consumer, "close", None)
                if callable(close):
                    close()

        t = threading.Thread(target=_runner, name=f"KGIngestConsumer-{i}", daemon=True)
        t.start()
        threads.append(t)
    logger.info(
        "kg-ingest consumer pool started: %d workers, group=%s, brokers=%s",
        count,
        INGEST_GROUP,
        servers,
    )
    return threads


def main(argv: list[str] | None = None) -> int:
    """Entry point: a standalone, host-role-free ingest worker process."""
    import argparse
    import os
    import signal

    parser = argparse.ArgumentParser(
        prog="kg-ingest-worker",
        description=(
            "Decoupled KG ingest worker (CONCEPT:KG-2.57): joins the "
            f"'{INGEST_GROUP}' Kafka consumer group and processes kg_tasks as "
            "an engine client — no KG host role required."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Consumer threads (default: CPU/memory autosized).",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default=None,
        help="Kafka brokers (default: KAFKA_BOOTSTRAP_SERVERS).",
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

    # Verify the client/auth path (CONCEPT:OS-5.14) BEFORE joining the group:
    # a worker that cannot reach the engine must fail loud, not consume+drop.
    try:
        engine.query_cypher("MATCH (t:Task) RETURN count(t) AS c")
    except Exception as e:  # noqa: BLE001
        parser.exit(
            2,
            "Cannot reach the epistemic-graph engine as a client: "
            f"{e}\nCheck GRAPH_SERVICE_TCP_ADDR / GRAPH_SERVICE_SOCKET and the "
            "shared HMAC secret (GRAPH_SERVICE_AUTH_SECRET or the host's "
            "data_dir()/engine_secret — CONCEPT:OS-5.14).\n",
        )

    stop = threading.Event()

    def _shutdown(signum: int, _frame: Any) -> None:
        logger.info("Signal %s received — draining and stopping workers.", signum)
        stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    threads = start_ingest_consumer_pool(
        engine,
        worker_count=args.workers,
        bootstrap_servers=args.bootstrap_servers,
        stop_event=stop,
    )
    while any(t.is_alive() for t in threads) and not stop.is_set():
        time.sleep(1.0)
    for t in threads:
        t.join(timeout=10.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
