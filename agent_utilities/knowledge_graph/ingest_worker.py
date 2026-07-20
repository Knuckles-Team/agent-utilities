#!/usr/bin/python
from __future__ import annotations

"""Decoupled KG ingest worker — the ``kg-ingest`` Kafka consumer group.

CONCEPT:AU-KG.ingest.decoupled-kg-ingest-consumer — Decoupled kg-ingest consumer group with idempotent at-least-once task claims
and lag-visible backpressure: ingest workers no longer
have to live as daemon threads inside the host engine process. With the Kafka
task queue selected (``TASK_QUEUE_BACKEND=kafka``, CONCEPT:AU-KG.backend.selectable-queue-backend) any number
of worker processes — on any host — join the ``kg-ingest`` consumer group,
consume keyed task messages from ``kg_tasks`` (CONCEPT:AU-KG.backend.keyed-ingest-partitions), and process
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
* **Idempotent claims.** Kafka is notification transport only. Every consumer
  must atomically claim the deterministic WorkItem; a negative claim is final
  and never consults a second status or lock authority.
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
from pathlib import Path
from typing import Any

from .core.engine_tasks import (
    _authorized_background_thread,
    _capture_verified_background_session,
    _resolve_task_target,
    compute_ingest_worker_count,
)
from .core.kafka_queue_backend import INGEST_GROUP, TASKS_TOPIC

logger = logging.getLogger(__name__)

#: Long-running parse/LLM tasks must not trip a group rebalance mid-task.
_MAX_POLL_INTERVAL_MS = 3_600_000  # 1h — below the native WorkItem lease cap


def claim_task_envelope(
    engine: Any, envelope: dict[str, Any]
) -> tuple[str, Path, bool, str] | None:
    """Idempotently claim ONE consumed task envelope (CONCEPT:AU-KG.ingest.decoupled-kg-ingest-consumer).

    Claims the deterministic WorkItem and returns
    ``(job_id, target, is_codebase, task_type)``. A
    duplicate delivery receives an authoritative negative claim and is skipped.
    """
    from agent_utilities.orchestration import work_item as _wi

    job_id = envelope.get("job_id")
    if not job_id:
        logger.warning("Ingest message without job_id skipped: %.120s", envelope)
        return None
    token = engine._get_host_token()
    item_id = _wi.ingest_task_work_item_id(job_id)
    claim = _wi.claim_specific(
        engine._work_item_engine,
        item_id,
        token=token,
        lease_ttl_s=_MAX_POLL_INTERVAL_MS / 1000.0,
    )
    if claim is None:
        logger.debug("Duplicate delivery of %s skipped by ClaimWorkItem.", job_id)
        return None
    if not _wi.mark_running(engine._work_item_engine, item_id, claim):
        return None
    meta = engine._ingest_task_metadata(job_id)
    target = meta.get("target")
    task_type = meta.get("type", "document")
    engine._remember_work_item_claim(job_id, claim)

    if not target:
        logger.error("WorkItem %s has no target in metadata, failing.", item_id)
        engine._update_task_status(
            job_id,
            "failed",
            {"error": "Missing target in task metadata", "type": "unknown"},
        )
        return None
    return job_id, _resolve_task_target(str(target)), task_type == "codebase", task_type


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
            logger.warning(
                "kg-ingest offset commit failed (%s); redelivery is safe.", e
            )


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
    background_session: Any = None,
) -> list[threading.Thread]:
    """Start ``worker_count`` ``kg-ingest`` consumer threads (CONCEPT:AU-KG.ingest.decoupled-kg-ingest-consumer).

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
    worker_session = background_session or _capture_verified_background_session()
    worker_session.engine_verified_context()

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

        t = _authorized_background_thread(
            worker_session,
            _runner,
            name=f"KGIngestConsumer-{i}",
        )
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
            "Decoupled KG ingest worker (CONCEPT:AU-KG.ingest.decoupled-kg-ingest-consumer): joins the "
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

    # Engine-client posture (CONCEPT:EG-KG.storage.nonblocking-checkpoint/OS-5.9): never contend for the host
    # flock, never spawn the consolidated daemon — this process only consumes.
    os.environ.setdefault("KG_DAEMON_ROLE", "client")

    from agent_utilities.core.config import config
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
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
        engine = IntelligenceGraphEngine.get_or_create()

        # Verify the client/auth path (CONCEPT:AU-OS.identity.authenticated-identity-enforcement) BEFORE joining the group:
        # a worker that cannot reach the engine must fail loud, not consume+drop.
        try:
            native = engine._work_item_engine._native_work_items()
            if not callable(getattr(native, "claim_work_item", None)):
                raise RuntimeError("engine does not expose native ClaimWorkItem")
            engine._work_item_engine.query_cypher(
                "MATCH (w:WorkItem) RETURN count(w) AS c"
            )
        except Exception as e:  # noqa: BLE001
            parser.exit(
                2,
                "Cannot reach the epistemic-graph engine as a client: "
                f"{e}\nCheck GRAPH_SERVICE_ENDPOINTS (external) or the packaged-local "
                "transport and "
                "shared HMAC secret (GRAPH_SERVICE_AUTH_SECRET or the host's "
                "data_dir()/engine_secret — CONCEPT:AU-OS.identity.authenticated-identity-enforcement).\n",
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
            background_session=session,
        )
        while any(t.is_alive() for t in threads) and not stop.is_set():
            time.sleep(1.0)
        for t in threads:
            t.join(timeout=10.0)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
