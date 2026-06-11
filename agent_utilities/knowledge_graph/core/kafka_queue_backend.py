# CONCEPT:ECO-4.05 - Pluggable Event Queue Backend
# CONCEPT:ORCH-1.10 - Reactive Event Sourcing

"""Production-grade Kafka ingest task queue.

CONCEPT:KG-2.55 — Fail-loud selectable ingest task-queue backend: when Kafka is
EXPLICITLY selected (``TASK_QUEUE_BACKEND=kafka``) an unreachable broker raises
:class:`~.queue_backend.TaskQueueUnavailable` at startup instead of silently
degrading to the per-host SQLite file (which would split the fleet's queue into
invisible islands). The legacy ``QUEUE_BACKEND=kafka`` alias keeps the old
graceful SQLite fallback.

CONCEPT:KG-2.56 — Keyed ingest partitions for per-tenant and per-repo ordering
without global serialization: every task is produced to the
``kg_tasks`` topic with a partition key so Kafka guarantees per-key ordering
without global serialization. Key hierarchy (first match wins):

1. ``tenant:<id>`` — the ambient :class:`ActorContext` tenant (multi-tenant
   isolation ⇒ per-tenant ordering);
2. ``corpus:<repo>`` — the repo/corpus identifier of the ingest target
   (provenance ``full_path`` from the batch ingestor, else the path-derived
   repo root) ⇒ per-repo ordering for codebase ingest;
3. ``type:<task_type>`` — the task type, the coarsest bucket.

Topic provisioning is idempotent at startup: ``kg_tasks`` is created with
``KG_TASKS_PARTITIONS`` partitions (grow-only — an existing topic with more
partitions is never shrunk; with fewer, partitions are added).

The decoupled ``kg-ingest`` consumer group lives in
:mod:`agent_utilities.knowledge_graph.ingest_worker` (CONCEPT:KG-2.57); this
module owns the producer/topic/lag side. Uses ``confluent_kafka`` (a core
dependency), imported lazily so environments without it degrade per the
fail-loud/graceful contract above.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from .queue_backend import QueueBackend, TaskQueueUnavailable

logger = logging.getLogger(__name__)

#: Task submission topic (kept from the original backend) and its staging twin.
TASKS_TOPIC = "kg_tasks"
STAGING_TOPIC = "kg_staging"
#: Consumer group for decoupled ingest workers (CONCEPT:KG-2.57).
INGEST_GROUP = "kg-ingest"
STAGING_GROUP = "kg_staging_group"

_DEFAULT_BOOTSTRAP = "localhost:9092"
_PROBE_TIMEOUT_S = 5.0


def _corpus_root(target: str) -> str:
    """Derive a stable repo/corpus identifier from an ingest target path.

    Heuristic: the component after the DEEPEST known workspace container
    directory (``agent-packages`` / ``workspace`` / ``worktrees`` / ``repos``)
    — i.e. the repo checkout dir — else the first three path components. Per-
    file tasks fanned out from one repo thus share a key (per-repo ordering)
    without needing filesystem access on the producer host.
    """
    parts = [p for p in str(target).replace("\\", "/").split("/") if p]
    containers = {"agent-packages", "workspace", "worktrees", "repos"}
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] in containers:
            return "/".join(parts[: i + 2])
    return "/".join(parts[:3]) if parts else "unknown"


def partition_key_for(item: dict[str, Any]) -> str:
    """Compute the ``kg_tasks`` partition key for a task envelope.

    CONCEPT:KG-2.56 — key hierarchy: tenant id (ambient ActorContext) →
    repo/corpus identifier of the ingest target → task type. Guarantees
    per-tenant / per-repo ordering while letting unrelated work parallelize
    across partitions.
    """
    try:
        from agent_utilities.security.brain_context import current_actor

        tenant = current_actor().tenant_id
        if tenant:
            return f"tenant:{tenant}"
    except Exception:  # noqa: BLE001 — ambient identity is best-effort
        pass

    props = item.get("props") or {}
    # Batch-ingest provenance stamps the stable repo key directly (KG-2.49).
    full_path = props.get("full_path")
    if full_path:
        return f"corpus:{full_path}"

    meta: dict[str, Any] = {}
    raw_meta = props.get("metadata")
    if raw_meta:
        from .engine_tasks import _decode_metadata

        meta = _decode_metadata(raw_meta)
    target = meta.get("target")
    if target:
        return f"corpus:{_corpus_root(target)}"

    task_type = meta.get("type") or item.get("type") or "task"
    return f"type:{task_type}"


class KafkaQueueBackend(QueueBackend):
    """Kafka-backed durable task queue with keyed partitions.

    ``fail_loud=True`` (explicit ``TASK_QUEUE_BACKEND=kafka``) raises
    :class:`TaskQueueUnavailable` when the broker is unreachable at startup
    and on produce failure — never a silent SQLite degrade. With
    ``fail_loud=False`` and a ``fallback_db_path`` the legacy graceful SQLite
    fallback is preserved (deprecated ``QUEUE_BACKEND=kafka`` alias).

    Test seams: ``producer``/``admin_client``/``consumer_factory`` accept
    pre-built (fake) confluent-kafka-shaped clients so unit tests never need a
    live broker or the ``confluent_kafka`` import.
    """

    def __init__(
        self,
        fallback_db_path: str | None = None,
        bootstrap_servers: str | list[str] | None = None,
        *,
        fail_loud: bool = False,
        partitions: int = 6,
        producer: Any = None,
        admin_client: Any = None,
        consumer_factory: Any = None,
    ):
        if isinstance(bootstrap_servers, list | tuple):
            bootstrap_servers = ",".join(str(s) for s in bootstrap_servers)
        if not bootstrap_servers:
            from agent_utilities.core.config import config as _cfg

            bootstrap_servers = (
                getattr(_cfg, "kafka_bootstrap_servers", None) or _DEFAULT_BOOTSTRAP
            )
        self.bootstrap_servers: str = bootstrap_servers
        self.fail_loud = fail_loud
        self.partitions = max(1, int(partitions))
        self.fallback_db_path = fallback_db_path
        self._fallback_queue: Any = None
        self._producer: Any = producer
        self._admin: Any = admin_client
        self._consumer_factory = consumer_factory
        self._task_consumer: Any = None
        self._staging_consumer: Any = None
        self._lag_probe: Any = None
        self._lock = threading.Lock()

        try:
            if self._producer is None:
                from confluent_kafka import Producer

                self._producer = Producer(
                    {
                        "bootstrap.servers": self.bootstrap_servers,
                        "socket.timeout.ms": 5000,
                        "message.timeout.ms": 10000,
                    }
                )
            self.ensure_topics()
            logger.info(
                "Kafka task queue ready (brokers=%s, topic=%s, partitions>=%d, "
                "group=%s)",
                self.bootstrap_servers,
                TASKS_TOPIC,
                self.partitions,
                INGEST_GROUP,
            )
        except TaskQueueUnavailable:
            raise
        except Exception as e:
            self._handle_unavailable("connect/ensure-topic", e)

    # ── availability handling ──────────────────────────────────────────

    def _handle_unavailable(self, op: str, e: Exception) -> None:
        """Fail loud (explicit selection) or degrade to SQLite (legacy alias)."""
        if self.fail_loud:
            if isinstance(e, ImportError):
                remedy = "Install the client: `pip install confluent-kafka`"
            else:
                remedy = (
                    "Start the kg-backbone Kafka stack (e.g. `docker compose -f "
                    "docker/kafka-kraft.compose.yml up -d`) and check "
                    "KAFKA_BOOTSTRAP_SERVERS"
                )
            raise TaskQueueUnavailable(
                "TASK_QUEUE_BACKEND=kafka is explicitly selected but the broker "
                f"at {self.bootstrap_servers!r} is unavailable ({op}: {e}). "
                f"{remedy}, or fall back by unsetting TASK_QUEUE_BACKEND (auto) "
                "/ setting it to 'sqlite' or 'postgres'."
            ) from e
        logger.warning(
            "Kafka unavailable (%s: %s) — falling back to local SQLite "
            "(deprecated QUEUE_BACKEND alias semantics).",
            op,
            e,
        )
        self._use_fallback()

    def _use_fallback(self) -> Any:
        """Install (idempotently) and return the SQLite fallback queue."""
        if self._fallback_queue is not None:
            return self._fallback_queue
        if not self.fallback_db_path:
            raise TaskQueueUnavailable(
                f"Kafka at {self.bootstrap_servers!r} is unavailable and no "
                "SQLite fallback path is configured."
            )
        from .engine_tasks import SQLiteTaskQueue

        self._fallback_queue = SQLiteTaskQueue(self.fallback_db_path)
        logger.info(
            "Kafka queue fell back to SQLiteTaskQueue at %s", self.fallback_db_path
        )
        return self._fallback_queue

    # ── topic provisioning ── CONCEPT:KG-2.56

    def _admin_client(self) -> Any:
        if self._admin is None:
            from confluent_kafka.admin import AdminClient

            self._admin = AdminClient({"bootstrap.servers": self.bootstrap_servers})
        return self._admin

    def ensure_topics(self) -> None:
        """Idempotently ensure ``kg_tasks``/``kg_staging`` exist with at least
        the configured partition count. Grow-only: never shrinks an existing
        topic (Kafka cannot shrink partitions; we never try)."""
        admin = self._admin_client()
        md = admin.list_topics(timeout=_PROBE_TIMEOUT_S)
        wanted = ((TASKS_TOPIC, self.partitions), (STAGING_TOPIC, 1))
        to_create: list[tuple[str, int]] = []
        to_grow: list[tuple[str, int]] = []
        for topic, parts in wanted:
            existing = getattr(md, "topics", {}).get(topic)
            if existing is None:
                to_create.append((topic, parts))
            else:
                have = len(getattr(existing, "partitions", {}) or {})
                if 0 < have < parts:
                    to_grow.append((topic, parts))

        if to_create:
            from confluent_kafka.admin import NewTopic

            futures = admin.create_topics(
                [
                    NewTopic(t, num_partitions=p, replication_factor=1)
                    for t, p in to_create
                ]
            )
            for topic, fut in futures.items():
                try:
                    fut.result(timeout=_PROBE_TIMEOUT_S)
                    logger.info("Created Kafka topic %s", topic)
                except Exception as e:  # noqa: BLE001 — racing creators are fine
                    if "exists" not in str(e).lower():
                        raise
        if to_grow:
            from confluent_kafka.admin import NewPartitions

            futures = admin.create_partitions([NewPartitions(t, p) for t, p in to_grow])
            for topic, fut in futures.items():
                try:
                    fut.result(timeout=_PROBE_TIMEOUT_S)
                    logger.info(
                        "Grew Kafka topic %s to %d partitions",
                        topic,
                        dict(to_grow)[topic],
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("create_partitions(%s) failed: %s", topic, e)

    # ── QueueBackend: task submission ───────────────────────────────────

    def put(self, item: dict[str, Any]) -> None:
        if self._fallback_queue is not None:
            self._fallback_queue.put(item)
            return
        try:
            key = partition_key_for(item)
            self._producer.produce(
                TASKS_TOPIC,
                value=json.dumps(item).encode("utf-8"),
                key=key.encode("utf-8"),
            )
            self._producer.flush(5.0)
        except Exception as e:
            self._handle_unavailable("produce", e)
            # graceful path only: _handle_unavailable installed the fallback
            self._use_fallback().put(item)

    def _consumer(self, topic: str, group: str) -> Any:
        if self._consumer_factory is not None:
            return self._consumer_factory(topic=topic, group=group)
        from confluent_kafka import Consumer

        consumer = Consumer(
            {
                "bootstrap.servers": self.bootstrap_servers,
                "group.id": group,
                "enable.auto.commit": False,
                "auto.offset.reset": "earliest",
            }
        )
        consumer.subscribe([topic])
        return consumer

    def get(self) -> tuple[Any, dict[str, Any]] | None:
        if self._fallback_queue is not None:
            return self._fallback_queue.get()
        try:
            with self._lock:
                if self._task_consumer is None:
                    self._task_consumer = self._consumer(TASKS_TOPIC, INGEST_GROUP)
                msg = self._task_consumer.poll(0.5)
            if msg is None or msg.error():
                return None
            return msg, json.loads(msg.value().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 — poll is best-effort
            logger.debug("Kafka get failed or timed out: %s", e)
            return None

    def ack(self, item_id: Any) -> None:
        if self._fallback_queue is not None:
            self._fallback_queue.ack(item_id)
            return
        try:
            with self._lock:
                if self._task_consumer is not None:
                    self._task_consumer.commit(message=item_id, asynchronous=False)
        except Exception as e:  # noqa: BLE001
            logger.error("Kafka commit/ack failed: %s", e)

    # ── depth / lag backpressure visibility ── CONCEPT:KG-2.57

    def consumer_lag(self, topic: str = TASKS_TOPIC, group: str = INGEST_GROUP) -> int:
        """Total ``kg-ingest`` consumer-group lag on ``topic`` (unconsumed
        messages across all partitions). Uses a non-subscribing probe consumer
        so it never joins (and never steals partitions from) the group."""
        if self._fallback_queue is not None:
            return self._fallback_queue.get_queue_size()
        from confluent_kafka import TopicPartition

        if self._lag_probe is None:
            if self._consumer_factory is not None:
                self._lag_probe = self._consumer_factory(
                    topic=topic, group=group, probe=True
                )
            else:
                from confluent_kafka import Consumer

                self._lag_probe = Consumer(
                    {
                        "bootstrap.servers": self.bootstrap_servers,
                        "group.id": group,
                        "enable.auto.commit": False,
                    }
                )
        md = self._admin_client().list_topics(topic=topic, timeout=_PROBE_TIMEOUT_S)
        topic_md = getattr(md, "topics", {}).get(topic)
        if topic_md is None:
            return 0
        tps = [TopicPartition(topic, p) for p in topic_md.partitions]
        committed = self._lag_probe.committed(tps, timeout=_PROBE_TIMEOUT_S)
        lag = 0
        for tp in committed:
            lo, hi = self._lag_probe.get_watermark_offsets(tp, timeout=_PROBE_TIMEOUT_S)
            consumed = tp.offset if tp.offset >= 0 else lo
            lag += max(0, hi - consumed)
        return lag

    def get_queue_size(self) -> int:
        """Queue depth = unconsumed ``kg_tasks`` messages (consumer-group lag)."""
        if self._fallback_queue is not None:
            return self._fallback_queue.get_queue_size()
        try:
            return self.consumer_lag()
        except Exception as e:  # noqa: BLE001 — depth probe is best-effort
            logger.debug("Kafka lag probe failed: %s", e)
            return 0

    # ── QueueBackend: staged-graph queue ───────────────────────────────

    def put_staged_graph(self, job_id: str, nodes: list, edges: list) -> None:
        if self._fallback_queue is not None:
            self._fallback_queue.put_staged_graph(job_id, nodes, edges)
            return
        try:
            payload = {"job_id": job_id, "nodes": nodes, "edges": edges}
            self._producer.produce(
                STAGING_TOPIC,
                value=json.dumps(payload).encode("utf-8"),
                key=str(job_id).encode("utf-8"),
            )
            self._producer.flush(5.0)
        except Exception as e:
            self._handle_unavailable("produce(staging)", e)
            self._use_fallback().put_staged_graph(job_id, nodes, edges)

    def get_staged_graph(self) -> tuple[Any, str, dict[str, Any]] | None:
        if self._fallback_queue is not None:
            return self._fallback_queue.get_staged_graph()
        try:
            with self._lock:
                if self._staging_consumer is None:
                    self._staging_consumer = self._consumer(
                        STAGING_TOPIC, STAGING_GROUP
                    )
                msg = self._staging_consumer.poll(0.5)
            if msg is None or msg.error():
                return None
            payload = json.loads(msg.value().decode("utf-8"))
            return (
                msg,
                payload.get("job_id", ""),
                {"nodes": payload.get("nodes", []), "edges": payload.get("edges", [])},
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Kafka get_staged_graph failed: %s", e)
            return None

    def ack_staged_graph(self, item_id: Any) -> None:
        if self._fallback_queue is not None:
            self._fallback_queue.ack_staged_graph(item_id)
            return
        try:
            with self._lock:
                if self._staging_consumer is not None:
                    self._staging_consumer.commit(message=item_id, asynchronous=False)
        except Exception as e:  # noqa: BLE001
            logger.error("Kafka staged-graph commit failed: %s", e)
