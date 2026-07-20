# CONCEPT:AU-ECO.bus.pluggable-event-queue - Pluggable Event Queue Backend
# CONCEPT:AU-ORCH.reactive.event-sourcing-ledger - Reactive Event Sourcing

"""Production-grade Kafka ingest task queue.

CONCEPT:AU-KG.backend.selectable-queue-backend — Fail-closed selectable ingest task-queue backend: when Kafka is
selected (``TASK_QUEUE_BACKEND=kafka``) an unreachable broker raises
:class:`~.queue_backend.TaskQueueUnavailable` at startup instead of silently
degrading to the per-host SQLite file (which would split the fleet's queue into
invisible islands).

CONCEPT:AU-KG.backend.keyed-ingest-partitions — Keyed ingest partitions for per-tenant and per-repo ordering
without global serialization: every task is produced to the
``kg_tasks`` topic with a partition key so Kafka guarantees per-key ordering
without global serialization. Key hierarchy (first match wins):

1. ``tenant:<opaque-ref>`` — the ambient :class:`ActorContext` tenant (multi-tenant
   isolation ⇒ per-tenant ordering);
2. ``corpus:<opaque-ref>`` — the repo/corpus identifier of the ingest target
   (provenance ``full_path`` from the batch ingestor, else the path-derived
   repo root) ⇒ per-repo ordering for codebase ingest;
3. ``type:<opaque-ref>`` — the task type, the coarsest bucket.

Topic provisioning is idempotent at startup: ``kg_tasks`` is created with
``KG_TASKS_PARTITIONS`` partitions (grow-only — an existing topic with more
partitions is never shrunk; with fewer, partitions are added).

The decoupled ``kg-ingest`` consumer group lives in
:mod:`agent_utilities.knowledge_graph.ingest_worker` (CONCEPT:AU-KG.ingest.decoupled-kg-ingest-consumer); this
module owns the producer/topic/lag side. Uses ``confluent_kafka`` (a core
dependency), imported lazily. A selected Kafka authority is mandatory and
never switches to another queue after startup.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .queue_backend import QueueBackend, TaskQueueUnavailable

logger = logging.getLogger(__name__)

#: Task submission topic (kept from the original backend) and its staging twin.
TASKS_TOPIC = "kg_tasks"
STAGING_TOPIC = "kg_staging"
#: Consumer group for decoupled ingest workers (CONCEPT:AU-KG.ingest.decoupled-kg-ingest-consumer).
INGEST_GROUP = "kg-ingest"
STAGING_GROUP = "kg_staging_group"

_DEFAULT_BOOTSTRAP = "localhost:9092"
_PROBE_TIMEOUT_S = 5.0
_DELIVERY_TIMEOUT_S = 10.0
_PRODUCER_BATCH_LINGER_S = 0.010
_PRODUCER_BATCH_MAX = 256


@dataclass
class _PendingDelivery:
    """One produced record awaiting the batch delivery barrier."""

    event: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


@dataclass(frozen=True)
class _KafkaReceipt:
    """Opaque acknowledgement receipt bound to its owning consumer."""

    consumer: Any
    message: Any


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
    """Compute the partition key for a task or agent-turn envelope.

    CONCEPT:AU-KG.backend.keyed-ingest-partitions — key hierarchy: tenant id (ambient ActorContext) →
    repo/corpus identifier of the ingest target → task type. Guarantees
    per-tenant / per-repo ordering while letting unrelated work parallelize
    across partitions.

    CONCEPT:AU-ORCH.dispatch.parameterized-queue-topic — a ``session_id`` on the envelope outranks everything,
    INCLUDING the ambient tenant: agent turns of one session must execute
    serially (turn N+1 reads the state turn N wrote — interleaving corrupts
    the conversation), whereas tenant keying is only an ordering/fairness
    *grouping* for ingest work. Per-session keys still preserve tenant
    isolation — a session never spans tenants — so session beats tenant
    without weakening any KG-2.56 guarantee.
    """
    from agent_utilities.security.persistence_privacy import persistence_reference

    def opaque(kind: str, value: Any) -> str:
        return persistence_reference(kind, value, namespace="kafka-work-partition")

    session = item.get("session_id") or (item.get("props") or {}).get("session_id")
    if session:
        return f"session:{opaque('session', session)}"
    partition_ref = item.get("partition_ref")
    if partition_ref:
        return f"work:{opaque('work', partition_ref)}"

    try:
        from agent_utilities.security.brain_context import current_actor

        tenant = current_actor().tenant_id
        if tenant:
            return f"tenant:{opaque('tenant', tenant)}"
    except Exception:  # noqa: BLE001 — ambient identity is best-effort
        pass

    props = item.get("props") or {}
    # Batch-ingest provenance stamps the stable repo key directly (KG-2.49).
    full_path = props.get("full_path")
    if full_path:
        return f"corpus:{opaque('corpus', full_path)}"

    meta: dict[str, Any] = {}
    raw_meta = props.get("metadata")
    if raw_meta:
        from .engine_tasks import _decode_metadata

        meta = _decode_metadata(raw_meta)
    target = meta.get("target")
    if target:
        return f"corpus:{opaque('corpus', _corpus_root(target))}"

    task_type = meta.get("type") or item.get("type") or "task"
    return f"type:{opaque('task_type', task_type)}"


class KafkaQueueBackend(QueueBackend):
    """Kafka-backed durable task queue with keyed opaque partitions.

    An unavailable broker raises :class:`TaskQueueUnavailable` at startup and
    on produce failure. Queue authority never changes during a process lifetime.

    Test seams: ``producer``/``admin_client``/``consumer_factory`` accept
    pre-built (fake) confluent-kafka-shaped clients so unit tests never need a
    live broker or the ``confluent_kafka`` import.
    """

    def __init__(
        self,
        bootstrap_servers: str | list[str] | None = None,
        *,
        partitions: int = 6,
        producer: Any = None,
        admin_client: Any = None,
        consumer_factory: Any = None,
        tasks_topic: str = TASKS_TOPIC,
        consumer_group: str = INGEST_GROUP,
    ):
        # CONCEPT:AU-ORCH.dispatch.parameterized-queue-topic — the topic/group are parameters so the agent
        # dispatch queue (``agent_turns`` / ``agent-dispatch``) reuses this
        # backend verbatim; the defaults keep the ingest queue unchanged. The
        # staging twin only exists for the ingest topic.
        self.tasks_topic = tasks_topic
        self.consumer_group = consumer_group
        self._include_staging = tasks_topic == TASKS_TOPIC
        if isinstance(bootstrap_servers, list | tuple):
            bootstrap_servers = ",".join(str(s) for s in bootstrap_servers)
        if not bootstrap_servers:
            from agent_utilities.core.config import config as _cfg

            bootstrap_servers = (
                getattr(_cfg, "kafka_bootstrap_servers", None) or _DEFAULT_BOOTSTRAP
            )
        self.bootstrap_servers: str = bootstrap_servers
        self.partitions = max(1, int(partitions))
        self._producer: Any = producer
        self._admin: Any = admin_client
        self._consumer_factory = consumer_factory
        self._consumer_local = threading.local()
        self._lag_probe: Any = None
        self._lag_lock = threading.Lock()
        self._admission_lock = threading.Lock()
        self._produce_condition = threading.Condition()
        self._pending_deliveries: list[_PendingDelivery] = []
        self._flush_leader = False

        try:
            if self._producer is None:
                from confluent_kafka import Producer

                self._producer = Producer(
                    {
                        "bootstrap.servers": self.bootstrap_servers,
                        "socket.timeout.ms": 5000,
                        "message.timeout.ms": 10000,
                        "enable.idempotence": True,
                        "acks": "all",
                        "linger.ms": int(_PRODUCER_BATCH_LINGER_S * 1000),
                    }
                )
            self.ensure_topics()
            from agent_utilities.security.persistence_privacy import (
                persistence_reference,
            )

            logger.info(
                "Kafka task queue ready (broker_ref=%s, topic=%s, partitions>=%d, "
                "group_ref=%s)",
                persistence_reference(
                    "broker", self.bootstrap_servers, namespace="kafka-queue"
                ),
                self.tasks_topic,
                self.partitions,
                persistence_reference(
                    "consumer_group", self.consumer_group, namespace="kafka-queue"
                ),
            )
        except TaskQueueUnavailable:
            raise
        except Exception as e:
            self._handle_unavailable("connect/ensure-topic", e)

    # ── availability handling ──────────────────────────────────────────

    def _handle_unavailable(self, op: str, e: Exception) -> None:
        """Fail closed without exposing broker identity or exception payloads."""
        raise TaskQueueUnavailable(
            "configured Kafka task authority is unavailable "
            f"(operation={op}, error_type={type(e).__name__})"
        ) from None

    # ── topic provisioning ── CONCEPT:AU-KG.backend.keyed-ingest-partitions

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
        wanted: tuple[tuple[str, int], ...] = ((self.tasks_topic, self.partitions),)
        if self._include_staging:
            wanted += ((STAGING_TOPIC, 1),)
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
                    logger.warning(
                        "create_partitions(%s) failed (error_type=%s)",
                        topic,
                        type(e).__name__,
                    )

    # ── QueueBackend: task submission ───────────────────────────────────

    def put(self, item: dict[str, Any]) -> None:
        self.put_many([item])

    def put_many(self, items: list[dict[str, Any]]) -> None:
        """Publish records through one coalescing delivery-confirmed batch.

        Concurrent producers share the short linger window and one ``flush``
        barrier. Every caller still waits for delivery confirmation for its own
        records, so batching never weakens the durable-enqueue contract.
        """
        records = [
            (
                self.tasks_topic,
                json.dumps(item).encode("utf-8"),
                partition_key_for(item).encode("utf-8"),
            )
            for item in items
        ]
        self._publish_confirmed(records, operation="produce")

    def put_if_below(self, item: dict[str, Any], max_depth: int) -> bool:
        """Fail-closed admission against authoritative consumer-group lag.

        Kafka does not expose a cross-producer compare-and-append primitive;
        the broker's topic quota/retention is the hard storage bound. This gate
        is serialized per producer and rejects when the lag authority cannot be
        read. A small cross-producer race can admit at most concurrent writers,
        never turn a failed probe into apparent capacity.
        """
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        with self._admission_lock:
            if self.consumer_lag() >= max_depth:
                return False
            self.put(item)
            return True

    def _publish_confirmed(
        self,
        records: list[tuple[str, bytes, bytes]],
        *,
        operation: str,
    ) -> None:
        if not records:
            return
        mine: list[_PendingDelivery] = []
        batch: list[_PendingDelivery] = []
        leader = False
        try:
            with self._produce_condition:
                for topic, value, key in records:
                    pending = _PendingDelivery()

                    def delivered(
                        error: Any,
                        _message: Any,
                        *,
                        delivery: _PendingDelivery = pending,
                    ) -> None:
                        if error is not None:
                            delivery.error = RuntimeError(
                                "Kafka broker rejected a queued record"
                            )
                        delivery.event.set()

                    self._producer.produce(
                        topic,
                        value=value,
                        key=key,
                        on_delivery=delivered,
                    )
                    mine.append(pending)
                    self._pending_deliveries.append(pending)

                if not self._flush_leader:
                    self._flush_leader = True
                    leader = True
                if len(self._pending_deliveries) >= _PRODUCER_BATCH_MAX:
                    self._produce_condition.notify_all()

            if leader:
                deadline = time.monotonic() + _PRODUCER_BATCH_LINGER_S
                with self._produce_condition:
                    while (
                        len(self._pending_deliveries) < _PRODUCER_BATCH_MAX
                        and time.monotonic() < deadline
                    ):
                        self._produce_condition.wait(
                            timeout=max(0.0, deadline - time.monotonic())
                        )
                    batch = self._pending_deliveries
                    self._pending_deliveries = []
                    remaining = int(self._producer.flush(_DELIVERY_TIMEOUT_S) or 0)
                    if remaining:
                        failure = TimeoutError(
                            "Kafka delivery barrier expired with records pending"
                        )
                        for pending in batch:
                            if not pending.event.is_set():
                                pending.error = failure
                                pending.event.set()
                    else:
                        # Test doubles and alternate clients may report a fully
                        # drained producer without invoking callbacks.
                        for pending in batch:
                            pending.event.set()
                    self._flush_leader = False
                    self._produce_condition.notify_all()

            deadline = time.monotonic() + _DELIVERY_TIMEOUT_S
            for pending in mine:
                if not pending.event.wait(max(0.0, deadline - time.monotonic())):
                    raise TimeoutError("Kafka delivery confirmation timed out")
                if pending.error is not None:
                    raise pending.error
        except Exception as e:
            with self._produce_condition:
                failed = list(batch or mine)
                if leader:
                    failed.extend(self._pending_deliveries)
                    self._pending_deliveries = []
                    self._flush_leader = False
                else:
                    self._pending_deliveries = [
                        pending
                        for pending in self._pending_deliveries
                        if pending not in mine
                    ]
                failure = RuntimeError("Kafka confirmed batch publication failed")
                for pending in failed:
                    if not pending.event.is_set():
                        pending.error = failure
                        pending.event.set()
                self._produce_condition.notify_all()
            self._handle_unavailable(operation, e)

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

    def _thread_consumer(self, attribute: str, topic: str, group: str) -> Any:
        consumer = getattr(self._consumer_local, attribute, None)
        if consumer is None:
            consumer = self._consumer(topic, group)
            setattr(self._consumer_local, attribute, consumer)
        return consumer

    def get(self) -> tuple[Any, dict[str, Any]] | None:
        try:
            consumer = self._thread_consumer(
                "task_consumer", self.tasks_topic, self.consumer_group
            )
            msg = consumer.poll(0.5)
            if msg is None or msg.error():
                return None
            return _KafkaReceipt(consumer, msg), json.loads(msg.value().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 — poll is best-effort
            logger.debug("Kafka get failed or timed out (%s)", type(e).__name__)
            return None

    def ack(self, item_id: Any) -> None:
        try:
            if not isinstance(item_id, _KafkaReceipt):
                raise TypeError("Kafka acknowledgement requires an owning receipt")
            item_id.consumer.commit(message=item_id.message, asynchronous=False)
        except Exception as e:  # noqa: BLE001
            self._handle_unavailable("ack", e)

    # ── depth / lag backpressure visibility ── CONCEPT:AU-KG.ingest.decoupled-kg-ingest-consumer

    def consumer_lag(self, topic: str | None = None, group: str | None = None) -> int:
        """Total consumer-group lag on ``topic`` (unconsumed messages across
        all partitions; defaults to this queue's topic/group). Uses a
        non-subscribing probe consumer so it never joins (and never steals
        partitions from) the group."""
        topic = topic or self.tasks_topic
        group = group or self.consumer_group
        from confluent_kafka import TopicPartition

        with self._lag_lock:
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
                lo, hi = self._lag_probe.get_watermark_offsets(
                    tp, timeout=_PROBE_TIMEOUT_S
                )
                consumed = tp.offset if tp.offset >= 0 else lo
                lag += max(0, hi - consumed)
            return lag

    def get_queue_size(self) -> int:
        """Queue depth = unconsumed task-topic messages (consumer-group lag)."""
        return self.consumer_lag()

    # ── QueueBackend: staged-graph queue ───────────────────────────────

    def put_staged_graph(self, job_id: str, nodes: list, edges: list) -> None:
        payload = {"job_id": job_id, "nodes": nodes, "edges": edges}
        self._publish_confirmed(
            [
                (
                    STAGING_TOPIC,
                    json.dumps(payload).encode("utf-8"),
                    str(job_id).encode("utf-8"),
                )
            ],
            operation="produce(staging)",
        )

    def get_staged_graph(self) -> tuple[Any, str, dict[str, Any]] | None:
        try:
            consumer = self._thread_consumer(
                "staging_consumer", STAGING_TOPIC, STAGING_GROUP
            )
            msg = consumer.poll(0.5)
            if msg is None or msg.error():
                return None
            payload = json.loads(msg.value().decode("utf-8"))
            return (
                _KafkaReceipt(consumer, msg),
                payload.get("job_id", ""),
                {"nodes": payload.get("nodes", []), "edges": payload.get("edges", [])},
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Kafka get_staged_graph failed (%s)", type(e).__name__)
            return None

    def ack_staged_graph(self, item_id: Any) -> None:
        try:
            if not isinstance(item_id, _KafkaReceipt):
                raise TypeError("Kafka acknowledgement requires an owning receipt")
            item_id.consumer.commit(message=item_id.message, asynchronous=False)
        except Exception as e:  # noqa: BLE001
            self._handle_unavailable("ack(staging)", e)
