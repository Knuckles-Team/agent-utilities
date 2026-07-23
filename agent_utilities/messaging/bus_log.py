"""Durable partitioned log — the AgentBus delivery/wakeup plane (CONCEPT:AU-ECO.bus.partitioned-log-delivery, AU-P1-2).

``AgentBus`` (``messaging/bus.py``) kept the semantic registry (who is on the
bus, who subscribes to what, topic metadata — ``:BusAgent``/``:Topic``/
``:BusSubscription``) as durable KG nodes, which is right: that state is small
and low-churn. But it *also* wrote one ``:BusMessage`` graph node PER RECIPIENT
on every ``send()`` (fan-out) and read the mailbox via a property-scoped
``MATCH (m:BusMessage {recipient/topic:...})`` scan on every ``receive()`` —
O(agents) writes and O(history) reads, on a graph store that is not a queue.

This module is the fix: a **durable partitioned log** carries the high-volume
message BODIES as the delivery/wakeup plane, with real offsets/consumer
cursors instead of a graph MATCH, a DLQ for poison messages, and backpressure
via queue depth — while the semantic registry stays exactly where it was.

Two backends are selected explicitly by AgentConfig:

1. **engine** — the epistemic-graph engine's NATIVE AMQP-style message broker
   (the same surface the ``graph_broker`` MCP tool exposes:
   ``declare_exchange``/``declare_queue``/``bind_queue``/``publish``/``consume``/
   acknowledgement, reached through the same ``SyncEpistemicGraphClient`` every
   other engine-surface tool uses). This is the best fit for AgentBus's
   point-to-point + pub/sub shape. A fixed number of tenant-qualified queues
   bound to stable routing partitions bounds queues and consumers independently
   of agent or subscription cardinality.
2. **kafka** — the existing ``confluent_kafka`` stack (mirrors
   ``kafka_queue_backend.py``'s keyed-partition conventions) when the engine
   broker is not configured/reachable. Two keyed topics
   (``agent_bus_direct``/``agent_bus_topic``), tenant-qualified partition
   keys and a fixed materializer consumer group per tenant/topic pair. There
   are no per-agent or per-subscription consumers.

CONCEPT:AU-ECO.bus.partitioned-log-delivery — durable partitioned log as the AgentBus delivery/wakeup plane
CONCEPT:AU-ECO.bus.log-backend-resolution — required engine-broker or Kafka log
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

from agent_utilities.messaging.bus_privacy import bus_reference, sanitize_bus_content

logger = logging.getLogger(__name__)

BUS_LOG_BACKENDS = ("engine", "kafka")

#: Kafka topics (mirrors ``kafka_queue_backend.py``'s naming conventions).
DIRECT_TOPIC = "agent_bus_direct"
TOPIC_TOPIC = "agent_bus_topic"
DLQ_TOPIC = "agent_bus_dlq"

_PROBE_TIMEOUT_S = 5.0
_DEFAULT_BOOTSTRAP = "localhost:9092"


class BusLogUnavailable(RuntimeError):
    """The selected bus-log backend is unreachable.

    Mirrors :class:`~agent_utilities.knowledge_graph.core.queue_backend.TaskQueueUnavailable`:
    ``AGENT_BUS_LOG_BACKEND`` is a hard contract with no alternate writer.
    """


# ── tenant-qualified keying (CONCEPT:AU-KG.backend.keyed-ingest-partitions, reused convention) ──
def current_bus_tenant() -> str:
    """The ambient tenant id, or ``"default"`` outside any tenant scope.

    Every partition key / queue / exchange name this module mints is
    tenant-qualified with this value, so bus traffic from one tenant can never
    land in another tenant's queue even when both share the same Kafka topic
    or engine broker connection.
    """
    try:
        from agent_utilities.security.brain_context import current_actor

        tenant = current_actor().tenant_id
        return tenant or "default"
    except Exception:  # noqa: BLE001 — ambient identity is best-effort
        return "default"


def bus_partition_key(tenant: str, target: str) -> str:
    """The tenant-qualified partition/routing key for one recipient or topic."""
    tenant_ref = bus_reference("tenant", tenant or "default")
    target_ref = bus_reference("route", target, tenant=tenant or "default")
    return f"{tenant_ref}:{target_ref}"


# ── envelope shape (matches AgentBus._shape_message so callers never change) ──
def encode_envelope(
    *,
    tenant: str,
    group: str,
    sender: str,
    recipient: str,
    topic: str,
    payload: str,
    meta_json: str,
    created: float,
) -> dict[str, Any]:
    tenant_ref = bus_reference("tenant", tenant or "default")
    group = bus_reference("message_group", group, tenant=tenant_ref)
    sender = bus_reference("agent", sender, tenant=tenant_ref)
    recipient = bus_reference("agent", recipient, tenant=tenant_ref)
    topic = bus_reference("topic", topic, tenant=tenant_ref)
    try:
        metadata = json.loads(meta_json) if meta_json else {}
    except (TypeError, ValueError):
        metadata = {}
    payload, meta_json, _privacy_report = sanitize_bus_content(payload, metadata)
    return {
        "id": f"busmsg:{group}:{recipient or topic}:{uuid.uuid4().hex}",
        "msg_group": group,
        "sender": sender,
        "recipient": recipient,
        "topic": topic,
        "payload": payload,
        "meta": meta_json,
        "status": "sent",
        "created": created,
        "tenant": tenant_ref,
    }


def decode_envelope(raw: Any) -> dict[str, Any] | None:
    """Decode one wire message back to the envelope shape, or ``None`` if poison."""
    try:
        if isinstance(raw, dict):
            obj = raw
        elif isinstance(raw, bytes | bytearray):
            obj = json.loads(raw.decode("utf-8"))
        else:
            obj = json.loads(raw)
        if not isinstance(obj, dict) or "payload" not in obj:
            return None
        if not all(obj.get(field) for field in ("tenant", "msg_group", "sender")):
            return None
        if bool(obj.get("recipient")) == bool(obj.get("topic")):
            return None
        return obj
    except (TypeError, ValueError, UnicodeDecodeError):
        return None


class BusLogBackend(ABC):
    """Interface every bus-log backend implements. Never instantiated directly."""

    name: str = "abstract"

    @abstractmethod
    def publish_direct(
        self,
        *,
        tenant: str,
        group: str,
        sender: str,
        to: str,
        payload: str,
        meta_json: str,
        created: float,
    ) -> bool: ...

    @abstractmethod
    def publish_topic(
        self,
        *,
        tenant: str,
        group: str,
        sender: str,
        topic: str,
        payload: str,
        meta_json: str,
        created: float,
    ) -> bool: ...

    @abstractmethod
    def receive(
        self,
        *,
        tenant: str,
        agent_id: str,
        topics: list[str],
        max_messages: int = 200,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def ack(self, message: dict[str, Any]) -> bool:
        """Commit one receipt after its durable inbox/WorkItem transaction."""
        ...

    @abstractmethod
    def nack(self, message: dict[str, Any], *, requeue: bool = True) -> bool:
        """Release a receipt whose inbox transaction failed."""
        ...

    @abstractmethod
    def read_dlq(
        self, *, tenant: str, max_messages: int = 50
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def stats(self) -> dict[str, Any]: ...


# ══════════════════════════════════════════════════════════════════════════
# Engine-native broker (AMQP-style exchanges/queues) — CONCEPT:AU-KG.coordination.engine-message-broker
# ══════════════════════════════════════════════════════════════════════════
class EngineBrokerBusLog(BusLogBackend):
    """AgentBus delivery over the epistemic-graph engine's native message broker.

    Direct and topic events share one direct exchange per tenant and route to
    ``AGENT_BUS_PARTITIONS`` durable queues. Materializers expand topic events
    from the semantic subscription registry while direct events retain their
    recipient in the envelope. Queue count is independent of agent cardinality.

    Declarations are idempotent and cached per-process so a hot ``send``/
    ``receive`` does not redeclare infrastructure on every call.
    """

    name = "engine"

    def __init__(
        self, client: Any, *, partitions: int = 6, delivery_lease_ms: int = 300_000
    ) -> None:
        self._client = client
        self.partitions = max(1, int(partitions))
        self.delivery_lease_ms = max(30_000, int(delivery_lease_ms))
        self._declared: set[str] = set()  # exchange/queue/bind idempotency cache
        self._lock = threading.Lock()

    def _broker(self) -> Any:
        broker = getattr(self._client, "broker", None)
        if broker is None:
            raise BusLogUnavailable(
                "the connected engine client has no 'broker' surface"
            )
        return broker

    def _declare_once(self, key: str, fn: Any, **kwargs: Any) -> None:
        with self._lock:
            if key in self._declared:
                return
            fn(**kwargs)
            self._declared.add(key)

    def _exchange(self, tenant: str) -> str:
        tenant = bus_reference("tenant", tenant or "default")
        exch = f"bus.log.{tenant}"
        self._declare_once(
            f"exch:{exch}",
            self._broker().declare_exchange,
            exchange=exch,
            kind="direct",
        )
        return exch

    def _partition(self, target: str) -> int:
        digest = hashlib.sha256(target.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.partitions

    def _partition_queue(self, tenant: str, partition: int) -> tuple[str, str]:
        tenant = bus_reference("tenant", tenant or "default")
        exch = self._exchange(tenant)
        route = f"p{int(partition)}"
        queue = f"bus.partition.{tenant}.{route}"
        self._declare_once(f"queue:{queue}", self._broker().declare_queue, queue=queue)
        self._declare_once(
            f"bind:{queue}:{exch}:{route}",
            self._broker().bind_queue,
            queue=queue,
            exchange=exch,
            routing_key=route,
        )
        return exch, queue

    def publish_direct(
        self,
        *,
        tenant: str,
        group: str,
        sender: str,
        to: str,
        payload: str,
        meta_json: str,
        created: float,
    ) -> bool:
        envelope = encode_envelope(
            tenant=tenant,
            group=group,
            sender=sender,
            recipient=to,
            topic="",
            payload=payload,
            meta_json=meta_json,
            created=created,
        )
        partition = self._partition(str(envelope["recipient"]))
        exch, _ = self._partition_queue(tenant, partition)
        return self._publish(exch, f"p{partition}", envelope)

    def publish_topic(
        self,
        *,
        tenant: str,
        group: str,
        sender: str,
        topic: str,
        payload: str,
        meta_json: str,
        created: float,
    ) -> bool:
        envelope = encode_envelope(
            tenant=tenant,
            group=group,
            sender=sender,
            recipient="",
            topic=topic,
            payload=payload,
            meta_json=meta_json,
            created=created,
        )
        partition = self._partition(str(envelope["topic"]))
        exch, _ = self._partition_queue(tenant, partition)
        return self._publish(exch, f"p{partition}", envelope)

    # ── consume ──
    def _publish(
        self, exchange: str, routing_key: str, envelope: dict[str, Any]
    ) -> bool:
        try:
            delivered = self._broker().publish(
                exchange=exchange,
                routing_key=routing_key,
                payload=json.dumps(
                    envelope,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8"),
            )
            return int(delivered) > 0
        except Exception as exc:  # noqa: BLE001 — publish failure is data, not a crash
            logger.warning(
                "[AU-P1-2] engine broker publish(exchange=%s) failed: %s",
                exchange,
                type(exc).__name__,
            )
            return False

    def _drain(
        self, queue: str, max_messages: int, *, dlq_tenant: str
    ) -> list[dict[str, Any]]:
        shaped: list[dict[str, Any]] = []
        for _ in range(max(0, int(max_messages))):
            try:
                claimed = self._broker().consume(
                    queue=queue,
                    group="agent-bus",
                    consumer="inbox-materializer",
                    now_ms=int(time.time() * 1000),
                    lease_ms=self.delivery_lease_ms,
                    prefetch=max_messages,
                )
            except Exception as exc:  # noqa: BLE001 — an unavailable broker is visible through health
                logger.debug(
                    "[AU-P1-2] engine broker consume(%s) failed (%s)",
                    queue,
                    type(exc).__name__,
                )
                break
            if claimed is None:
                break
            if (
                not isinstance(claimed, (list, tuple))
                or len(claimed) != 2
                or not isinstance(claimed[1], dict)
            ):
                raise BusLogUnavailable(
                    "engine broker returned an invalid current consume result"
                )
            node_id, properties = claimed
            delivery_tag = properties.get("delivery_tag")
            if delivery_tag is None:
                raise BusLogUnavailable(
                    "engine broker returned an unacknowledgeable delivery without delivery_tag"
                )
            payload = _decode_engine_payload(properties.get("payload"))
            env = decode_envelope(payload)
            if env is None:
                self._to_dlq(dlq_tenant, queue, payload, "decode_error")
                self._nack_tag(int(delivery_tag), requeue=False)
                continue
            env["_receipt"] = {
                "backend": self.name,
                "delivery_tag": int(delivery_tag),
                "queue": queue,
                "node_id": str(node_id),
            }
            shaped.append(env)
        return shaped

    def _ack_tag(self, delivery_tag: int) -> bool:
        method = getattr(self._broker(), "ack_tag", None)
        if not callable(method):
            raise BusLogUnavailable("engine broker has no ack_tag method")
        return bool(method(delivery_tag=delivery_tag))

    def _nack_tag(self, delivery_tag: int, *, requeue: bool) -> bool:
        method = getattr(self._broker(), "nack_tag", None)
        if not callable(method):
            raise BusLogUnavailable("engine broker has no nack_tag method")
        outcome = method(
            delivery_tag=delivery_tag,
            requeue=requeue,
            now_ms=int(time.time() * 1000),
        )
        return str(outcome).lower() not in {"absent", "false", "none"}

    def ack(self, message: dict[str, Any]) -> bool:
        receipt = message.get("_receipt") or {}
        if receipt.get("backend") != self.name or receipt.get("delivery_tag") is None:
            return False
        return self._ack_tag(int(receipt["delivery_tag"]))

    def nack(self, message: dict[str, Any], *, requeue: bool = True) -> bool:
        receipt = message.get("_receipt") or {}
        if receipt.get("backend") != self.name or receipt.get("delivery_tag") is None:
            return False
        return self._nack_tag(int(receipt["delivery_tag"]), requeue=requeue)

    def receive(
        self,
        *,
        tenant: str,
        agent_id: str,
        topics: list[str],
        max_messages: int = 200,
    ) -> list[dict[str, Any]]:
        tenant = bus_reference("tenant", tenant or "default")
        del agent_id, topics
        out: list[dict[str, Any]] = []
        remaining = max(0, int(max_messages))
        for partition in range(self.partitions):
            if remaining == 0:
                break
            remaining_partitions = self.partitions - partition
            quota = max(
                1, (remaining + remaining_partitions - 1) // remaining_partitions
            )
            _, queue = self._partition_queue(tenant, partition)
            batch = self._drain(queue, quota, dlq_tenant=tenant)
            out.extend(batch)
            remaining -= len(batch)
        out.sort(key=lambda m: float(m.get("created", 0) or 0))
        return out

    def _dlq_queue(self, tenant: str) -> tuple[str, str, str]:
        tenant = bus_reference("tenant", tenant or "default")
        exchange = self._exchange(tenant)
        routing_key = "dlq"
        queue = f"bus.dlq.{tenant}"
        self._declare_once(f"queue:{queue}", self._broker().declare_queue, queue=queue)
        self._declare_once(
            f"bind:{queue}:{exchange}:{routing_key}",
            self._broker().bind_queue,
            exchange=exchange,
            queue=queue,
            routing_key=routing_key,
        )
        return exchange, queue, routing_key

    def _to_dlq(self, tenant: str, source_queue: str, raw: bytes, error: str) -> None:
        exchange, _dlq, routing_key = self._dlq_queue(tenant)
        raw_bytes = bytes(raw)
        payload = json.dumps(
            {
                "source_queue": source_queue,
                "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "raw_bytes": len(raw_bytes),
                "error": error,
                "ts": time.time(),
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        try:
            self._broker().publish(
                exchange=exchange,
                routing_key=routing_key,
                payload=payload.encode("utf-8"),
            )
        except Exception as exc:  # noqa: BLE001 — DLQ is best-effort, never blocks delivery
            logger.warning(
                "[AU-P1-2] engine broker DLQ publish failed (%s)",
                type(exc).__name__,
            )

    def read_dlq(self, *, tenant: str, max_messages: int = 50) -> list[dict[str, Any]]:
        _exchange, dlq, _routing_key = self._dlq_queue(tenant)
        out: list[dict[str, Any]] = []
        for _ in range(max(0, int(max_messages))):
            try:
                claimed = self._broker().consume(
                    queue=dlq,
                    group="agent-bus-dlq",
                    consumer="operator",
                    now_ms=int(time.time() * 1000),
                    lease_ms=self.delivery_lease_ms,
                    prefetch=max_messages,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[AU-P1-2] engine broker DLQ read failed (%s)",
                    type(exc).__name__,
                )
                break
            if claimed is None:
                break
            if (
                not isinstance(claimed, (list, tuple))
                or len(claimed) != 2
                or not isinstance(claimed[1], dict)
            ):
                raise BusLogUnavailable(
                    "engine broker returned an invalid current consume result"
                )
            _node_id, properties = claimed
            raw = _decode_engine_payload(properties.get("payload"))
            try:
                out.append(json.loads(raw))
            except (TypeError, ValueError, UnicodeDecodeError):
                out.append({"raw_sha256": hashlib.sha256(raw).hexdigest()})
            delivery_tag = properties.get("delivery_tag")
            if delivery_tag is None or not self._ack_tag(int(delivery_tag)):
                raise BusLogUnavailable("engine broker could not acknowledge DLQ read")
        return out

    def stats(self) -> dict[str, Any]:
        return {"backend": self.name, "partitions": self.partitions}


def _decode_engine_payload(value: Any) -> bytes:
    """Decode the current engine broker's hex-encoded graph property."""
    if isinstance(value, bytes | bytearray):
        return bytes(value)
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return bytes(value)
    if not isinstance(value, str):
        return b""
    try:
        return bytes.fromhex(value)
    except ValueError:
        return b""


# ══════════════════════════════════════════════════════════════════════════
# Kafka backend — CONCEPT:AU-KG.backend.keyed-ingest-partitions (reused convention)
# ══════════════════════════════════════════════════════════════════════════
class KafkaBusLog(BusLogBackend):
    """AgentBus delivery over Kafka when the engine broker is not configured/reachable.

    Two keyed topics carry the two delivery shapes (mirrors
    ``kafka_queue_backend.py``'s idempotent grow-only topic provisioning):

    * ``agent_bus_direct`` — direct sends, keyed ``tenant:recipient``.
    * ``agent_bus_topic``  — topic sends, keyed ``tenant:topic``.

    A fixed materializer consumer group per tenant and wire topic drains all
    records and commits them into recipient inboxes. Consumer count is bounded
    by ``AGENT_BUS_MAX_CONSUMERS`` and never scales with agents/subscriptions.

    Test seams: ``producer``/``admin_client``/``consumer_factory`` accept
    pre-built fake confluent-kafka-shaped clients, same DI pattern as
    :class:`~agent_utilities.knowledge_graph.core.kafka_queue_backend.KafkaQueueBackend`.
    """

    name = "kafka"

    def __init__(
        self,
        *,
        bootstrap_servers: str | None = None,
        partitions: int = 6,
        producer: Any = None,
        admin_client: Any = None,
        consumer_factory: Any = None,
        fail_loud: bool = False,
        max_consumers: int = 32,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers or _DEFAULT_BOOTSTRAP
        self.partitions = max(1, int(partitions))
        self.fail_loud = fail_loud
        self.max_consumers = max(2, int(max_consumers))
        self._producer: Any = producer
        self._admin: Any = admin_client
        self._consumer_factory = consumer_factory
        self._consumers: dict[tuple[Any, ...], Any] = {}
        self._receipt_seq = 0
        self._pending_receipts: dict[str, tuple[Any, Any, tuple[Any, ...]]] = {}
        self._lock = threading.Lock()
        try:
            self._ensure_topics()
        except Exception as exc:  # noqa: BLE001 — surface one typed failure
            raise BusLogUnavailable(
                "the Kafka bus-log backend is unavailable: "
                "the configured broker could not be reached/provisioned "
                f"({type(exc).__name__}). Start the Kafka stack and check "
                "KAFKA_BOOTSTRAP_SERVERS."
            ) from exc

    # ── admin / topic provisioning (mirrors kafka_queue_backend.ensure_topics) ──
    def _admin_client(self) -> Any:
        if self._admin is None:
            from confluent_kafka.admin import AdminClient

            self._admin = AdminClient({"bootstrap.servers": self.bootstrap_servers})
        return self._admin

    def _ensure_topics(self) -> None:
        admin = self._admin_client()
        md = admin.list_topics(timeout=_PROBE_TIMEOUT_S)
        wanted = (
            (DIRECT_TOPIC, self.partitions),
            (TOPIC_TOPIC, self.partitions),
            (DLQ_TOPIC, 1),
        )
        to_create = []
        to_grow = []
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
                except Exception as e:  # noqa: BLE001 — racing creators are fine
                    if "exists" not in str(e).lower():
                        raise
        if to_grow:
            from confluent_kafka.admin import NewPartitions

            futures = admin.create_partitions([NewPartitions(t, p) for t, p in to_grow])
            for topic, fut in futures.items():
                try:
                    fut.result(timeout=_PROBE_TIMEOUT_S)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "create_partitions(%s) failed (%s)", topic, type(e).__name__
                    )

    def _producer_client(self) -> Any:
        if self._producer is None:
            from confluent_kafka import Producer

            self._producer = Producer(
                {
                    "bootstrap.servers": self.bootstrap_servers,
                    "socket.timeout.ms": 5000,
                    "message.timeout.ms": 10000,
                }
            )
        return self._producer

    def _publish(self, topic: str, key: str, envelope: dict[str, Any]) -> bool:
        try:
            self._producer_client().produce(
                topic,
                value=json.dumps(
                    envelope,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8"),
                key=key.encode("utf-8"),
            )
            self._producer_client().flush(5.0)
            return True
        except Exception as exc:  # noqa: BLE001 — publish failure is data, not a crash
            logger.warning(
                "[AU-P1-2] kafka bus publish(%s) failed (%s)",
                topic,
                type(exc).__name__,
            )
            return False

    def publish_direct(
        self,
        *,
        tenant: str,
        group: str,
        sender: str,
        to: str,
        payload: str,
        meta_json: str,
        created: float,
    ) -> bool:
        key = bus_partition_key(tenant, to)
        envelope = encode_envelope(
            tenant=tenant,
            group=group,
            sender=sender,
            recipient=to,
            topic="",
            payload=payload,
            meta_json=meta_json,
            created=created,
        )
        return self._publish(DIRECT_TOPIC, key, envelope)

    def publish_topic(
        self,
        *,
        tenant: str,
        group: str,
        sender: str,
        topic: str,
        payload: str,
        meta_json: str,
        created: float,
    ) -> bool:
        key = bus_partition_key(tenant, topic)
        envelope = encode_envelope(
            tenant=tenant,
            group=group,
            sender=sender,
            recipient="",
            topic=topic,
            payload=payload,
            meta_json=meta_json,
            created=created,
        )
        return self._publish(TOPIC_TOPIC, key, envelope)

    # ── bounded tenant materializer consumers ──
    def _consumer(
        self,
        cache_key: tuple[Any, ...],
        *,
        group_id: str,
        topic: str,
        default_offset: str = "latest",
    ) -> Any:
        """Get-or-create the cached consumer for ``cache_key``.

        ``default_offset`` governs a brand-new materializer group's starting
        position. Delivery materializers use ``earliest`` so every uncommitted
        record can be recovered after downtime.
        """
        with self._lock:
            existing = self._consumers.get(cache_key)
            if existing is not None:
                return existing
            if len(self._consumers) >= self.max_consumers:
                raise BusLogUnavailable(
                    f"AgentBus Kafka consumer bound reached ({self.max_consumers})"
                )
            if self._consumer_factory is not None:
                consumer = self._consumer_factory(
                    topic=topic,
                    group=group_id,
                    default_offset=default_offset,
                )
            else:
                consumer = self._build_real_consumer(
                    group_id=group_id,
                    topic=topic,
                    default_offset=default_offset,
                )
            self._consumers[cache_key] = consumer
            return consumer

    def _build_real_consumer(
        self,
        *,
        group_id: str,
        topic: str,
        default_offset: str = "latest",
    ) -> Any:
        from confluent_kafka import Consumer

        consumer = Consumer(
            {
                "bootstrap.servers": self.bootstrap_servers,
                "group.id": group_id,
                "enable.auto.commit": False,
                "auto.offset.reset": default_offset,
            }
        )
        consumer.subscribe([topic])
        return consumer

    def _drain(
        self,
        consumer: Any,
        *,
        max_messages: int,
        dlq_tenant: str,
        source_topic: str,
    ) -> list[dict[str, Any]]:
        """Poll records without committing matching deliveries.

        Records carry an opaque receipt and remain uncommitted until
        :meth:`ack`; one bounded tenant materializer group owns each topic.
        """
        out: list[dict[str, Any]] = []
        seen = 0
        while seen < max_messages:
            msg = consumer.poll(0.5)
            if msg is None:
                break
            seen += 1
            if msg.error():
                continue
            env = decode_envelope(msg.value())
            if env is None:
                self._to_dlq(dlq_tenant, source_topic, msg.value(), "decode_error")
                consumer.commit(message=msg, asynchronous=False)
                continue
            self._receipt_seq += 1
            receipt_id = f"kafka:{self._receipt_seq}"
            cache_key = next(
                (key for key, value in self._consumers.items() if value is consumer),
                ("unknown",),
            )
            self._pending_receipts[receipt_id] = (consumer, msg, cache_key)
            env["_receipt"] = {"backend": self.name, "receipt_id": receipt_id}
            out.append(env)
        return out

    def ack(self, message: dict[str, Any]) -> bool:
        receipt = message.get("_receipt") or {}
        receipt_id = str(receipt.get("receipt_id") or "")
        pending = self._pending_receipts.pop(receipt_id, None)
        if receipt.get("backend") != self.name or pending is None:
            return False
        consumer, kafka_message, _cache_key = pending
        consumer.commit(message=kafka_message, asynchronous=False)
        return True

    def nack(self, message: dict[str, Any], *, requeue: bool = True) -> bool:
        receipt = message.get("_receipt") or {}
        receipt_id = str(receipt.get("receipt_id") or "")
        pending = self._pending_receipts.pop(receipt_id, None)
        if receipt.get("backend") != self.name or pending is None:
            return False
        consumer, kafka_message, cache_key = pending
        if not requeue:
            consumer.commit(message=kafka_message, asynchronous=False)
            return True
        seek = getattr(consumer, "seek", None)
        topic = getattr(kafka_message, "topic", None)
        partition = getattr(kafka_message, "partition", None)
        offset = getattr(kafka_message, "offset", None)
        if (
            seek is not None
            and callable(seek)
            and all(callable(value) for value in (topic, partition, offset))
        ):
            try:
                from confluent_kafka import TopicPartition

                seek(TopicPartition(topic(), partition(), offset()))
                return True
            except Exception:  # noqa: BLE001 - rebuild from committed offset below
                pass
        # A fake/older client without seek is discarded. Its replacement starts
        # from the last committed offset, so the unacked message is redelivered.
        with self._lock:
            self._consumers.pop(cache_key, None)
        close = getattr(consumer, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()
        return True

    def receive(
        self,
        *,
        tenant: str,
        agent_id: str,
        topics: list[str],
        max_messages: int = 200,
    ) -> list[dict[str, Any]]:
        tenant = bus_reference("tenant", tenant or "default")
        del agent_id, topics
        out: list[dict[str, Any]] = []
        direct_key = ("direct", tenant)
        direct_consumer = self._consumer(
            direct_key,
            group_id=f"agentbus-materializer-{tenant}-direct",
            topic=DIRECT_TOPIC,
            default_offset="earliest",
        )
        out.extend(
            self._drain(
                direct_consumer,
                max_messages=max_messages,
                dlq_tenant=tenant,
                source_topic=DIRECT_TOPIC,
            )
        )
        topic_key = ("topic", tenant)
        topic_consumer = self._consumer(
            topic_key,
            group_id=f"agentbus-materializer-{tenant}-topic",
            topic=TOPIC_TOPIC,
            default_offset="earliest",
        )
        out.extend(
            self._drain(
                topic_consumer,
                max_messages=max_messages,
                dlq_tenant=tenant,
                source_topic=TOPIC_TOPIC,
            )
        )
        out.sort(key=lambda m: float(m.get("created", 0) or 0))
        return out[:max_messages]

    def _to_dlq(self, tenant: str, source_topic: str, raw: Any, error: str) -> None:
        raw_bytes = _kafka_wire_bytes(raw)
        payload = {
            "source_topic": source_topic,
            "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "raw_bytes": len(raw_bytes),
            "error": error,
            "ts": time.time(),
        }
        self._publish(DLQ_TOPIC, bus_partition_key(tenant, "dlq"), payload)

    def read_dlq(self, *, tenant: str, max_messages: int = 50) -> list[dict[str, Any]]:
        cache_key = ("dlq", tenant)
        consumer = self._consumer(
            cache_key,
            group_id=f"agentbus-dlq-reader-{tenant}",
            topic=DLQ_TOPIC,
            default_offset="earliest",  # an operator inspecting the DLQ wants the accumulated backlog
        )
        out: list[dict[str, Any]] = []
        seen = 0
        while seen < max_messages:
            msg = consumer.poll(0.5)
            if msg is None:
                break
            seen += 1
            if msg.error():
                continue
            try:
                out.append(json.loads(msg.value()))
            except (TypeError, ValueError, UnicodeDecodeError):
                raw_bytes = _kafka_wire_bytes(msg.value())
                out.append(
                    {
                        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                        "raw_bytes": len(raw_bytes),
                    }
                )
            consumer.commit(message=msg, asynchronous=False)
        return out

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "depth": len(self._pending_receipts),
            "active_consumers": len(self._consumers),
            "max_consumers": self.max_consumers,
        }


def _kafka_wire_bytes(value: Any) -> bytes:
    """Return only the byte representation admitted by Kafka's wire API.

    Poison records are referenced by digest and length; arbitrary object
    ``repr`` output is never persisted because it can contain credentials,
    local paths, or other environment-specific details.
    """
    if isinstance(value, bytes | bytearray | memoryview):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return b""


# ══════════════════════════════════════════════════════════════════════════
# Resolution — required engine or Kafka log
# ══════════════════════════════════════════════════════════════════════════
def resolve_bus_log_backend(*, engine: Any = None, config: Any = None) -> BusLogBackend:
    """Resolve the required durable bus log; fail closed when unavailable."""
    if config is None:
        from agent_utilities.core.config import config as _cfg

        config = _cfg

    raw = (
        str(getattr(config, "agent_bus_log_backend", "engine") or "engine")
        .strip()
        .lower()
    )
    if raw not in BUS_LOG_BACKENDS:
        raise ValueError(
            f"AGENT_BUS_LOG_BACKEND={raw!r} is not one of {BUS_LOG_BACKENDS}"
        )
    partitions = int(getattr(config, "agent_bus_partitions", 6) or 6)
    delivery_lease_ms = (
        int(getattr(config, "agent_bus_delivery_lease_seconds", 300) or 300) * 1000
    )
    if raw == "engine":
        # A broker already present on the bound engine object wins outright — no separate
        # MCP-tool client connection needed (also the direct test seam: inject a fake engine
        # with a ``.broker`` attribute rather than monkeypatching ``engine_tools._client_for``).
        if engine is not None and getattr(engine, "broker", None) is not None:
            return EngineBrokerBusLog(
                engine,
                partitions=partitions,
                delivery_lease_ms=delivery_lease_ms,
            )
        backend = _try_engine_broker(
            fail_loud=True,
            partitions=partitions,
            delivery_lease_ms=delivery_lease_ms,
        )
        if backend is not None:
            return backend
        if raw == "engine":
            raise BusLogUnavailable(
                "AGENT_BUS_LOG_BACKEND=engine is explicitly selected but the "
                "connected engine client has no broker surface (or the engine "
                "is unreachable). Fix GRAPH_SERVICE_ENDPOINTS or select Kafka."
            )

    return KafkaBusLog(
        bootstrap_servers=getattr(config, "kafka_bootstrap_servers", None),
        partitions=partitions,
        fail_loud=True,
        max_consumers=int(getattr(config, "agent_bus_max_consumers", 32) or 32),
    )


def _try_engine_broker(
    *, fail_loud: bool, partitions: int, delivery_lease_ms: int
) -> EngineBrokerBusLog | None:
    try:
        from agent_utilities.mcp.tools import engine_tools

        client = engine_tools._client_for("")
    except Exception as exc:  # noqa: BLE001 — engine unreachable degrades to the next tier
        if fail_loud:
            logger.warning(
                "[AU-P1-2] engine broker unreachable (%s)", type(exc).__name__
            )
        else:
            logger.debug(
                "[AU-P1-2] engine broker unreachable in auto mode (%s)",
                type(exc).__name__,
            )
        return None
    if getattr(client, "broker", None) is None:
        logger.debug("[AU-P1-2] connected engine client has no 'broker' surface")
        return None
    return EngineBrokerBusLog(
        client, partitions=partitions, delivery_lease_ms=delivery_lease_ms
    )
