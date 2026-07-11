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

Three backends, resolved by :func:`resolve_bus_log_backend` in preference
order (CONCEPT:AU-ECO.bus.log-backend-resolution):

1. **engine** — the epistemic-graph engine's NATIVE AMQP-style message broker
   (the same surface the ``graph_broker`` MCP tool exposes:
   ``declare_exchange``/``declare_queue``/``bind``/``publish``/``consume``/
   ``stats``, reached through the same ``SyncEpistemicGraphClient`` every
   other engine-surface tool uses). This is the best fit for AgentBus's
   point-to-point + pub/sub shape: a direct exchange + one durable queue per
   recipient gives native per-recipient delivery with NO client-side
   filtering, and the broker owns fan-out internally — never application-level
   per-recipient writes.
2. **kafka** — the existing ``confluent_kafka`` stack (mirrors
   ``kafka_queue_backend.py``'s keyed-partition conventions) when the engine
   broker is not configured/reachable. Two keyed topics
   (``agent_bus_direct``/``agent_bus_topic``), tenant-qualified partition
   keys, one dedicated consumer per subscriber tracking its own committed
   offset (the one Kafka trade-off: a subscriber's consumer reads the whole
   keyed topic and filters client-side to its own recipient/topic — bounded by
   traffic volume, never by registered-agent count, so it is NOT the O(agents)
   fan-out this workstream removes).
3. **graph** (``None`` from the resolver) — the ORIGINAL :BusMessage graph
   model, kept as the zero-infra dev fallback exactly as it worked before this
   workstream. ``AgentBus`` runs its unchanged graph-node code path whenever
   the resolver returns ``None``.

CONCEPT:AU-ECO.bus.partitioned-log-delivery — durable partitioned log as the AgentBus delivery/wakeup plane
CONCEPT:AU-ECO.bus.log-backend-resolution — engine-broker → kafka → graph-fallback resolution order
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

BUS_LOG_BACKENDS = ("engine", "kafka", "graph")

#: Kafka topics (mirrors ``kafka_queue_backend.py``'s naming conventions).
DIRECT_TOPIC = "agent_bus_direct"
TOPIC_TOPIC = "agent_bus_topic"
DLQ_TOPIC = "agent_bus_dlq"

_PROBE_TIMEOUT_S = 5.0
_DEFAULT_BOOTSTRAP = "localhost:9092"

#: Default topic-replay window for a late subscriber's ``replay_recent`` seed,
#: mirrored from ``messaging/bus.py``'s ``TOPIC_REPLAY_RECENT_S`` so both the
#: log-backed and graph-fallback paths agree on "how far back is 'recent'".
DEFAULT_REPLAY_RECENT_S = 3_600.0


class BusLogUnavailable(RuntimeError):
    """An EXPLICITLY selected bus-log backend (``engine``/``kafka``) is unreachable.

    Mirrors :class:`~agent_utilities.knowledge_graph.core.queue_backend.TaskQueueUnavailable`:
    an operator-pinned ``AGENT_BUS_LOG_BACKEND`` is a hard contract, never a
    silent degrade to the graph fallback.
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
    return f"{tenant or 'default'}:{target}"


# ── envelope shape (matches AgentBus._shape_message so callers never change) ──
def encode_envelope(
    *,
    group: str,
    sender: str,
    recipient: str,
    topic: str,
    payload: str,
    meta_json: str,
    created: float,
) -> dict[str, Any]:
    return {
        "id": f"busmsg:{group}:{recipient or topic}:{uuid.uuid4().hex[:8]}",
        "msg_group": group,
        "sender": sender,
        "recipient": recipient,
        "topic": topic,
        "payload": payload,
        "meta": meta_json,
        "status": "sent",
        "created": created,
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
        return obj
    except (TypeError, ValueError, UnicodeDecodeError):
        return None


class BusLogBackend:
    """Interface every bus-log backend implements. Never instantiated directly."""

    name: str = "abstract"

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
        raise NotImplementedError  # ABSTRACT-OK

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
        raise NotImplementedError  # ABSTRACT-OK

    def bind_subscriber(
        self, *, tenant: str, agent_id: str, topic: str, from_ts: float | None = None
    ) -> None:
        raise NotImplementedError  # ABSTRACT-OK

    def unbind_subscriber(self, *, tenant: str, agent_id: str, topic: str) -> None:
        raise NotImplementedError  # ABSTRACT-OK

    def receive(
        self,
        *,
        tenant: str,
        agent_id: str,
        topics: list[str],
        max_messages: int = 200,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError  # ABSTRACT-OK

    def read_dlq(self, *, tenant: str, max_messages: int = 50) -> list[dict[str, Any]]:
        raise NotImplementedError  # ABSTRACT-OK

    def stats(self) -> dict[str, Any]:
        raise NotImplementedError  # ABSTRACT-OK


# ══════════════════════════════════════════════════════════════════════════
# Engine-native broker (AMQP-style exchanges/queues) — CONCEPT:AU-KG.coordination.engine-message-broker
# ══════════════════════════════════════════════════════════════════════════
class EngineBrokerBusLog(BusLogBackend):
    """AgentBus delivery over the epistemic-graph engine's native message broker.

    Maps AgentBus's two delivery shapes onto AMQP-native primitives so the
    BROKER owns fan-out, not application code:

    * **direct** (one recipient) — a ``direct``-type exchange per tenant
      (``bus.direct.<tenant>``) with one durable queue per recipient
      (``bus.inbox.<tenant>.<agent_id>``) bound on routing_key=``agent_id``.
      ``publish(routing_key=to)`` lands in exactly one queue — no fan-out
      write, no filtering on read.
    * **topic** (N subscribers) — a ``fanout``-type exchange per
      (tenant, topic) (``bus.topic.<tenant>.<topic>``) with one durable queue
      per subscriber (``bus.subq.<tenant>.<agent_id>.<topic>``) bound to it.
      ONE ``publish`` call reaches every bound queue — the broker fans out
      internally, never the KG.

    Declarations are idempotent and cached per-process so a hot ``send``/
    ``receive`` does not re-declare infrastructure on every call.
    """

    name = "engine"

    def __init__(self, client: Any) -> None:
        self._client = client
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

    # ── direct (per-recipient queue) ──
    def _direct_exchange(self, tenant: str) -> str:
        exch = f"bus.direct.{tenant}"
        self._declare_once(
            f"exch:{exch}",
            self._broker().declare_exchange,
            exchange=exch,
            exchange_type="direct",
        )
        return exch

    def _direct_inbox(self, tenant: str, agent_id: str) -> tuple[str, str]:
        exch = self._direct_exchange(tenant)
        queue = f"bus.inbox.{tenant}.{agent_id}"
        self._declare_once(
            f"queue:{queue}", self._broker().declare_queue, queue=queue, durable=True
        )
        self._declare_once(
            f"bind:{queue}:{exch}:{agent_id}",
            self._broker().bind,
            queue=queue,
            exchange=exch,
            routing_key=agent_id,
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
        exch, _ = self._direct_inbox(tenant, to)
        envelope = encode_envelope(
            group=group,
            sender=sender,
            recipient=to,
            topic="",
            payload=payload,
            meta_json=meta_json,
            created=created,
        )
        return self._publish(exch, to, envelope)

    # ── topic (fanout to per-subscriber queues) ──
    def _topic_exchange(self, tenant: str, topic: str) -> str:
        exch = f"bus.topic.{tenant}.{topic}"
        self._declare_once(
            f"exch:{exch}",
            self._broker().declare_exchange,
            exchange=exch,
            exchange_type="fanout",
        )
        return exch

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
        exch = self._topic_exchange(tenant, topic)
        envelope = encode_envelope(
            group=group,
            sender=sender,
            recipient="",
            topic=topic,
            payload=payload,
            meta_json=meta_json,
            created=created,
        )
        return self._publish(exch, topic, envelope)

    def bind_subscriber(
        self, *, tenant: str, agent_id: str, topic: str, from_ts: float | None = None
    ) -> None:
        exch = self._topic_exchange(tenant, topic)
        queue = f"bus.subq.{tenant}.{agent_id}.{topic}"
        self._declare_once(
            f"queue:{queue}", self._broker().declare_queue, queue=queue, durable=True
        )
        self._declare_once(
            f"bind:{queue}:{exch}",
            self._broker().bind,
            queue=queue,
            exchange=exch,
            routing_key=topic,
        )

    def unbind_subscriber(self, *, tenant: str, agent_id: str, topic: str) -> None:
        # Best-effort: leave the queue declared (idempotent re-subscribe keeps
        # any backlog); nothing further to do without a broker unbind action.
        return None

    # ── consume ──
    def _publish(
        self, exchange: str, routing_key: str, envelope: dict[str, Any]
    ) -> bool:
        try:
            self._broker().publish(
                exchange=exchange,
                routing_key=routing_key,
                payload=json.dumps(envelope, default=str),
            )
            return True
        except Exception as exc:  # noqa: BLE001 — publish failure is data, not a crash
            logger.warning(
                "[AU-P1-2] engine broker publish(exchange=%s) failed: %s",
                exchange,
                exc,
            )
            return False

    def _drain(
        self, queue: str, max_messages: int, *, dlq_tenant: str
    ) -> list[dict[str, Any]]:
        try:
            result = self._broker().consume(
                queue=queue, max_messages=max_messages, ack=True
            )
        except Exception as exc:  # noqa: BLE001 — a missing/unreachable broker degrades to empty
            logger.debug("[AU-P1-2] engine broker consume(%s) failed: %s", queue, exc)
            return []
        raw_messages = _coerce_message_list(result)
        shaped: list[dict[str, Any]] = []
        for raw in raw_messages:
            env = decode_envelope(raw)
            if env is None:
                self._to_dlq(dlq_tenant, queue, raw, "decode_error")
                continue
            shaped.append(env)
        return shaped

    def receive(
        self,
        *,
        tenant: str,
        agent_id: str,
        topics: list[str],
        max_messages: int = 200,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        _, inbox = self._direct_inbox(tenant, agent_id)
        out.extend(self._drain(inbox, max_messages, dlq_tenant=tenant))
        for topic in topics:
            queue = f"bus.subq.{tenant}.{agent_id}.{topic}"
            # A late subscriber may not have bound yet on this process — bind
            # lazily so ``receive`` is safe to call without a prior explicit
            # ``bind_subscriber`` (the graph fallback has the same laxity).
            self.bind_subscriber(tenant=tenant, agent_id=agent_id, topic=topic)
            out.extend(self._drain(queue, max_messages, dlq_tenant=tenant))
        # Never hand an agent its own topic broadcast back (mirrors the
        # original graph model's ``sender != agent_id`` backlog filter).
        out = [m for m in out if m.get("sender") != agent_id or not m.get("topic")]
        out.sort(key=lambda m: float(m.get("created", 0) or 0))
        return out

    def _dlq_queue(self, tenant: str) -> str:
        queue = f"bus.dlq.{tenant}"
        self._declare_once(
            f"queue:{queue}", self._broker().declare_queue, queue=queue, durable=True
        )
        return queue

    def _to_dlq(self, tenant: str, source_queue: str, raw: Any, error: str) -> None:
        dlq = self._dlq_queue(tenant)
        payload = json.dumps(
            {
                "source_queue": source_queue,
                "raw": _safe_repr(raw),
                "error": error,
                "ts": time.time(),
            },
            default=str,
        )
        try:
            self._broker().publish(exchange="", routing_key=dlq, payload=payload)
        except Exception as exc:  # noqa: BLE001 — DLQ is best-effort, never blocks delivery
            logger.warning("[AU-P1-2] engine broker DLQ publish failed: %s", exc)

    def read_dlq(self, *, tenant: str, max_messages: int = 50) -> list[dict[str, Any]]:
        dlq = self._dlq_queue(tenant)
        try:
            result = self._broker().consume(
                queue=dlq, max_messages=max_messages, ack=True
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[AU-P1-2] engine broker DLQ read failed: %s", exc)
            return []
        out = []
        for raw in _coerce_message_list(result):
            try:
                out.append(raw if isinstance(raw, dict) else json.loads(raw))
            except (TypeError, ValueError):
                out.append({"raw": _safe_repr(raw)})
        return out

    def stats(self) -> dict[str, Any]:
        try:
            return {"backend": self.name, **(self._broker().stats() or {})}
        except Exception as exc:  # noqa: BLE001
            return {"backend": self.name, "error": str(exc)}


def _coerce_message_list(result: Any) -> list[Any]:
    """Normalize a broker ``consume`` result to a flat list of raw messages.

    The engine build's exact return shape is not pinned down here (this tool
    degrades cleanly like every other ``engine_surface_tools`` wrapper), so
    accept a bare list, or a dict with a ``messages``/``items`` key.
    """
    if result is None:
        return []
    if isinstance(result, list | tuple):
        return list(result)
    if isinstance(result, dict):
        for key in ("messages", "items", "results"):
            val = result.get(key)
            if isinstance(val, list):
                return val
    return []


def _safe_repr(raw: Any) -> str:
    if isinstance(raw, bytes | bytearray):
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return repr(raw)
    return str(raw)


# ══════════════════════════════════════════════════════════════════════════
# Kafka fallback — CONCEPT:AU-KG.backend.keyed-ingest-partitions (reused convention)
# ══════════════════════════════════════════════════════════════════════════
class KafkaBusLog(BusLogBackend):
    """AgentBus delivery over Kafka when the engine broker is not configured/reachable.

    Two keyed topics carry the two delivery shapes (mirrors
    ``kafka_queue_backend.py``'s idempotent grow-only topic provisioning):

    * ``agent_bus_direct`` — direct sends, keyed ``tenant:recipient``.
    * ``agent_bus_topic``  — topic sends, keyed ``tenant:topic``.

    Every subscriber (a recipient's inbox, or one agent's subscription to one
    topic) gets its OWN Kafka consumer group so it tracks its own committed
    offset — no shared cursor, no graph node. The one Kafka trade-off vs the
    engine's native per-recipient queues: a subscriber's consumer is assigned
    every partition of the shared keyed topic (Kafka has no native
    routing-key-to-queue binding), so it reads — and discards — traffic
    addressed to other recipients/tenants on the same topic. That cost scales
    with total bus TRAFFIC, never with the number of registered agents, so it
    is not the O(agents) graph fan-out this workstream removes; a follow-up
    could shard onto per-recipient topics if traffic volume warrants it.

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
    ) -> None:
        self.bootstrap_servers = bootstrap_servers or _DEFAULT_BOOTSTRAP
        self.partitions = max(1, int(partitions))
        self.fail_loud = fail_loud
        self._producer: Any = producer
        self._admin: Any = admin_client
        self._consumer_factory = consumer_factory
        self._consumers: dict[tuple[Any, ...], Any] = {}
        self._lock = threading.Lock()
        try:
            self._ensure_topics()
        except Exception as exc:  # noqa: BLE001 — always typed so the caller can degrade
            # This class has no internal SQLite-style fallback (unlike
            # ``KafkaQueueBackend``) — that fallback IS the graph backend, one layer up
            # in ``resolve_bus_log_backend``. So always raise the SAME typed exception
            # here; ``fail_loud`` only changes whether the CALLER treats it as a hard
            # contract (explicit selection) or catches it and falls through to the next
            # tier (auto mode).
            raise BusLogUnavailable(
                "the Kafka bus-log backend is unavailable: "
                f"broker {self.bootstrap_servers!r} could not be reached/provisioned "
                f"({exc}). Start the Kafka stack and check KAFKA_BOOTSTRAP_SERVERS, or "
                "set AGENT_BUS_LOG_BACKEND=graph."
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
                    logger.warning("create_partitions(%s) failed: %s", topic, e)

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
                value=json.dumps(envelope, default=str).encode("utf-8"),
                key=key.encode("utf-8"),
            )
            self._producer_client().flush(5.0)
            return True
        except Exception as exc:  # noqa: BLE001 — publish failure is data, not a crash
            logger.warning("[AU-P1-2] kafka bus publish(%s) failed: %s", topic, exc)
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
            group=group,
            sender=sender,
            recipient="",
            topic=topic,
            payload=payload,
            meta_json=meta_json,
            created=created,
        )
        return self._publish(TOPIC_TOPIC, key, envelope)

    # ── per-subscriber consumer group (own committed offset, not a graph cursor) ──
    def _consumer(
        self,
        cache_key: tuple[Any, ...],
        *,
        group_id: str,
        topic: str,
        seed_ts: float | None,
        default_offset: str = "latest",
    ) -> Any:
        """Get-or-create the cached consumer for ``cache_key``.

        ``default_offset`` governs a BRAND-NEW consumer group's starting position when
        ``seed_ts`` is not given: ``"earliest"`` for the direct inbox (a direct message is
        durable regardless of when the recipient first registered — matches the original
        graph model), ``"latest"`` for a topic subscription (a fresh subscriber does NOT get
        the whole history dumped on it by default — matches ``AgentBus.subscribe``'s
        no-history-dump default). ``seed_ts``, when given, always wins via an explicit seek
        (the ``replay_recent`` bounded-window backfill).
        """
        with self._lock:
            existing = self._consumers.get(cache_key)
            if existing is not None:
                return existing
            if self._consumer_factory is not None:
                consumer = self._consumer_factory(
                    topic=topic,
                    group=group_id,
                    seed_ts=seed_ts,
                    default_offset=default_offset,
                )
            else:
                consumer = self._build_real_consumer(
                    group_id=group_id,
                    topic=topic,
                    seed_ts=seed_ts,
                    default_offset=default_offset,
                )
            self._consumers[cache_key] = consumer
            return consumer

    def _build_real_consumer(
        self,
        *,
        group_id: str,
        topic: str,
        seed_ts: float | None,
        default_offset: str = "latest",
    ) -> Any:
        from confluent_kafka import Consumer, TopicPartition

        consumer = Consumer(
            {
                "bootstrap.servers": self.bootstrap_servers,
                "group.id": group_id,
                "enable.auto.commit": False,
                "auto.offset.reset": default_offset,
            }
        )
        if seed_ts is not None:
            # Explicit timestamp seek (CONCEPT:AU-ECO.bus.store-and-forward-log): a late topic
            # subscriber replays only messages newer than ``seed_ts`` — the
            # log-backed equivalent of the graph model's per-(agent,topic)
            # cursor baseline. Bypasses group-coordinated ``subscribe`` (a
            # rebalance callback) in favor of a direct ``assign`` since this
            # group always has exactly one member.
            md = self._admin_client().list_topics(topic=topic, timeout=_PROBE_TIMEOUT_S)
            topic_md = getattr(md, "topics", {}).get(topic)
            parts = (
                list(getattr(topic_md, "partitions", {}).keys())
                if topic_md is not None
                else list(range(self.partitions))
            )
            ts_ms = int(seed_ts * 1000)
            query = [TopicPartition(topic, p, ts_ms) for p in parts]
            resolved = consumer.offsets_for_times(query, timeout=_PROBE_TIMEOUT_S)
            consumer.assign(resolved)
        else:
            consumer.subscribe([topic])
        return consumer

    def bind_subscriber(
        self, *, tenant: str, agent_id: str, topic: str, from_ts: float | None = None
    ) -> None:
        cache_key = ("topic", tenant, agent_id, topic)
        if cache_key in self._consumers:
            return
        group_id = f"agentbus-topic-{tenant}-{agent_id}-{topic}"
        self._consumer(
            cache_key,
            group_id=group_id,
            topic=TOPIC_TOPIC,
            seed_ts=from_ts,
            default_offset="latest",  # no history dump by default (CONCEPT:AU-ECO.bus.store-and-forward-log)
        )

    def unbind_subscriber(self, *, tenant: str, agent_id: str, topic: str) -> None:
        cache_key = ("topic", tenant, agent_id, topic)
        with self._lock:
            consumer = self._consumers.pop(cache_key, None)
        if consumer is not None:
            close = getattr(consumer, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()

    def _drain(
        self,
        consumer: Any,
        *,
        max_messages: int,
        want_key: tuple[str, str] | None,
        dlq_tenant: str,
        source_topic: str,
    ) -> list[dict[str, Any]]:
        """Poll up to ``max_messages`` records, decode + DLQ poison, filter, commit.

        ``want_key`` is ``(field, value)`` the envelope must match (e.g.
        ``("recipient", agent_id)`` or ``("topic", topic)``); records that
        don't match are still consumed + committed (this consumer group is
        dedicated to one subscriber) but not returned to the caller.
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
            consumer.commit(message=msg, asynchronous=False)
            if want_key is not None:
                field, value = want_key
                if env.get(field) != value:
                    continue
            out.append(env)
        return out

    def receive(
        self,
        *,
        tenant: str,
        agent_id: str,
        topics: list[str],
        max_messages: int = 200,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        direct_key = ("direct", tenant, agent_id)
        direct_consumer = self._consumer(
            direct_key,
            group_id=f"agentbus-direct-{tenant}-{agent_id}",
            topic=DIRECT_TOPIC,
            seed_ts=None,
            default_offset="earliest",  # direct inbox: durable regardless of when the agent registered
        )
        out.extend(
            self._drain(
                direct_consumer,
                max_messages=max_messages,
                want_key=("recipient", agent_id),
                dlq_tenant=tenant,
                source_topic=DIRECT_TOPIC,
            )
        )
        for topic in topics:
            self.bind_subscriber(tenant=tenant, agent_id=agent_id, topic=topic)
            consumer = self._consumers[("topic", tenant, agent_id, topic)]
            out.extend(
                self._drain(
                    consumer,
                    max_messages=max_messages,
                    want_key=("topic", topic),
                    dlq_tenant=tenant,
                    source_topic=TOPIC_TOPIC,
                )
            )
        out = [m for m in out if m.get("sender") != agent_id or not m.get("topic")]
        out.sort(key=lambda m: float(m.get("created", 0) or 0))
        return out

    def _to_dlq(self, tenant: str, source_topic: str, raw: Any, error: str) -> None:
        payload = {
            "source_topic": source_topic,
            "raw": _safe_repr(raw),
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
            seed_ts=None,
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
            except (TypeError, ValueError):
                out.append({"raw": _safe_repr(msg.value())})
            consumer.commit(message=msg, asynchronous=False)
        return out

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "bootstrap_servers": self.bootstrap_servers,
            "active_consumers": len(self._consumers),
        }


# ══════════════════════════════════════════════════════════════════════════
# Resolution — engine → kafka → graph (CONCEPT:AU-ECO.bus.log-backend-resolution)
# ══════════════════════════════════════════════════════════════════════════
def resolve_bus_log_backend(
    *, engine: Any = None, config: Any = None
) -> BusLogBackend | None:
    """Pick the AgentBus delivery/wakeup backend, or ``None`` for the graph fallback.

    ``None`` means "run the original :BusMessage graph-node code path" — the
    dev/zero-infra default when nothing is configured, matching every other
    selectable-backend module in this codebase (``TASK_QUEUE_BACKEND``,
    ``AGENT_DISPATCH_BACKEND``): auto-mode never attempts a real network
    connection unless an operator signal says one is configured, so unit tests
    and zero-infra deployments never pay a connect-timeout cost.
    """
    if config is None:
        from agent_utilities.core.config import config as _cfg

        config = _cfg

    raw = str(getattr(config, "agent_bus_log_backend", None) or "").strip().lower()
    if raw and raw not in BUS_LOG_BACKENDS:
        raise ValueError(
            f"AGENT_BUS_LOG_BACKEND={raw!r} is not one of {BUS_LOG_BACKENDS}"
        )
    explicit = bool(raw)

    if raw == "graph":
        return None

    if raw == "engine" or (not explicit and getattr(config, "engine_endpoint", None)):
        # A broker already present on the bound engine object wins outright — no separate
        # MCP-tool client connection needed (also the direct test seam: inject a fake engine
        # with a ``.broker`` attribute rather than monkeypatching ``engine_tools._client_for``).
        if engine is not None and getattr(engine, "broker", None) is not None:
            return EngineBrokerBusLog(engine)
        backend = _try_engine_broker(fail_loud=(raw == "engine"))
        if backend is not None:
            return backend
        if raw == "engine":
            raise BusLogUnavailable(
                "AGENT_BUS_LOG_BACKEND=engine is explicitly selected but the "
                "connected engine client has no broker surface (or the engine "
                "is unreachable). Fix ENGINE_ENDPOINT, or unset "
                "AGENT_BUS_LOG_BACKEND (auto) / set it to 'kafka' or 'graph'."
            )
        # auto: engine signaled but unreachable/no-broker → fall through to kafka/graph

    if raw == "kafka" or (
        not explicit
        and (
            str(getattr(config, "task_queue_backend", "") or "").lower() == "kafka"
            or getattr(config, "kafka_bootstrap_servers", None)
        )
    ):
        try:
            return KafkaBusLog(
                bootstrap_servers=getattr(config, "kafka_bootstrap_servers", None),
                partitions=int(getattr(config, "agent_bus_partitions", 6) or 6),
                fail_loud=(raw == "kafka"),
            )
        except BusLogUnavailable:
            if raw == "kafka":
                raise
            logger.debug(
                "[AU-P1-2] kafka bus-log unavailable (auto) — falling back to graph"
            )
            return None

    return None


def _try_engine_broker(*, fail_loud: bool) -> EngineBrokerBusLog | None:
    try:
        from agent_utilities.mcp.tools import engine_tools

        client = engine_tools._client_for("")
    except Exception as exc:  # noqa: BLE001 — engine unreachable degrades to the next tier
        if fail_loud:
            logger.warning("[AU-P1-2] engine broker unreachable: %s", exc)
        else:
            logger.debug("[AU-P1-2] engine broker unreachable (auto): %s", exc)
        return None
    if getattr(client, "broker", None) is None:
        logger.debug("[AU-P1-2] connected engine client has no 'broker' surface")
        return None
    return EngineBrokerBusLog(client)
