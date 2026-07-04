#!/usr/bin/python
from __future__ import annotations

"""Abstracted Event Backbone.

CONCEPT:AU-ECO.bus.pluggable-event-queue — Pluggable Event Queue Backend
CONCEPT:AU-ORCH.reactive.event-sourcing-ledger — Reactive Event Sourcing
CONCEPT:AU-KG.query.vendor-agnostic-traversal — Vendor-Agnostic Event Backbone

Provides a protocol-based event backbone abstraction with:
    - **MemoryEventBackend** (default): Zero-dependency, in-process pub/sub
      using asyncio queues. Ideal for testing, dev, and single-process agents.
    - **RedpandaEventBackend** (optional): Distributed event backbone using
      Kafka (KRaft mode). Required for multi-agent, multi-process deployments.

Standard topic taxonomy:
    - ``kg.mutations``  — Graph CRUD events (node/edge add/update/delete)
    - ``kg.tasks``      — Task queue events (schedule, complete, fail)
    - ``kg.staging``    — Staged graph payloads awaiting write
    - ``kg.telemetry``  — Agent execution traces, latency, error rates
    - ``kg.evolution``  — Self-improvement triggers, AHE cycle events

Usage::

    from agent_utilities.knowledge_graph.core.event_backend import (
        create_event_backend,
    )

    # In-memory (default, zero config)
    backend = create_event_backend("memory")

    # Kafka (production)
    backend = create_event_backend("redpanda", bootstrap_servers="redpanda:9092")

    await backend.start()
    await backend.subscribe("kg.mutations", "my-group", my_handler)
    await backend.publish("kg.mutations", {"action": "add_node", "id": "A"})
    await backend.stop()

Environment Variables:
    EVENT_BACKEND: Backend type ("memory", "redpanda"). Default: "redpanda".
    REDPANDA_BROKERS: Kafka broker addresses. Default: "localhost:9092".
    REDPANDA_CONSUMER_GROUP: Default consumer group. Default: "agent-utilities".
"""

import asyncio
import collections
import inspect
import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

# Standard topic taxonomy
TOPIC_MUTATIONS = "kg.mutations"
TOPIC_TASKS = "kg.tasks"
TOPIC_STAGING = "kg.staging"
TOPIC_TELEMETRY = "kg.telemetry"
TOPIC_EVOLUTION = "kg.evolution"

ALL_TOPICS = [
    TOPIC_MUTATIONS,
    TOPIC_TASKS,
    TOPIC_STAGING,
    TOPIC_TELEMETRY,
    TOPIC_EVOLUTION,
]

# Type alias for event handlers
EventHandler = Callable[[str, dict[str, Any]], Any]


@runtime_checkable
class EventBackend(Protocol):
    """Vendor-agnostic event backbone protocol.

    All event backends must implement these methods. The protocol
    supports both sync and async handlers — async is preferred for
    production use.
    """

    async def start(self) -> None:
        """Initialize connections and begin consuming."""
        raise RuntimeError("Protocol method called directly")

    async def stop(self) -> None:
        """Gracefully shut down, flush pending events, close connections."""
        raise RuntimeError("Protocol method called directly")

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Publish an event to a topic.

        Args:
            topic: Topic name (e.g. "kg.mutations").
            event: Event payload (must be JSON-serializable).
        """
        raise RuntimeError("Protocol method called directly")

    async def subscribe(
        self,
        topic: str,
        group: str,
        handler: EventHandler,
    ) -> None:
        """Subscribe to a topic with a consumer group.

        Args:
            topic: Topic pattern (supports wildcards: ``kg.*``).
            group: Consumer group name for load balancing.
            handler: Callback ``(topic, event_dict) -> None``.
                     May be sync or async.
        """
        raise RuntimeError("Protocol method called directly")

    def get_stats(self) -> dict[str, Any]:
        """Return backend statistics (published, consumed, lag, errors)."""
        raise RuntimeError("Protocol method called directly")


class MemoryEventBackend:
    """In-memory event backbone using asyncio queues.

    Zero dependencies. All events are dispatched within the same
    process — no persistence, no network. Suitable for:
        - Unit/integration testing
        - Single-process agents
        - Development / prototyping

    Supports wildcard topic subscriptions (e.g. ``kg.*`` matches
    ``kg.mutations``, ``kg.tasks``, etc.).
    """

    def __init__(self, max_queue_size: int = 10_000) -> None:
        self._subscriptions: dict[str, list[tuple[str, EventHandler]]] = (
            collections.defaultdict(list)
        )
        self._max_queue_size = max_queue_size
        self._running = False
        self._published = 0
        self._consumed = 0
        self._errors = 0
        self._event_log: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=1000
        )

    async def start(self) -> None:
        """Mark the backend as running."""
        self._running = True
        logger.info("MemoryEventBackend started (max_queue=%d)", self._max_queue_size)

    async def stop(self) -> None:
        """Mark the backend as stopped."""
        self._running = False
        logger.info(
            "MemoryEventBackend stopped (published=%d, consumed=%d, errors=%d)",
            self._published,
            self._consumed,
            self._errors,
        )

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Publish an event and immediately dispatch to matching subscribers.

        Events are dispatched synchronously within the publish call.
        """
        if not self._running:
            logger.warning("MemoryEventBackend: publish called while stopped")

        self._published += 1
        self._event_log.append(
            {"topic": topic, "event": event, "timestamp": time.time()}
        )

        # Find all matching subscribers
        for pattern, handlers in self._subscriptions.items():
            if self._topic_matches(pattern, topic):
                for group, handler in handlers:
                    try:
                        if inspect.iscoroutinefunction(handler):
                            await handler(topic, event)
                        else:
                            handler(topic, event)
                        self._consumed += 1
                    except Exception as e:
                        self._errors += 1
                        logger.error(
                            "MemoryEventBackend: handler error on topic=%s group=%s: %s",
                            topic,
                            group,
                            e,
                        )

    async def subscribe(
        self,
        topic: str,
        group: str,
        handler: EventHandler,
    ) -> None:
        """Subscribe a handler to a topic pattern.

        Args:
            topic: Topic name or wildcard pattern (``kg.*``).
            group: Consumer group name.
            handler: Event handler callback.
        """
        self._subscriptions[topic].append((group, handler))
        logger.debug(
            "MemoryEventBackend: subscribed group=%s to topic=%s", group, topic
        )

    def get_stats(self) -> dict[str, Any]:
        """Return event backbone statistics."""
        return {
            "backend": "memory",
            "running": self._running,
            "published": self._published,
            "consumed": self._consumed,
            "errors": self._errors,
            "subscriptions": sum(len(v) for v in self._subscriptions.values()),
            "topics": list(self._subscriptions.keys()),
        }

    def get_recent_events(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the last N events from the event log."""
        return list(self._event_log)[-n:]

    @staticmethod
    def _topic_matches(pattern: str, topic: str) -> bool:
        """Check if a topic matches a subscription pattern.

        Supports:
            - Exact match: ``kg.mutations``
            - Single-level wildcard: ``kg.*`` matches ``kg.mutations``
            - Multi-level wildcard: ``kg.>`` matches ``kg.mutations.nodes``
        """
        if pattern == topic:
            return True
        # Convert pattern to regex
        regex_pattern = (
            pattern.replace(".", r"\.").replace("*", r"[^.]+").replace(">", r".+")
        )
        return bool(re.match(f"^{regex_pattern}$", topic))


class RedpandaEventBackend:
    """Redpanda event backbone for distributed agents.

    Uses ``confluent-kafka`` for high-throughput, persistent event
    streaming. Requires a running Kafka cluster.

    Features:
        - Topic-per-concern: ``kg.mutations``, ``kg.tasks``, etc.
        - Consumer groups for load balancing across agent instances
        - Automatic topic creation (if broker allows)
        - JSON serialization with schema evolution support
        - Circuit breaker: falls back to MemoryEventBackend on failure

    Environment Variables:
        REDPANDA_BROKERS: Comma-separated broker addresses.
        REDPANDA_CONSUMER_GROUP: Default consumer group name.
        REDPANDA_SECURITY_PROTOCOL: Security protocol (PLAINTEXT, SSL, SASL_SSL).
    """

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        consumer_group: str | None = None,
        security_protocol: str | None = None,
    ) -> None:
        self._bootstrap_servers = (
            bootstrap_servers or setting("REDPANDA_BROKERS") or "localhost:9092"
        )
        self._consumer_group = (
            consumer_group or setting("REDPANDA_CONSUMER_GROUP") or "agent-utilities"
        )
        self._security_protocol = (
            security_protocol or setting("REDPANDA_SECURITY_PROTOCOL") or "PLAINTEXT"
        )

        self._producer: Any = None
        self._consumers: dict[str, Any] = {}
        self._handlers: dict[str, list[tuple[str, EventHandler]]] = (
            collections.defaultdict(list)
        )
        self._running = False
        self._consumer_task: asyncio.Task | None = None
        self._published = 0
        self._consumed = 0
        self._errors = 0

        # Fallback for when Kafka is unavailable
        self._fallback: MemoryEventBackend | None = None

    async def start(self) -> None:
        """Initialize Kafka producer and start consumer polling."""
        try:
            from confluent_kafka import Producer

            self._producer = Producer(
                {
                    "bootstrap.servers": self._bootstrap_servers,
                    "security.protocol": self._security_protocol,
                    "linger.ms": 10,
                    "batch.num.messages": 100,
                    "queue.buffering.max.messages": 100_000,
                }
            )
            self._running = True
            logger.info(
                "RedpandaEventBackend started: servers=%s, group=%s",
                self._bootstrap_servers,
                self._consumer_group,
            )
        except ImportError:
            logger.warning(
                "confluent-kafka not installed. Falling back to MemoryEventBackend. "
                "Install with: pip install 'agent-utilities[event-kafka]'"
            )
            self._fallback = MemoryEventBackend()
            await self._fallback.start()
            self._running = True
        except Exception as e:
            logger.warning(
                "Redpanda connection failed: %s. Falling back to MemoryEventBackend.", e
            )
            self._fallback = MemoryEventBackend()
            await self._fallback.start()
            self._running = True

    async def stop(self) -> None:
        """Flush pending events and shut down."""
        self._running = False

        if self._fallback:
            await self._fallback.stop()
            return

        if self._producer:
            try:
                self._producer.flush(timeout=5.0)
            except Exception as e:
                logger.warning("Redpanda flush failed: %s", e)

        for consumer in self._consumers.values():
            try:
                consumer.close()
            except Exception:
                return None

        if self._consumer_task:
            self._consumer_task.cancel()

        logger.info(
            "RedpandaEventBackend stopped (published=%d, consumed=%d, errors=%d)",
            self._published,
            self._consumed,
            self._errors,
        )

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Publish an event to a Kafka topic.

        Events are serialized as JSON. The producer uses batching
        for throughput optimization (10ms linger, 100 msg batch).
        """
        if self._fallback:
            await self._fallback.publish(topic, event)
            self._published += 1
            return

        if not self._producer:
            logger.warning("RedpandaEventBackend: producer not initialized")
            return

        try:
            payload = json.dumps(event, default=str).encode("utf-8")
            # Redpanda Partitioning: Extract partition_key for causal ordering
            partition_key = event.get("partition_key")
            key_bytes = str(partition_key).encode("utf-8") if partition_key else None

            self._producer.produce(
                topic,
                key=key_bytes,
                value=payload,
                callback=self._delivery_callback,
            )
            # Trigger any pending callbacks
            self._producer.poll(0)
            self._published += 1
        except Exception as e:
            self._errors += 1
            logger.error("Redpanda publish failed on topic=%s: %s", topic, e)

    async def subscribe(
        self,
        topic: str,
        group: str,
        handler: EventHandler,
    ) -> None:
        """Subscribe to a Kafka topic with a consumer group.

        Creates a dedicated consumer for each unique (topic, group) pair.
        Starts a background polling loop if not already running.
        """
        if self._fallback:
            await self._fallback.subscribe(topic, group, handler)
            return

        self._handlers[topic].append((group, handler))

        # Create consumer for this topic+group if needed
        consumer_key = f"{topic}:{group}"
        if consumer_key not in self._consumers:
            try:
                from confluent_kafka import Consumer

                consumer = Consumer(
                    {
                        "bootstrap.servers": self._bootstrap_servers,
                        "group.id": group,
                        "auto.offset.reset": "latest",
                        "security.protocol": self._security_protocol,
                    }
                )
                consumer.subscribe([topic])
                self._consumers[consumer_key] = consumer
                logger.info(
                    "RedpandaEventBackend: consumer created for topic=%s, group=%s",
                    topic,
                    group,
                )
            except Exception as e:
                logger.error("Failed to create Redpanda consumer: %s", e)

        # Start background consumer loop if needed
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(self._consumer_loop())

    def get_stats(self) -> dict[str, Any]:
        """Return Kafka event backbone statistics."""
        if self._fallback:
            stats = self._fallback.get_stats()
            stats["backend"] = "kafka-fallback-memory"
            return stats

        return {
            "backend": "redpanda",
            "running": self._running,
            "bootstrap_servers": self._bootstrap_servers,
            "consumer_group": self._consumer_group,
            "published": self._published,
            "consumed": self._consumed,
            "errors": self._errors,
            "active_consumers": len(self._consumers),
            "topics": list(self._handlers.keys()),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _consumer_loop(self) -> None:
        """Background polling loop for all active consumers."""
        while self._running:
            for consumer_key, consumer in list(self._consumers.items()):
                try:
                    msg = consumer.poll(0.01)
                    if msg is None:
                        continue
                    if msg.error():
                        self._errors += 1
                        logger.warning("Redpanda consumer error: %s", msg.error())
                        continue

                    topic = msg.topic()
                    try:
                        event = json.loads(msg.value().decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        event = {"raw": msg.value()}

                    # Dispatch to matching handlers
                    for handler_topic, handlers in self._handlers.items():
                        if handler_topic == topic:
                            for group, handler in handlers:
                                try:
                                    if inspect.iscoroutinefunction(handler):
                                        await handler(topic, event)
                                    else:
                                        handler(topic, event)
                                    self._consumed += 1
                                except Exception as e:
                                    self._errors += 1
                                    logger.error(
                                        "Handler error on topic=%s: %s", topic, e
                                    )

                except Exception as e:
                    self._errors += 1
                    logger.error("Consumer loop error: %s", e)

            # Yield control to event loop
            await asyncio.sleep(0.01)

    def _delivery_callback(self, err: Any, msg: Any) -> None:
        """Producer delivery callback for monitoring."""
        if err:
            self._errors += 1
            logger.warning("Redpanda delivery failed: %s", err)


# ------------------------------------------------------------------
# Factory and Singleton
# ------------------------------------------------------------------

_GLOBAL_EVENT_BACKEND: MemoryEventBackend | RedpandaEventBackend | None = None


def get_event_backend(**kwargs: Any) -> MemoryEventBackend | RedpandaEventBackend:
    """Get the global EventBackend instance, creating it if necessary."""
    global _GLOBAL_EVENT_BACKEND
    if _GLOBAL_EVENT_BACKEND is None:
        _GLOBAL_EVENT_BACKEND = create_event_backend(**kwargs)
        # Ideally, start() is called by the application lifecycle, but for safety in dev:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_GLOBAL_EVENT_BACKEND.start())
        except RuntimeError:
            pass
    return _GLOBAL_EVENT_BACKEND


def create_event_backend(
    backend_type: str | None = None,
    **kwargs: Any,
) -> MemoryEventBackend | RedpandaEventBackend:
    """Factory function to create an event backbone backend.

    Args:
        backend_type: One of "memory", "redpanda". Falls back to
            ``EVENT_BACKEND`` env var, then "memory".
        **kwargs: Backend-specific configuration.

    Returns:
        A configured EventBackend instance.
    """
    kafka_enabled = setting("KAFKA_ENABLED", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    backend_type = (
        (backend_type or setting("EVENT_BACKEND") or "redpanda").lower().strip()
    )

    if backend_type == "memory" or not kafka_enabled:
        max_queue = kwargs.get("max_queue_size", 10_000)
        return MemoryEventBackend(max_queue_size=max_queue)

    elif backend_type == "redpanda":
        return RedpandaEventBackend(
            bootstrap_servers=kwargs.get("bootstrap_servers"),
            consumer_group=kwargs.get("consumer_group"),
            security_protocol=kwargs.get("security_protocol"),
        )

    else:
        logger.warning(
            "Unknown event backend type: '%s'. Falling back to memory.", backend_type
        )
        return MemoryEventBackend()
