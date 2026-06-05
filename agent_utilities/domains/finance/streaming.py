"""
Universal Real-Time Streaming Infrastructure — CONCEPT:KG-2.6

Provides a protocol-based, domain-agnostic WebSocket streaming framework
that can be leveraged for financial market data, system telemetry,
agent event buses, and any future real-time use case.

Sources: FinceptTerminal WebSocket Architecture (AGPL-inspired design, independent impl)
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StreamMessage:
    """A single message in a real-time stream."""

    topic: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    source: str = ""
    sequence: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    def to_json(self) -> str:
        return json.dumps(
            {
                "topic": self.topic,
                "data": self.data,
                "timestamp": self.timestamp,
                "source": self.source,
                "sequence": self.sequence,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> "StreamMessage":
        d = json.loads(raw)
        return cls(**d)


class StreamSubscriber(ABC):
    """Abstract base class for stream subscribers."""

    @abstractmethod
    async def on_message(self, message: StreamMessage) -> None:
        """Handle an incoming stream message."""

    @abstractmethod
    async def on_error(self, error: Exception) -> None:
        """Handle a stream error."""

    @abstractmethod
    async def on_disconnect(self) -> None:
        """Handle stream disconnection."""


class CallbackSubscriber(StreamSubscriber):
    """Subscriber that delegates to callback functions."""

    def __init__(
        self,
        on_message_fn: Callable | None = None,
        on_error_fn: Callable | None = None,
        on_disconnect_fn: Callable | None = None,
    ):
        self._on_message = on_message_fn
        self._on_error = on_error_fn
        self._on_disconnect = on_disconnect_fn

    async def on_message(self, message: StreamMessage) -> None:
        if self._on_message:
            result = self._on_message(message)
            if asyncio.iscoroutine(result):
                await result

    async def on_error(self, error: Exception) -> None:
        if self._on_error:
            result = self._on_error(error)
            if asyncio.iscoroutine(result):
                await result

    async def on_disconnect(self) -> None:
        if self._on_disconnect:
            result = self._on_disconnect()
            if asyncio.iscoroutine(result):
                await result


class StreamBus:
    """
    In-process pub/sub message bus for real-time streaming.

    This is the core routing layer — domain-specific adapters (market data,
    telemetry, agent events) publish to topics, and subscribers receive
    filtered messages. Designed to be extended with WebSocket transport.

    Usage:
        bus = StreamBus()
        bus.subscribe("market.AAPL.trades", my_subscriber)
        await bus.publish(StreamMessage(topic="market.AAPL.trades", data={...}))
    """

    def __init__(self):
        self._subscribers: dict[str, list[StreamSubscriber]] = defaultdict(list)
        self._sequence: int = 0
        self._running = False
        self._history: list[StreamMessage] = []
        self._max_history: int = 1000

    def subscribe(self, topic: str, subscriber: StreamSubscriber) -> None:
        """Subscribe to a topic pattern. Supports exact match and wildcard '*'."""
        self._subscribers[topic].append(subscriber)
        logger.debug(f"Subscribed to topic: {topic}")

    def unsubscribe(self, topic: str, subscriber: StreamSubscriber) -> None:
        """Remove a subscriber from a topic."""
        if topic in self._subscribers:
            self._subscribers[topic] = [
                s for s in self._subscribers[topic] if s is not subscriber
            ]

    async def publish(self, message: StreamMessage) -> int:
        """
        Publish a message to all matching subscribers.

        Returns:
            Number of subscribers that received the message.
        """
        self._sequence += 1
        message.sequence = self._sequence

        # Store in history
        self._history.append(message)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        delivered = 0

        for pattern, subscribers in self._subscribers.items():
            if self._matches(pattern, message.topic):
                for subscriber in subscribers:
                    try:
                        await subscriber.on_message(message)
                        delivered += 1
                    except Exception as e:
                        logger.error(f"Subscriber error on topic {message.topic}: {e}")
                        try:
                            await subscriber.on_error(e)
                        except Exception:
                            pass  # nosec

        return delivered

    def _matches(self, pattern: str, topic: str) -> bool:
        """Check if a topic matches a subscription pattern (supports '*' wildcard)."""
        if pattern == topic:
            return True
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return topic.startswith(prefix + ".")
        return False

    @property
    def topic_count(self) -> int:
        """Number of subscribed topics."""
        return len(self._subscribers)

    @property
    def subscriber_count(self) -> int:
        """Total number of subscribers across all topics."""
        return sum(len(subs) for subs in self._subscribers.values())

    def get_history(
        self, topic: str | None = None, limit: int = 100
    ) -> list[StreamMessage]:
        """Get recent message history, optionally filtered by topic."""
        if topic:
            filtered = [m for m in self._history if self._matches(topic, m.topic)]
        else:
            filtered = self._history
        return filtered[-limit:]


class WebSocketStreamAdapter:
    """
    Adapter to bridge the StreamBus with WebSocket connections.
    Handles connection lifecycle, reconnection, and message serialization.

    This is the transport layer — it connects the in-process StreamBus
    to external WebSocket servers or clients.
    """

    def __init__(self, bus: StreamBus, url: str = ""):
        self.bus = bus
        self.url = url
        self._connected = False
        self._connection: Any = None

    async def connect(self) -> bool:
        """Connect to a WebSocket server and begin streaming."""
        try:
            import websockets
        except ImportError:
            logger.error(
                "websockets package required. Install agent-utilities[finance]"
            )
            return False

        try:
            self._connection = await websockets.connect(self.url)
            self._connected = True
            logger.info(f"WebSocket connected to {self.url}")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        if self._connection:
            await self._connection.close()
        self._connected = False
        logger.info("WebSocket disconnected")

    async def listen(self, topic_prefix: str = ""):
        """Listen for incoming WebSocket messages and publish to the bus."""
        if not self._connection:
            return

        try:
            async for raw_message in self._connection:
                try:
                    msg = StreamMessage.from_json(raw_message)
                    if topic_prefix:
                        msg.topic = f"{topic_prefix}.{msg.topic}"
                    await self.bus.publish(msg)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {raw_message[:100]}")
        except Exception as e:
            logger.error(f"WebSocket listen error: {e}")
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected
