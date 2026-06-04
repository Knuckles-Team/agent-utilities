# CONCEPT:ECO-4.05 - Pluggable Event Queue Backend
# CONCEPT:ORCH-1.10 - Reactive Event Sourcing

import collections
import inspect
from typing import Any, Protocol


class QueueBackend(Protocol):
    """Abstract protocol representing a pluggable task execution and graph staging queue."""

    def put(self, item: dict[str, Any]) -> None:
        """Insert a task item into the queue."""
        raise RuntimeError("Protocol method called directly")

    def get(self) -> tuple[Any, dict[str, Any]] | None:
        """Fetch the next available task item from the queue, returning (item_id, payload)."""
        raise RuntimeError("Protocol method called directly")

    def ack(self, item_id: Any) -> None:
        """Acknowledge and remove a completed task item from the queue."""
        raise RuntimeError("Protocol method called directly")

    def get_queue_size(self) -> int:
        """Return the current number of pending tasks in the queue."""
        raise RuntimeError("Protocol method called directly")

    def put_staged_graph(self, job_id: str, nodes: list, edges: list) -> None:
        """Insert a serialized graph into the staging queue for writing."""
        raise RuntimeError("Protocol method called directly")

    def get_staged_graph(self) -> tuple[Any, str, dict[str, Any]] | None:
        """Fetch the next staged graph payload from the queue, returning (item_id, job_id, graph_data)."""
        raise RuntimeError("Protocol method called directly")

    def ack_staged_graph(self, item_id: Any) -> None:
        """Acknowledge and remove a processed staged graph from the queue."""
        raise RuntimeError("Protocol method called directly")

    @classmethod
    def create(cls, backend_type: str, **kwargs) -> "QueueBackend":
        """Factory method to instantiate a pluggable queue backend."""
        if backend_type == "memory":
            return MemoryQueueBackend()
        elif backend_type == "nats":
            from .nats_queue_backend import NatsQueueBackend

            fallback_db = kwargs.pop("fallback_db_path", ".tmp/nats_fallback.db")  # nosec B108
            nats_url = kwargs.pop("nats_url", kwargs.pop("servers", [None])[0])
            return NatsQueueBackend(fallback_db_path=fallback_db, nats_url=nats_url)
        elif backend_type == "kafka":
            from .kafka_queue_backend import KafkaQueueBackend

            fallback_db = kwargs.pop("fallback_db_path", ".tmp/kafka_fallback.db")  # nosec B108
            bootstrap_servers = kwargs.pop("bootstrap_servers", ["localhost:9092"])
            return KafkaQueueBackend(
                fallback_db_path=fallback_db, bootstrap_servers=bootstrap_servers
            )
        else:
            raise ValueError(f"Unknown queue backend type: {backend_type}")


class MemoryQueueBackend(QueueBackend):
    """In-memory, lightweight queue backend conforming to the QueueBackend Protocol."""

    def __init__(self):
        self._tasks: collections.deque[Any] = collections.deque()
        self._staged_graphs: collections.deque[Any] = collections.deque()
        self._counter = 0
        self._connected = False
        self._subscriptions = collections.defaultdict(list)

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def subscribe(self, topic: str, handler) -> None:
        self._subscriptions[topic].append(handler)

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        # Robust NATS-style pub-sub topic matching for local memory testing
        import re

        for sub_topic, handlers in self._subscriptions.items():
            pattern = (
                sub_topic.replace(".", r"\.").replace("*", r"[^.]+").replace(">", r".+")
            )
            if re.match(f"^{pattern}$", topic):
                for h in handlers:
                    if inspect.iscoroutinefunction(h):
                        await h(topic, payload)
                    else:
                        h(topic, payload)

    def put(self, item: dict[str, Any]) -> None:
        self._counter += 1
        self._tasks.append((self._counter, item))

    def get(self) -> tuple[Any, dict[str, Any]] | None:
        if self._tasks:
            return self._tasks[0]  # Return head without removing (waiting for ack)
        return None

    def ack(self, item_id: Any) -> None:
        # Find and remove by item_id
        for i, (uid, _) in enumerate(self._tasks):
            if uid == item_id:
                del self._tasks[i]
                break

    def get_queue_size(self) -> int:
        return len(self._tasks)

    def put_staged_graph(self, job_id: str, nodes: list, edges: list) -> None:
        self._counter += 1
        self._staged_graphs.append(
            (self._counter, job_id, {"nodes": nodes, "edges": edges})
        )

    def get_staged_graph(self) -> tuple[Any, str, dict[str, Any]] | None:
        if self._staged_graphs:
            return self._staged_graphs[0]
        return None

    def ack_staged_graph(self, item_id: Any) -> None:
        for i, (uid, _, _) in enumerate(self._staged_graphs):
            if uid == item_id:
                del self._staged_graphs[i]
                break
