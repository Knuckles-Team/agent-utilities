# CONCEPT:AU-ECO.bus.pluggable-event-queue - Pluggable Event Queue Backend
# CONCEPT:AU-ORCH.reactive.event-sourcing-ledger - Reactive Event Sourcing
# CONCEPT:AU-KG.backend.selectable-queue-backend - Fail-loud selectable ingest task-queue backend with explicit kafka and postgres and auto sqlite modes

import collections
import inspect
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Values accepted by ``TASK_QUEUE_BACKEND``.
TASK_QUEUE_BACKENDS = ("sqlite", "postgres", "kafka")


class TaskQueueUnavailable(RuntimeError):
    """An EXPLICITLY selected task-queue backend is unreachable at startup.

    CONCEPT:AU-KG.backend.selectable-queue-backend — when an operator pins ``TASK_QUEUE_BACKEND=kafka`` (or
    ``postgres``) the queue is a hard contract: silently degrading to the
    per-host SQLite file would split the fleet's queue into invisible islands.
    The message always names the endpoint that failed and how to fall back.
    """


def resolve_task_queue_backend(config: Any) -> tuple[str, bool]:
    """Resolve the ingest task-queue choice as ``(backend, explicit)``.

    Resolution order (CONCEPT:AU-KG.backend.selectable-queue-backend):

    1. ``task_queue_backend`` (``TASK_QUEUE_BACKEND``) — explicit, fail-loud.
    2. Auto: ``postgres`` when ``state_db_uri`` is set (durable state
       externalized, CONCEPT:AU-OS.state.unified-durable-state-externalization/KG-2.54), else ``sqlite`` (zero-infra).
    """
    raw = getattr(config, "task_queue_backend", None)
    if raw:
        choice = str(raw).strip().lower()
        if choice not in TASK_QUEUE_BACKENDS:
            raise ValueError(
                f"TASK_QUEUE_BACKEND={choice!r} is not one of {TASK_QUEUE_BACKENDS}"
            )
        return choice, True

    if getattr(config, "state_db_uri", None):
        return "postgres", False
    return "sqlite", False


def create_task_queue(config: Any, fallback_db_path: str) -> tuple["QueueBackend", str]:
    """Build the selected ingest task queue as ``(queue, backend_name)``.

    CONCEPT:AU-KG.backend.selectable-queue-backend — the ONE construction path for the durable ingest queue
    (engine startup and the ``--stage-to-queue`` CLI both use it):

    * explicit ``kafka``/``postgres`` → unreachable broker/state-store raises
      :class:`TaskQueueUnavailable` at startup with the endpoint and the
      fall-back instructions — never a silent SQLite degrade;
    * auto / deprecated-alias modes keep the graceful per-host SQLite fallback
      (zero-infra default preserved).
    """
    choice, explicit = resolve_task_queue_backend(config)

    if choice == "kafka":
        from .kafka_queue_backend import KafkaQueueBackend

        return (
            KafkaQueueBackend(
                fallback_db_path=None if explicit else fallback_db_path,
                bootstrap_servers=getattr(config, "kafka_bootstrap_servers", None),
                fail_loud=explicit,
                partitions=int(getattr(config, "kg_tasks_partitions", 6) or 6),
            ),
            "kafka",
        )

    if choice == "postgres":
        try:
            from .postgres_queue_backend import PostgresTaskQueue

            return PostgresTaskQueue(), "postgres"
        except Exception as e:  # noqa: BLE001 — explicit ⇒ fail loud, auto ⇒ degrade
            if explicit:
                raise TaskQueueUnavailable(
                    "TASK_QUEUE_BACKEND=postgres is explicitly selected but the "
                    f"state-store Postgres ({getattr(config, 'state_db_uri', None)!r}) "
                    f"is unavailable: {e}. Fix STATE_DB_URI / the database, or "
                    "unset TASK_QUEUE_BACKEND (auto) / set it to 'sqlite' to "
                    "fall back to the per-host queue."
                ) from e
            logger.warning(
                "STATE_DB_URI set but the Postgres task queue is unavailable "
                "(%s) — falling back to the per-host SQLite queue.",
                e,
            )

    from .engine_tasks import SQLiteTaskQueue

    return SQLiteTaskQueue(fallback_db_path), "sqlite"


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
