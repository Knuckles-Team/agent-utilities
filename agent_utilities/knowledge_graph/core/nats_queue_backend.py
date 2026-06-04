# CONCEPT:ECO-4.05 - Pluggable Event Queue Backend
# CONCEPT:ORCH-1.10 - Reactive Event Sourcing

import asyncio
import json
import logging
import os
import threading
from typing import Any

from .queue_backend import QueueBackend

try:
    import nats
except ImportError:
    nats = None  # type: ignore

logger = logging.getLogger(__name__)


class NatsQueueBackend(QueueBackend):
    """NATS JetStream-backed task queue with automatic local SQLite fallback."""

    def __init__(self, fallback_db_path: str, nats_url: str | None = None):
        self.fallback_db_path = fallback_db_path
        self.nats_url = nats_url or os.environ.get(
            "AGENT_UTILITIES_NATS_URL", "nats://localhost:4222"
        )
        self._fallback_queue: Any = None
        self._nats_client: Any = None
        self._js: Any = None
        self._loop = None

        # Check if we can connect to NATS
        try:
            self._loop = asyncio.new_event_loop()
            self._run_sync(self._connect_nats())
            logger.info("Successfully connected to NATS JetStream queue backend.")
        except Exception as e:
            logger.warning(
                "NATS connection failed or 'nats-py' not installed. Falling back to local SQLite: %s",
                e,
            )
            self._use_fallback()

    def _run_sync(self, coro):
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None and running_loop.is_running():
            result = None
            exception = None

            def target():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    result = new_loop.run_until_complete(coro)
                except Exception as e:
                    exception = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join()
            if exception:
                raise exception
            return result
        else:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
            return self._loop.run_until_complete(coro)

    async def _connect_nats(self):
        if nats is None:
            raise ImportError("nats-py is not installed")
        self._nats_client = await nats.connect(self.nats_url, connect_timeout=2.0)
        self._js = self._nats_client.jetstream()

        # Ensure streams exist
        await self._js.add_stream(name="kg_tasks", subjects=["kg.tasks.*"])
        await self._js.add_stream(name="kg_staging", subjects=["kg.staging.*"])

    def _use_fallback(self):
        from .engine_tasks import SQLiteTaskQueue

        self._fallback_queue = SQLiteTaskQueue(self.fallback_db_path)
        logger.info(
            "NATS Queue fell back to SQLiteTaskQueue at %s", self.fallback_db_path
        )

    def put(self, item: dict[str, Any]) -> None:
        if self._fallback_queue or self._js is None:
            if not self._fallback_queue:
                self._use_fallback()
            assert self._fallback_queue is not None
            return self._fallback_queue.put(item)

        try:
            payload = json.dumps(item).encode("utf-8")
            self._run_sync(self._js.publish("kg.tasks.submit", payload))
        except Exception as e:
            logger.error("NATS put failed, executing SQLite fallback write: %s", e)
            if not self._fallback_queue:
                self._use_fallback()
            assert self._fallback_queue is not None
            self._fallback_queue.put(item)

    def get(self) -> tuple[Any, dict[str, Any]] | None:
        if self._fallback_queue or self._js is None:
            if not self._fallback_queue:
                self._use_fallback()
            assert self._fallback_queue is not None
            return self._fallback_queue.get()

        try:

            async def _fetch():
                assert self._js is not None
                sub = await self._js.pull_subscribe(
                    "kg.tasks.*", "kg-task-pull-durable"
                )
                msgs = await sub.fetch(1, timeout=1.0)
                if msgs:
                    msg = msgs[0]
                    return msg, json.loads(msg.data.decode("utf-8"))
                return None

            return self._run_sync(_fetch())
        except Exception as e:
            logger.debug("NATS get timed out or failed: %s", e)
            return None

    def ack(self, item_id: Any) -> None:
        if self._fallback_queue:
            return self._fallback_queue.ack(item_id)

        try:

            async def _ack():
                await item_id.ack()

            self._run_sync(_ack())
        except Exception as e:
            logger.error("NATS ack failed: %s", e)

    def get_queue_size(self) -> int:
        if self._fallback_queue or self._js is None:
            if not self._fallback_queue:
                self._use_fallback()
            assert self._fallback_queue is not None
            return self._fallback_queue.get_queue_size()

        try:

            async def _get_size():
                assert self._js is not None
                info = await self._js.stream_info("kg_tasks")
                return info.state.messages

            return self._run_sync(_get_size())
        except Exception:
            return 0

    def put_staged_graph(self, job_id: str, nodes: list, edges: list) -> None:
        if self._fallback_queue or self._js is None:
            if not self._fallback_queue:
                self._use_fallback()
            assert self._fallback_queue is not None
            return self._fallback_queue.put_staged_graph(job_id, nodes, edges)

        try:
            payload = json.dumps({"job_id": job_id, "nodes": nodes, "edges": edges})
            self._run_sync(
                self._js.publish(f"kg.staging.{job_id}", payload.encode("utf-8"))
            )
        except Exception as e:
            logger.error("NATS put_staged_graph failed, falling back: %s", e)
            if not self._fallback_queue:
                self._use_fallback()
            assert self._fallback_queue is not None
            self._fallback_queue.put_staged_graph(job_id, nodes, edges)

    def get_staged_graph(self) -> tuple[Any, str, dict[str, Any]] | None:
        if self._fallback_queue or self._js is None:
            if not self._fallback_queue:
                self._use_fallback()
            assert self._fallback_queue is not None
            return self._fallback_queue.get_staged_graph()

        try:

            async def _fetch_staged():
                assert self._js is not None
                sub = await self._js.pull_subscribe(
                    "kg.staging.*", "kg-staging-pull-durable"
                )
                msgs = await sub.fetch(1, timeout=1.0)
                if msgs:
                    msg = msgs[0]
                    payload = json.loads(msg.data.decode("utf-8"))
                    return (
                        msg,
                        payload["job_id"],
                        {"nodes": payload["nodes"], "edges": payload["edges"]},
                    )
                return None

            return self._run_sync(_fetch_staged())
        except Exception as e:
            logger.debug("NATS get_staged_graph timed out or failed: %s", e)
            return None

    def ack_staged_graph(self, item_id: Any) -> None:
        if self._fallback_queue:
            return self._fallback_queue.ack_staged_graph(item_id)
        self.ack(item_id)
