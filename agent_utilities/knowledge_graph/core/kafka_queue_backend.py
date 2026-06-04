# CONCEPT:ECO-4.05 - Pluggable Event Queue Backend
# CONCEPT:ORCH-1.10 - Reactive Event Sourcing

import json
import logging
import os
from typing import Any

from .queue_backend import QueueBackend

logger = logging.getLogger(__name__)


class KafkaQueueBackend(QueueBackend):
    """Kafka-backed task queue with automatic local SQLite fallback."""

    def __init__(self, fallback_db_path: str, bootstrap_servers: str | None = None):
        self.fallback_db_path = fallback_db_path
        self.bootstrap_servers = bootstrap_servers or os.environ.get(
            "AGENT_UTILITIES_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self._fallback_queue: Any = None
        self._producer: Any = None
        self._consumer: Any = None

        # Check if we can connect to Kafka
        try:
            from kafka import KafkaProducer

            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=2000,
            )
            logger.info("Successfully connected to Kafka producer backend.")
        except Exception as e:
            logger.warning(
                "Kafka connection failed or 'kafka-python' not installed. Falling back to local SQLite: %s",
                e,
            )
            self._use_fallback()

    def _use_fallback(self):
        from .engine_tasks import SQLiteTaskQueue

        self._fallback_queue = SQLiteTaskQueue(self.fallback_db_path)
        logger.info(
            "Kafka Queue fell back to SQLiteTaskQueue at %s", self.fallback_db_path
        )

    def put(self, item: dict[str, Any]) -> None:
        if self._fallback_queue or self._producer is None:
            if not self._fallback_queue:
                self._use_fallback()
            assert self._fallback_queue is not None
            return self._fallback_queue.put(item)

        try:
            assert self._producer is not None
            self._producer.send("kg_tasks", value=item)
            self._producer.flush(timeout=2.0)
        except Exception as e:
            logger.error("Kafka put failed, executing SQLite fallback write: %s", e)
            if not self._fallback_queue:
                self._use_fallback()
            assert self._fallback_queue is not None
            self._fallback_queue.put(item)

    def get(self) -> tuple[Any, dict[str, Any]] | None:
        if self._fallback_queue:
            return self._fallback_queue.get()

        try:
            from kafka import KafkaConsumer

            if not self._consumer:
                self._consumer = KafkaConsumer(
                    "kg_tasks",
                    bootstrap_servers=self.bootstrap_servers,
                    group_id="kg_task_group",
                    auto_offset_reset="earliest",
                    enable_auto_commit=False,
                    consumer_timeout_ms=1000,
                )

            assert self._consumer is not None
            # Fetch a message
            records = self._consumer.poll(timeout_ms=100)
            for tp, messages in records.items():
                if messages:
                    msg = messages[0]
                    # Retain the partition/offset context to commit/ack later
                    return (tp, msg.offset), json.loads(msg.value.decode("utf-8"))
            return None
        except Exception as e:
            logger.debug("Kafka get failed or timed out: %s", e)
            return None

    def ack(self, item_id: Any) -> None:
        if self._fallback_queue or self._consumer is None:
            if not self._fallback_queue:
                self._use_fallback()
            assert self._fallback_queue is not None
            return self._fallback_queue.ack(item_id)

        try:
            from kafka import OffsetAndMetadata

            tp, offset = item_id
            assert self._consumer is not None
            self._consumer.commit({tp: OffsetAndMetadata(offset + 1, None)})
        except Exception as e:
            logger.error("Kafka commit/ack failed: %s", e)

    def get_queue_size(self) -> int:
        if self._fallback_queue:
            return self._fallback_queue.get_queue_size()
        return 0  # Dynamic sizes in Kafka require querying metadata offsets

    def put_staged_graph(self, job_id: str, nodes: list, edges: list) -> None:
        if self._fallback_queue or self._producer is None:
            if not self._fallback_queue:
                self._use_fallback()
            assert self._fallback_queue is not None
            return self._fallback_queue.put_staged_graph(job_id, nodes, edges)

        try:
            assert self._producer is not None
            payload = {"job_id": job_id, "nodes": nodes, "edges": edges}
            self._producer.send("kg_staging", value=payload)
            self._producer.flush(timeout=2.0)
        except Exception as e:
            logger.error("Kafka put_staged_graph failed, falling back: %s", e)
            if not self._fallback_queue:
                self._use_fallback()
            assert self._fallback_queue is not None
            self._fallback_queue.put_staged_graph(job_id, nodes, edges)

    def get_staged_graph(self) -> tuple[Any, str, dict[str, Any]] | None:
        if self._fallback_queue:
            return self._fallback_queue.get_staged_graph()

        try:
            from kafka import KafkaConsumer

            # We use a separate consumer or group for staging graphs
            consumer = KafkaConsumer(
                "kg_staging",
                bootstrap_servers=self.bootstrap_servers,
                group_id="kg_staging_group",
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                consumer_timeout_ms=1000,
            )
            records = consumer.poll(timeout_ms=100)
            for tp, messages in records.items():
                if messages:
                    msg = messages[0]
                    payload = json.loads(msg.value.decode("utf-8"))
                    return (
                        (tp, msg.offset),
                        payload["job_id"],
                        {"nodes": payload["nodes"], "edges": payload["edges"]},
                    )
            return None
        except Exception as e:
            logger.debug("Kafka get_staged_graph failed: %s", e)
            return None

    def ack_staged_graph(self, item_id: Any) -> None:
        if self._fallback_queue:
            return self._fallback_queue.ack_staged_graph(item_id)
        # We can commit the staged queue offset
        pass
