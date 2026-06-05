#!/usr/bin/python
from __future__ import annotations

"""Real Kafka/Redpanda stream adapter (CONCEPT:KG-2.6).

Implements ``BaseStreamAdapter`` over ``aiokafka`` (optional dependency). A
consumer may be injected for tests so the adapter is exercisable offline. Maps
each Kafka record into a normalized event dict consumed by
``EventStreamIngester``.
"""

import json
import logging
import time
from typing import Any

from ..core.company_brain import BaseStreamAdapter, StreamBatch

logger = logging.getLogger(__name__)


class KafkaStreamAdapter(BaseStreamAdapter):
    """aiokafka-backed adapter. Inject ``consumer`` to test without a broker."""

    def __init__(self, config: Any, consumer: Any = None) -> None:
        self.config = config
        self._consumer = consumer
        self._connected = consumer is not None
        self._owns_consumer = consumer is None

    def _servers(self) -> str:
        return (
            getattr(self.config, "endpoint", None)
            or getattr(self.config, "bootstrap_servers", None)
            or "localhost:9092"
        )

    def _topic(self) -> str:
        return (
            getattr(self.config, "topic", None)
            or getattr(self.config, "name", None)
            or "company-brain"
        )

    async def connect(self) -> None:
        if self._consumer is not None:
            self._connected = True
            return
        try:
            from aiokafka import AIOKafkaConsumer
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "Kafka adapter requires 'aiokafka'. Install agent-utilities[kafka]."
            ) from exc
        self._consumer = AIOKafkaConsumer(
            self._topic(),
            bootstrap_servers=self._servers(),
            group_id=getattr(self.config, "group_id", "company-brain"),
            enable_auto_commit=True,
            auto_offset_reset="latest",
        )
        await self._consumer.start()
        self._connected = True
        logger.info(
            "Kafka adapter connected to %s topic %s", self._servers(), self._topic()
        )

    async def disconnect(self) -> None:
        if self._consumer is not None and self._owns_consumer:
            try:
                await self._consumer.stop()
            except Exception as exc:  # pragma: no cover - shutdown best-effort
                logger.debug("Kafka stop failed: %s", exc)
        self._connected = False

    @staticmethod
    def _decode(value: Any) -> dict[str, Any]:
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except Exception:  # pragma: no cover
                return {"raw": repr(value)}
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (ValueError, json.JSONDecodeError):
                return {"raw": value}
        return value if isinstance(value, dict) else {"raw": str(value)}

    async def consume_batch(self, batch_size: int = 100) -> StreamBatch:
        if not self._connected or self._consumer is None:
            raise RuntimeError("Kafka adapter not connected")
        # aiokafka getmany returns {TopicPartition: [records]}; tolerate a fake
        # consumer that returns a flat list of records for testing.
        raw = await self._consumer.getmany(
            timeout_ms=getattr(self.config, "poll_timeout_ms", 1000),
            max_records=batch_size,
        )
        records: list[Any] = []
        if isinstance(raw, dict):
            for recs in raw.values():
                records.extend(recs)
        elif isinstance(raw, list):
            records = raw

        src_type: Any = getattr(self.config, "source_type", "kafka")
        events: list[dict[str, Any]] = []
        for rec in records[:batch_size]:
            payload = self._decode(getattr(rec, "value", rec))
            events.append(
                {
                    "event_id": payload.get("event_id")
                    or f"kafka_{getattr(rec, 'offset', len(events))}_{int(time.time() * 1000)}",
                    "source_type": src_type,
                    "event_type": payload.get("event_type", "stream_event"),
                    "tenant_id": payload.get("tenant_id", ""),
                    "payload": payload.get("payload", payload),
                    "timestamp": payload.get("timestamp", time.time()),
                }
            )
        return StreamBatch(
            stream_id=getattr(self.config, "stream_id", "kafka"),
            source_type=src_type,
            events=events,
        )
