#!/usr/bin/python
from __future__ import annotations

"""Real NATS JetStream stream adapter (CONCEPT:KG-2.6).

Implements ``BaseStreamAdapter`` over ``nats-py`` (optional dependency). A
subscription may be injected for tests so the adapter is exercisable offline.
"""

import json
import logging
import time
from typing import Any

from ..core.company_brain import BaseStreamAdapter, StreamBatch

logger = logging.getLogger(__name__)


class NatsStreamAdapter(BaseStreamAdapter):
    """nats-py JetStream adapter. Inject ``subscription`` to test offline."""

    def __init__(self, config: Any, subscription: Any = None) -> None:
        self.config = config
        self._sub = subscription
        self._nc: Any = None
        self._connected = subscription is not None
        self._owns_conn = subscription is None

    def _servers(self) -> str:
        return getattr(self.config, "endpoint", None) or "nats://localhost:4222"

    def _subject(self) -> str:
        return (
            getattr(self.config, "subject", None)
            or getattr(self.config, "name", None)
            or "company-brain"
        )

    async def connect(self) -> None:
        if self._sub is not None:
            self._connected = True
            return
        try:
            import nats
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "NATS adapter requires 'nats-py'. Install agent-utilities[nats]."
            ) from exc
        self._nc = await nats.connect(self._servers())
        js = self._nc.jetstream()
        self._sub = await js.pull_subscribe(
            self._subject(), durable=getattr(self.config, "group_id", "company-brain")
        )
        self._connected = True
        logger.info(
            "NATS adapter connected to %s subject %s", self._servers(), self._subject()
        )

    async def disconnect(self) -> None:
        if self._nc is not None and self._owns_conn:
            try:
                await self._nc.close()
            except Exception as exc:  # pragma: no cover - shutdown best-effort
                logger.debug("NATS close failed: %s", exc)
        self._connected = False

    @staticmethod
    def _decode(data: Any) -> dict[str, Any]:
        if isinstance(data, bytes):
            try:
                data = data.decode("utf-8")
            except Exception:  # pragma: no cover
                return {"raw": repr(data)}
        if isinstance(data, str):
            try:
                return json.loads(data)
            except (ValueError, json.JSONDecodeError):
                return {"raw": data}
        return data if isinstance(data, dict) else {"raw": str(data)}

    async def consume_batch(self, batch_size: int = 100) -> StreamBatch:
        if not self._connected or self._sub is None:
            raise RuntimeError("NATS adapter not connected")
        try:
            msgs = await self._sub.fetch(
                batch_size, timeout=getattr(self.config, "poll_timeout_s", 1)
            )
        except Exception:  # pragma: no cover - timeout = empty batch
            msgs = []
        src_type: Any = getattr(self.config, "source_type", "nats")
        events: list[dict[str, Any]] = []
        for msg in msgs:
            payload = self._decode(getattr(msg, "data", msg))
            events.append(
                {
                    "event_id": payload.get("event_id")
                    or f"nats_{len(events)}_{int(time.time() * 1000)}",
                    "source_type": src_type,
                    "event_type": payload.get("event_type", "stream_event"),
                    "tenant_id": payload.get("tenant_id", ""),
                    "payload": payload.get("payload", payload),
                    "timestamp": payload.get("timestamp", time.time()),
                }
            )
            ack = getattr(msg, "ack", None)
            if callable(ack):
                try:
                    await ack()
                except Exception as exc:  # pragma: no cover
                    logger.debug("NATS ack failed: %s", exc)
        return StreamBatch(
            stream_id=getattr(self.config, "stream_id", "nats"),
            source_type=src_type,
            events=events,
        )
