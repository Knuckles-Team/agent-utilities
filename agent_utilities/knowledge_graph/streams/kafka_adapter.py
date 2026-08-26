#!/usr/bin/python
from __future__ import annotations

"""Real Kafka/Redpanda stream adapter (CONCEPT:AU-KG.research.research-pipeline-runner).

Implements ``BaseStreamAdapter`` over ``aiokafka`` (optional dependency). A
consumer may be injected for tests so the adapter is exercisable offline. Maps
each Kafka record into a normalized event dict consumed by
``EventStreamIngester``.
"""

import json
import logging
import re
import time
from typing import Any

from ..core.company_brain import BaseStreamAdapter, StreamBatch

logger = logging.getLogger(__name__)

# CA-21 (CONCEPT:AU-KG.ingest.debezium-changeenvelope) — env feature flag gating the
# Debezium consumer entry point below. Default unset/off; rollback is unsetting it
# (no data migration, no schema change — see debezium_envelope.py's module docstring).
KAFKA_CDC_DEBEZIUM_ENABLED_ENV = "KAFKA_CDC_DEBEZIUM_ENABLED"


def debezium_consumer_enabled() -> bool:
    """Whether the Debezium CDC consumer entry point is enabled (env, default off).

    Routed through ``core._env.setting`` (never a bare ``os.environ`` read —
    the codebase-wide rule ``scripts/check_no_env_sprawl.py``/the env-var
    drift guard both enforce), so ``KAFKA_CDC_DEBEZIUM_ENABLED`` is a live,
    call-time read (a test's ``monkeypatch.setenv`` takes effect, matching
    CA-21-W05's "unset the flag = immediate rollback" contract).
    """
    from ...core._env import setting

    return bool(setting(KAFKA_CDC_DEBEZIUM_ENABLED_ENV, default=False))


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
            except Exception as exc:  # pragma: no cover - shutdown best-effort  # noqa: BLE001 — shutdown-path best-effort consumer stop; self._connected is set False unconditionally on the next line regardless of whether the stop() call itself succeeded
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


# Debezium's real topic-naming convention for this deployment: "<topic.prefix>.
# <schema>.<table>", where DEC-CA-03's post-freeze correction (2026-08-26, found by
# CA-51 "proving delivery end to end") pins topic.prefix itself to the literal
# "cdc.<db>" (NOT bare "<db>" — Debezium prepends nothing on its own). The live
# pilot connector's real topic, confirmed by consuming it directly off the broker
# (`kafka-console-consumer --topic cdc.ca51pilot.public.orders`, 2026-08-26), is
# "cdc.ca51pilot.public.orders" — four dot-separated segments, matching this pattern.
_DEBEZIUM_DEFAULT_TOPIC_PATTERN = re.compile(r"^cdc\.[^.]+\.[^.]+\.[^.]+$")


class DebeziumKafkaConsumer(KafkaStreamAdapter):
    """Debezium-shaped consumer entry point (CONCEPT:AU-KG.ingest.debezium-changeenvelope, CA-21).

    Subscribes by topic PATTERN (default: Debezium's real ``<prefix>.<schema>.
    <table>`` naming — see :data:`_DEBEZIUM_DEFAULT_TOPIC_PATTERN`, overridable
    via ``config.topic_pattern``) rather than one fixed topic name, decodes
    each record's Debezium change-event payload, and drains it through
    ``debezium_envelope.map_debezium_event``/``envelope_ingest.ingest_envelope``
    — reusing :class:`KafkaStreamAdapter`'s injectable-consumer test seam
    rather than a second aiokafka wrapper.

    **Fail-closed offset semantics.** Auto-commit is OFF
    (``enable_auto_commit=False``, unlike the base adapter): a consumer that
    cannot process an envelope must not advance its offset as though it had.
    :meth:`drain_once` commits, per partition, only up through the last
    record that was durably handled — a successful/skipped
    ``ingest_envelope`` commit, or an intentional quarantine skip (no engine
    call, but a deliberate, logged non-write, not a stall) — and STOPS at
    the first ``failed``/``rejected`` result, leaving that record and
    everything after it in the batch uncommitted so a crash/restart
    re-delivers from exactly there. This never advances past an unverified
    write.
    """

    def __init__(
        self,
        config: Any,
        consumer: Any = None,
        *,
        topic_pattern: re.Pattern[str] | None = None,
    ) -> None:
        super().__init__(config, consumer=consumer)
        self._topic_pattern = (
            topic_pattern
            or getattr(config, "topic_pattern", None)
            or _DEBEZIUM_DEFAULT_TOPIC_PATTERN
        )

    def _servers(self) -> str:
        configured = getattr(self.config, "endpoint", None) or getattr(
            self.config, "bootstrap_servers", None
        )
        if configured:
            return configured
        from ...core.config import config as central_config

        return central_config.kafka_bootstrap_servers or "localhost:9092"

    async def connect(self) -> None:
        if self._consumer is not None:
            self._connected = True
            return
        try:
            from aiokafka import AIOKafkaConsumer
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "Debezium consumer requires 'aiokafka'. Install agent-utilities[kafka]."
            ) from exc
        self._consumer = AIOKafkaConsumer(
            bootstrap_servers=self._servers(),
            group_id=getattr(self.config, "group_id", "au-cdc-debezium"),
            # Fail-closed (see class docstring): WE commit explicit offsets
            # in drain_once, only after a verified write — never a timer.
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        )
        await self._consumer.start()
        self._consumer.subscribe(pattern=self._topic_pattern.pattern)
        self._connected = True
        logger.info(
            "Debezium consumer connected to %s, topic pattern %r",
            self._servers(),
            self._topic_pattern.pattern,
        )

    @staticmethod
    def _decode_json(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except Exception:  # noqa: BLE001 — undecodable bytes fail closed (below), never raise mid-batch
                return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (ValueError, json.JSONDecodeError):
                return None
        return value

    async def drain_once(
        self,
        engine: Any,
        *,
        batch_size: int = 500,
        tenant: str = "",
    ) -> dict[str, Any]:
        """One bounded poll + map + commit pass. Never blocks indefinitely.

        Returns an ``EtlResult``-compatible dict (``status``/``counts``/
        ``details``) — see the module docstring for offset-commit semantics.
        """
        if not self._connected or self._consumer is None:
            raise RuntimeError("Debezium consumer not connected")

        from ..ingestion.debezium_envelope import QuarantinedRecord, map_debezium_event
        from ..ingestion.envelope_ingest import ingest_envelope

        raw = await self._consumer.getmany(
            timeout_ms=getattr(self.config, "poll_timeout_ms", 1000),
            max_records=batch_size,
        )
        partitions = raw if isinstance(raw, dict) else {}

        succeeded = skipped = quarantined = failed = 0
        stopped_early = False
        last_checkpoint: str | None = None
        failure_detail: dict[str, Any] | None = None

        for topic_partition, records in partitions.items():
            commit_offset: int | None = None
            for rec in records:
                key = self._decode_json(getattr(rec, "key", None))
                value = self._decode_json(getattr(rec, "value", rec))
                if value is None:
                    # Malformed payload — rejected, not silently skipped
                    # (W06's negative test): stop here, do not commit past it.
                    failed += 1
                    failure_detail = {
                        "status": "rejected",
                        "reason": "undecodable Debezium payload (not valid JSON)",
                    }
                    stopped_early = True
                    break

                mapped = map_debezium_event({"key": key, "value": value}, tenant=tenant)
                if isinstance(mapped, QuarantinedRecord):
                    quarantined += 1
                    logger.warning(
                        "debezium consumer: quarantined %s.%s (%s)",
                        mapped.db,
                        mapped.table,
                        mapped.reason,
                    )
                    commit_offset = getattr(rec, "offset", commit_offset)
                    continue

                result = ingest_envelope(engine, mapped)
                status = result.get("status")
                if status in ("success", "skipped"):
                    if status == "success":
                        succeeded += 1
                    else:
                        skipped += 1
                    last_checkpoint = mapped.checkpoint or last_checkpoint
                    commit_offset = getattr(rec, "offset", commit_offset)
                    continue

                # failed / rejected: STOP — never advance the offset past an
                # unverified write (the codebase-wide rule: record.consumed =
                # result.ok, never unconditional).
                failed += 1
                failure_detail = {
                    "status": status,
                    "reason": result.get("reason") or result.get("error"),
                }
                stopped_early = True
                break

            if commit_offset is not None:
                # aiokafka's commit() accepts a plain next-offset int per
                # partition (the offset of the NEXT record to read, i.e. the
                # last handled offset + 1) — no OffsetAndMetadata wrapping
                # required. Committing exactly this offset, never the
                # fetcher's own internal read-ahead position, is what keeps a
                # stop-early batch from silently over-committing past
                # buffered-but-unprocessed records.
                await self._consumer.commit({topic_partition: commit_offset + 1})
            if stopped_early:
                break

        status = "failed" if stopped_early else "ok"
        return {
            "status": status,
            "source": "cdc",
            "mode": "delta",
            "counts": {
                "succeeded": succeeded,
                "skipped": skipped,
                "quarantined": quarantined,
                "failed": failed,
            },
            "watermark": last_checkpoint,
            "details": {"failure": failure_detail} if failure_detail else {},
        }
