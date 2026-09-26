#!/usr/bin/python
from __future__ import annotations

"""OpenLineage RunEvent consumer -> PROV-O mapping (CA-25, DEC-CA-05).

(CONCEPT:AU-KG.ingest.openlineage-consumer)

**The gap this closes.** eg already emits OpenLineage RunEvents from its own
lake operations (``src/server/lake/lineage.rs``) and, once CA-52/CA-53
configure their listeners, Spark and Trino will too — but until this module
au has zero code that understands the OpenLineage wire format. This module
is the au-side consumer ``DEC-CA-05`` defines: a Kafka consumer on the
``openlineage.events`` topic that maps each RunEvent into the existing
PROV-O vocabulary (``knowledge_graph.etl.lineage.record_openlineage_run_event``)
and, when the run originated from a tool call, links it back into the
existing ``:RunTrace`` chain (``observability.trace_ontology``).

**Design (per ``DEC-CA-05``'s Contract table).**

* ``run.runId`` + ``job`` -> the ``:RunTrace`` correlation key (see
  :mod:`~agent_utilities.observability.trace_ontology`); this module itself
  only *extracts* the candidate id(s), the correlation lookup lives there.
* ``eventType`` (``START``/``RUNNING``/``COMPLETE``/``ABORT``/``FAIL``) ->
  ``prov:Activity`` status, via :data:`EVENT_TYPE_TO_STATUS`. An event whose
  ``eventType`` is not one OpenLineage defines is quarantined rather than
  guessed.
* ``inputs[]``/``outputs[]`` dataset objects -> ``prov:Entity`` node ids,
  via :func:`dataset_entity_id`, per the naming rule below.
* **Dataset naming (the DEC-CA-05 contract, shared with CA-30/CA-40 and
  egeria-mcp's ``parse_iceberg_dataset_name``/``reconcile_openlineage_asset``,
  see ``agents/egeria-mcp/egeria_mcp/reconcile.py``):**
  ``iceberg://<catalog>/<ns>/<table>@<snapshot>`` — this string IS the
  ``prov:Entity`` id, the Egeria ``DataAsset`` ``qualifiedName``, and (per
  CA-46's AGENTS.md "Node-id convention", offered not imposed, ADOPTED here
  because it is the one deterministic key that keeps this lane, CA-30, and
  CA-46 agreeing with no separate mapping table) the ``externalId`` half of
  a ``<domain>:<class>:<externalId>`` KG node id. A dataset's ``namespace``
  is expected in the form ``iceberg://<catalog>/<ns>`` and ``name`` the bare
  table name; the snapshot comes from OpenLineage's standard ``version``
  dataset facet (``facets.version.datasetVersion`` — the
  ``VersionDatasetFacet`` every OpenLineage-Iceberg integration is expected
  to attach). Any dataset that does not resolve to this shape is
  quarantined (:class:`MalformedLineageDataset`) — never guessed, matching
  ``DEC-CA-03``'s CDC quarantine pattern egeria-mcp's own parser mirrors
  independently (no cross-package import — ``egeria-mcp`` is CA-46's
  package; this is au's own, deliberately duplicated, copy of the same
  regex so the two systems agree on the wire format without sharing code).
* **Failure semantics:** unlike CA-21's Debezium consumer (which STOPS a
  batch on the first failure to protect offset-ordered CDC correctness),
  lineage is fire-and-forget by design (``DEC-CA-05``'s Decision) — a
  quarantined or failed RunEvent is logged and counted, never fatal to the
  batch, and the consumer always advances past it (``record_openlineage_run_event``
  writes are idempotent by construction, so redelivery after a crash is
  harmless either way).
"""

import logging
from typing import Any, TypedDict

from agent_connector_sdk.transports.stream_payload import decode_json_object_or_none
from epistemic_graph.openlineage_derivation import (
    EVENT_TYPE_TO_STATUS,
    MalformedLineageDataset,
    MappedRunEvent,
    QuarantinedLineageEvent,
    dataset_entity_id,
    map_openlineage_event,
)

from ..streams.kafka_adapter import KafkaStreamAdapter

logger = logging.getLogger(__name__)

__all__ = [
    "KAFKA_OPENLINEAGE_CONSUMER_ENABLED_ENV",
    "OPENLINEAGE_TOPIC",
    "EVENT_TYPE_TO_STATUS",
    "MalformedLineageDataset",
    "MappedRunEvent",
    "QuarantinedLineageEvent",
    "dataset_entity_id",
    "map_openlineage_event",
    "openlineage_consumer_enabled",
    "OpenLineageKafkaConsumer",
    "run_openlineage_catchup",
]

class OpenLineageDrainCounts(TypedDict):
    """Per-batch outcome tally — the typed twin of the raw dict every other
    consumer in this codebase (``DebeziumKafkaConsumer.drain_once``) still
    returns untyped; named here so this seam doesn't silently drift."""

    succeeded: int
    quarantined: int
    failed: int


class OpenLineageDrainResult(TypedDict, total=False):
    """Return shape of :meth:`OpenLineageKafkaConsumer.drain_once` and
    :func:`run_openlineage_catchup` — ``reason`` is present only on the
    ``status="skipped"`` path."""

    status: str
    source: str
    counts: OpenLineageDrainCounts
    reason: str


# Env feature flag (CA-25-W06's rollback contract: unset = the consumer never
# touches Kafka at all — no other state to unwind, mirrors CA-21's
# KAFKA_CDC_DEBEZIUM_ENABLED exactly).
KAFKA_OPENLINEAGE_CONSUMER_ENABLED_ENV = "KAFKA_OPENLINEAGE_CONSUMER_ENABLED"

# DEC-CA-05's Contract table: the one topic every OpenLineage producer
# (eg, and eventually Spark/Trino) writes to. A fixed topic, not a pattern —
# unlike CA-21's Debezium consumer, there is exactly one logical stream here.
OPENLINEAGE_TOPIC = "openlineage.events"


def openlineage_consumer_enabled() -> bool:
    """Whether the OpenLineage consumer entry point is enabled (env, default off).

    Mirrors ``kafka_adapter.debezium_consumer_enabled`` exactly — a live,
    call-time read through the one centralized accessor (never a bare
    ``os.environ`` read; ``scripts/check_no_env_sprawl.py`` enforces this).
    ``default=False`` is a ``bool``, so ``setting()`` infers ``cast=to_boolean``
    on its own — passing ``cast=bool`` here would be wrong (``bool("false")``
    is ``True``); this deliberately does NOT pass an explicit ``cast``.
    """
    from ...core._env import setting

    return bool(setting(KAFKA_OPENLINEAGE_CONSUMER_ENABLED_ENV, default=False))


class OpenLineageKafkaConsumer(KafkaStreamAdapter):
    """OpenLineage-shaped consumer entry point (CA-25, DEC-CA-05).

    Subscribes to the single fixed :data:`OPENLINEAGE_TOPIC`, decodes each
    record's RunEvent JSON payload, and drains it through
    :func:`map_openlineage_event` /
    ``knowledge_graph.etl.lineage.record_openlineage_run_event`` — reusing
    :class:`~..streams.kafka_adapter.KafkaStreamAdapter`'s injectable-consumer
    test seam rather than a second aiokafka wrapper.

    **Offset semantics differ from CA-21's Debezium consumer on purpose.**
    Lineage is fire-and-forget (``DEC-CA-05``): every record in a batch is
    attempted and its offset committed regardless of outcome
    (success/quarantine/failure) — a lost lineage event degrades
    observability, never correctness, and ``record_openlineage_run_event``'s
    writes are idempotent by construction (deterministic activity id), so
    redelivery after a crash is always safe to replay.
    """

    def __init__(self, config: Any, consumer: Any = None) -> None:
        super().__init__(config, consumer=consumer)

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
                "OpenLineage consumer requires 'aiokafka'. Install agent-utilities[kafka]."
            ) from exc
        self._consumer = AIOKafkaConsumer(
            OPENLINEAGE_TOPIC,
            bootstrap_servers=self._servers(),
            group_id=getattr(self.config, "group_id", "au-lineage-openlineage"),
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        )
        await self._consumer.start()
        self._connected = True
        logger.info(
            "OpenLineage consumer connected to %s, topic %r",
            self._servers(),
            OPENLINEAGE_TOPIC,
        )

    def _process_record(
        self, engine: Any, rec: Any, record_openlineage_run_event: Any
    ) -> tuple[int, int, int]:
        """Map and persist one record, returning success/quarantine/failure deltas."""
        value = decode_json_object_or_none(getattr(rec, "value", rec))
        if not isinstance(value, dict):
            logger.warning(
                "openlineage consumer: undecodable RunEvent payload at offset %s",
                getattr(rec, "offset", "?"),
            )
            return 0, 0, 1

        mapped = map_openlineage_event(value)
        if isinstance(mapped, QuarantinedLineageEvent):
            logger.warning(
                "openlineage consumer: quarantined run=%s job=%s (%s)",
                mapped.run_id,
                mapped.job_name,
                mapped.reason,
            )
            return 0, 1, 0

        activity_id = record_openlineage_run_event(engine, value)
        return (1, 0, 0) if activity_id else (0, 0, 1)

    async def drain_once(
        self, engine: Any, *, batch_size: int = 500
    ) -> OpenLineageDrainResult:
        """One bounded poll + map + best-effort write pass. Never blocks
        indefinitely, and never stops a batch early — see class docstring.
        """
        if not self._connected or self._consumer is None:
            raise RuntimeError("OpenLineage consumer not connected")

        from .lineage import record_openlineage_run_event

        raw = await self._consumer.getmany(
            timeout_ms=getattr(self.config, "poll_timeout_ms", 1000),
            max_records=batch_size,
        )
        partitions = raw if isinstance(raw, dict) else {}

        succeeded = quarantined = failed = 0

        for topic_partition, records in partitions.items():
            commit_offset: int | None = None
            for rec in records:
                commit_offset = getattr(rec, "offset", commit_offset)
                success_delta, quarantine_delta, failure_delta = self._process_record(
                    engine, rec, record_openlineage_run_event
                )
                succeeded += success_delta
                quarantined += quarantine_delta
                failed += failure_delta

            if commit_offset is not None:
                # Fire-and-forget (see class docstring): commit through the
                # whole batch regardless of per-record outcome.
                await self._consumer.commit({topic_partition: commit_offset + 1})

        return {
            "status": "ok",
            "source": "openlineage",
            "counts": {
                "succeeded": succeeded,
                "quarantined": quarantined,
                "failed": failed,
            },
        }


def run_openlineage_catchup(
    engine: Any, *, client: Any = None
) -> OpenLineageDrainResult:
    """A bounded, on-demand OpenLineage catch-up poll.

    The scheduler-facing entry point (CA-28's ``deploy/schedules.yml`` entry
    is out of scope here — this is the callable that entry will eventually
    invoke). Gated by :func:`openlineage_consumer_enabled`
    (``KAFKA_OPENLINEAGE_CONSUMER_ENABLED``): disabled means this function
    never touches Kafka at all, reporting ``status="skipped"`` — a real,
    immediate rollback, mirroring CA-21's ``run_cdc_catchup`` contract.
    ``client``, when given, is an already-connected
    :class:`OpenLineageKafkaConsumer` (or a test double with the same
    ``connect``/``drain_once``/``disconnect`` surface) — the same
    "explicit injection wins" convention every consumer in this codebase
    already offers.
    """
    from ...protocols.source_connectors.connectors.mcp_package import _run_async

    owns_client = client is None
    if owns_client and not openlineage_consumer_enabled():
        return {
            "status": "skipped",
            "source": "openlineage",
            "reason": "KAFKA_OPENLINEAGE_CONSUMER_ENABLED is unset — consumer disabled",
            "counts": {"succeeded": 0, "quarantined": 0, "failed": 0},
        }
    consumer = client or OpenLineageKafkaConsumer(config=None)

    async def _drain() -> OpenLineageDrainResult:
        if owns_client:
            await consumer.connect()
        try:
            return await consumer.drain_once(engine)
        finally:
            if owns_client:
                await consumer.disconnect()

    return _run_async(_drain())
