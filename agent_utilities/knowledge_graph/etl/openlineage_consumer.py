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

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, TypedDict

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

# Env feature flag (CA-25-W06's rollback contract: unset = the consumer never
# touches Kafka at all — no other state to unwind, mirrors CA-21's
# KAFKA_CDC_DEBEZIUM_ENABLED exactly).
KAFKA_OPENLINEAGE_CONSUMER_ENABLED_ENV = "KAFKA_OPENLINEAGE_CONSUMER_ENABLED"

# DEC-CA-05's Contract table: the one topic every OpenLineage producer
# (eg, and eventually Spark/Trino) writes to. A fixed topic, not a pattern —
# unlike CA-21's Debezium consumer, there is exactly one logical stream here.
OPENLINEAGE_TOPIC = "openlineage.events"

# OpenLineage's own RunEvent.eventType enum -> this KG's prov:Activity status
# vocabulary. An eventType outside this map is quarantined, never guessed.
EVENT_TYPE_TO_STATUS: dict[str, str] = {
    "START": "running",
    "RUNNING": "running",
    "COMPLETE": "completed",
    "ABORT": "aborted",
    "FAIL": "failed",
    "OTHER": "unknown",
}


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


# DEC-CA-05's dataset-naming rule: this regex is deliberately IDENTICAL to
# egeria-mcp's own ``_ICEBERG_DATASET_RE`` (reconcile.py) — the same wire
# format, independently validated on each side of the MCP boundary rather
# than imported across it.
_ICEBERG_DATASET_RE = re.compile(
    r"^iceberg://(?P<catalog>[^/]+)/(?P<namespace>[^/]+)/(?P<table>[^/@]+)@(?P<snapshot>[^/@]+)$"
)

# OpenLineage's own top-level/nested RunEvent field names, named rather than
# repeated as literal strings — this module only ever READS them (no producer
# lives in this repo, by design: the wire format is defined upstream by
# OpenLineage/Spark/Trino/eg), so a literal-string ``.get("outputs")`` etc.
# reads as an orphaned key to a naive same-string-anywhere-in-repo scan the
# instant an unrelated module elsewhere happens to read the same short field
# name fewer than 3 times (as ``ecosystem.media.gateway`` does for
# "outputs", coincidentally, for an unrelated payload shape).
_FIELD_RUN = "run"
_FIELD_RUN_ID = "runId"
_FIELD_JOB = "job"
_FIELD_JOB_NAME = "name"
_FIELD_JOB_NAMESPACE = "namespace"
_FIELD_EVENT_TYPE = "eventType"
_FIELD_INPUTS = "inputs"
_FIELD_OUTPUTS = "outputs"
_FIELD_FACETS = "facets"
_FIELD_PARENT = "parent"
_FIELD_VERSION = "version"
_FIELD_DATASET_VERSION = "datasetVersion"


class MalformedLineageDataset(ValueError):
    """A dataset did not resolve to ``iceberg://<catalog>/<ns>/<table>@<snapshot>``.

    Raised by :func:`dataset_entity_id`, caught by :func:`map_openlineage_event`
    to quarantine the whole RunEvent — never a partially-mapped one with a
    fabricated entity id.
    """


def _as_dict(value: Any) -> dict[str, Any]:
    """Narrow ``value`` to a dict or an empty one — the one place this module
    dereferences an optional nested mapping, so mypy can narrow a plain local
    variable instead of re-checking a repeated ``.get()`` call expression
    (mirrors ``..ingestion.debezium_envelope._as_dict`` exactly)."""
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    """The list twin of :func:`_as_dict`, for ``inputs``/``outputs``."""
    return value if isinstance(value, list) else []


def dataset_entity_id(dataset: dict[str, Any]) -> str:
    """Return the ``iceberg://<catalog>/<ns>/<table>@<snapshot>`` id for one
    OpenLineage dataset object, or raise :class:`MalformedLineageDataset`.

    Expects ``namespace`` already in ``iceberg://<catalog>/<ns>`` form,
    ``name`` the bare table name, and the snapshot in the standard OpenLineage
    ``version`` dataset facet (``facets.version.datasetVersion``).
    """
    namespace = str(dataset.get("namespace") or "").strip()
    name = str(dataset.get("name") or "").strip()
    facets = _as_dict(dataset.get(_FIELD_FACETS))
    version_facet = facets.get(_FIELD_VERSION)
    snapshot = ""
    if isinstance(version_facet, dict):
        snapshot = str(version_facet.get(_FIELD_DATASET_VERSION) or "").strip()
    candidate = (
        f"{namespace.rstrip('/')}/{name}@{snapshot}" if namespace and name else ""
    )
    if not _ICEBERG_DATASET_RE.match(candidate):
        raise MalformedLineageDataset(
            f"dataset namespace={namespace!r} name={name!r} snapshot={snapshot!r} "
            "does not resolve to iceberg://<catalog>/<ns>/<table>@<snapshot>"
        )
    return candidate


@dataclass(frozen=True)
class MappedRunEvent:
    """One OpenLineage RunEvent, validated and normalized (no graph I/O)."""

    run_id: str
    job_name: str
    job_namespace: str
    event_type: str
    activity_status: str
    input_dataset_ids: tuple[str, ...] = field(default_factory=tuple)
    output_dataset_ids: tuple[str, ...] = field(default_factory=tuple)
    parent_run_id: str = ""


@dataclass(frozen=True)
class QuarantinedLineageEvent:
    """A RunEvent this module refused to auto-map — logged, never guessed.

    Mirrors :class:`~..ingestion.debezium_envelope.QuarantinedRecord`'s
    fail-closed shape for the lineage domain.
    """

    run_id: str
    job_name: str
    event_type: str
    reason: str


def _run_id(event: dict[str, Any]) -> str:
    return str(_as_dict(event.get(_FIELD_RUN)).get(_FIELD_RUN_ID) or "").strip()


def _parent_run_id(event: dict[str, Any]) -> str:
    run = _as_dict(event.get(_FIELD_RUN))
    facets = _as_dict(run.get(_FIELD_FACETS))
    parent = facets.get(_FIELD_PARENT)
    if not isinstance(parent, dict):
        return ""
    parent_run = _as_dict(parent.get(_FIELD_RUN))
    return str(parent_run.get(_FIELD_RUN_ID) or "").strip()


def _dataset_ids(event: dict[str, Any], field: str) -> tuple[str, ...]:
    """Map valid dataset objects from one event field to entity ids.

    OpenLineage payloads may contain non-mapping values; the existing mapper
    intentionally ignores those while allowing :class:`MalformedLineageDataset`
    from a mapping to quarantine the whole event in its caller.
    """
    datasets = _as_list(event.get(field))
    return tuple(
        dataset_entity_id(dataset) for dataset in datasets if isinstance(dataset, dict)
    )


def map_openlineage_event(
    event: dict[str, Any],
) -> MappedRunEvent | QuarantinedLineageEvent:
    """Validate + normalize one raw OpenLineage RunEvent dict.

    Pure and engine-independent — every failure mode a fixture can hit is
    testable here without a graph. See the module docstring for the mapping
    rule; ``inputs``/``outputs`` are validated as a whole (one malformed
    dataset quarantines the entire event rather than mapping the rest).
    """
    run_id = _run_id(event)
    job = _as_dict(event.get(_FIELD_JOB))
    job_name = str(job.get(_FIELD_JOB_NAME) or "").strip()
    job_namespace = str(job.get(_FIELD_JOB_NAMESPACE) or "").strip()
    event_type = str(event.get(_FIELD_EVENT_TYPE) or "").strip().upper()

    if not run_id:
        return QuarantinedLineageEvent(
            run_id=run_id,
            job_name=job_name,
            event_type=event_type,
            reason="missing run.runId",
        )
    if not job_name:
        return QuarantinedLineageEvent(
            run_id=run_id,
            job_name=job_name,
            event_type=event_type,
            reason="missing job.name",
        )
    status = EVENT_TYPE_TO_STATUS.get(event_type)
    if status is None:
        return QuarantinedLineageEvent(
            run_id=run_id,
            job_name=job_name,
            event_type=event_type,
            reason=f"unmapped eventType {event_type!r}",
        )

    try:
        input_ids = _dataset_ids(event, _FIELD_INPUTS)
        output_ids = _dataset_ids(event, _FIELD_OUTPUTS)
    except MalformedLineageDataset as exc:
        return QuarantinedLineageEvent(
            run_id=run_id,
            job_name=job_name,
            event_type=event_type,
            reason=str(exc),
        )

    return MappedRunEvent(
        run_id=run_id,
        job_name=job_name,
        job_namespace=job_namespace,
        event_type=event_type,
        activity_status=status,
        input_dataset_ids=input_ids,
        output_dataset_ids=output_ids,
        parent_run_id=_parent_run_id(event),
    )


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

    @staticmethod
    def _decode_json(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except Exception:  # noqa: BLE001 — undecodable bytes fail closed (below)
                return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (ValueError, json.JSONDecodeError):
                return None
        return value

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
                value = self._decode_json(getattr(rec, "value", rec))
                commit_offset = getattr(rec, "offset", commit_offset)
                if not isinstance(value, dict):
                    failed += 1
                    logger.warning(
                        "openlineage consumer: undecodable RunEvent payload at offset %s",
                        getattr(rec, "offset", "?"),
                    )
                    continue

                mapped = map_openlineage_event(value)
                if isinstance(mapped, QuarantinedLineageEvent):
                    quarantined += 1
                    logger.warning(
                        "openlineage consumer: quarantined run=%s job=%s (%s)",
                        mapped.run_id,
                        mapped.job_name,
                        mapped.reason,
                    )
                    continue

                activity_id = record_openlineage_run_event(engine, value)
                if activity_id:
                    succeeded += 1
                else:
                    failed += 1

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
