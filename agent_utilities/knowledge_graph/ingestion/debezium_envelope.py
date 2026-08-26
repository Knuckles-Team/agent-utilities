"""Debezium → ``ChangeEnvelope`` mapping + envelope-source registry.

(CONCEPT:AU-KG.ingest.debezium-changeenvelope, CA-21, DEC-CA-03)

**The gap this closes.** au has an ``aiokafka`` adapter
(:mod:`knowledge_graph.streams.kafka_adapter`) but, until this module,
zero code path that understands Debezium's ``{payload: {before, after, op,
source: {lsn, table, schema, db}}}`` change-event shape or turns one into a
:class:`~.change_envelope.ChangeEnvelope`. This module is that translator —
the au-side half of the CDC seam ``DEC-CA-03`` defines (the eg-side
producer is CA-11's; the Kafka Connect/Debezium deployment is CA-51's;
this lane never touches either).

**Design (per ``DEC-CA-03``'s mapping table).**

* Debezium ``op`` → :attr:`~.change_envelope.ChangeEnvelope.operation`:
  ``c``/``u``/``r`` → ``"upsert"``, ``d`` → ``"delete"`` (``DEC-CA-03``'s
  prose calls the delete case "tombstone" — the concrete
  :data:`~.change_envelope.Operation` literal this codebase defines has no
  ``"tombstone"`` value, only ``"upsert" | "delete" | "snapshot_complete"``
  (``change_envelope.py:93``), so ``"delete"`` is the exact binding).
* ``source.table`` resolves to a target ontology class via the source
  database's own ``connector_manifest.yml`` (:func:`_resolve_ontology_class`)
  — an unmapped table is **quarantined**, never guess-classified
  (:class:`QuarantinedRecord`), mirroring
  :meth:`~agent_utilities.protocols.source_connectors.base.ExternalAccess.quarantined`'s
  fail-closed pattern.
* Idempotency key ``(db, table, pk, lsn)`` rides
  :meth:`~.change_envelope.ChangeEnvelope.from_connector_record`'s existing
  auto-derivation (``connector``, ``tenant``, ``source_instance``,
  ``source_object_id``, ``source_version``, ``operation`` — SHA-256, see
  ``change_envelope.py:_stable_key``) — no new hash function.
* The LSN rides :attr:`~.change_envelope.ChangeEnvelope.checkpoint`, the
  existing watermark-cursor field ``envelope_ingest.py``'s
  ``_position_advances``/``_typed_position`` already enforce monotonically
  inside the same commit that writes the row — not new logic this module
  writes.

**A topic-naming premise this lane independently re-measured (report,
don't silently work around — per the W0 review note).** ``DEC-CA-03`` as
originally accepted stated the CDC topic pattern as ``cdc.<db>.<table>``
with ``topic.prefix=<db>``. Checking the live CA-51 pilot connector
directly (``kubectl -n apps exec deploy/kafka-connect -- curl
localhost:8083/connectors/ca51pilot``, 2026-08-26) found it configured
with ``"topic.prefix": "ca51pilot"`` (the bare *database* name) — which,
under Debezium's real ``<topic.prefix>.<schema>.<table>`` topic-naming
convention, produces no ``cdc.`` literal component at all, and also omits
the schema segment the original pattern didn't account for. ``DEC-CA-03``
has since taken a **post-freeze correction (2026-08-26, credited to
CA-51's own end-to-end delivery proof)**: ``topic.prefix`` must literally
be ``cdc.<db>`` for the ``cdc.<db>.<table>`` pattern to hold at all — with
the schema segment, the corrected, LIVE-VERIFIED topic is
``cdc.<db>.<schema>.<table>``. Re-consuming the live pilot topic directly
after that fix confirms it: ``cdc.ca51pilot.public.orders`` exists on the
broker and carries a real Debezium ``op: "c"`` event
(``kafka-console-consumer --topic cdc.ca51pilot.public.orders
--from-beginning``, 2026-08-26). This module does not depend on the topic
string either way (every field it needs — ``db``/``table``/``op``/``lsn``
— comes from the Debezium payload's own ``source`` object); the consumer
entry point in :mod:`~..streams.kafka_adapter` subscribes by the corrected
4-segment pattern (:data:`~..streams.kafka_adapter._DEBEZIUM_DEFAULT_TOPIC_PATTERN`),
not the uncorrected literal ``cdc.*`` glob the pre-correction design
section named.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .change_envelope import ChangeEnvelope, Operation

logger = logging.getLogger(__name__)

__all__ = [
    "DEBEZIUM_OP_MAP",
    "QuarantinedRecord",
    "map_debezium_event",
    "register_envelope_source",
    "get_envelope_source",
    "list_envelope_sources",
    "run_cdc_catchup",
]

# Debezium's ``op`` field -> ChangeEnvelope Operation (DEC-CA-03's mapping table).
# "r" (snapshot read, the initial-load replay of a pre-existing row) is an
# upsert exactly like a live "c"/"u" — there is no separate backfill path in
# this lane; Debezium's own snapshot covers initial load (see the lane doc's
# "Excluded" note).
DEBEZIUM_OP_MAP: dict[str, Operation] = {
    "c": "upsert",
    "u": "upsert",
    "r": "upsert",
    "d": "delete",
}


@dataclass(frozen=True)
class QuarantinedRecord:
    """A Debezium event this module refused to auto-classify.

    Never a guess: an unmapped table, an unmapped ``op``, or a payload
    missing the identity/table fields a safe classification needs all land
    here instead of a fabricated :class:`~.change_envelope.ChangeEnvelope`
    — the fail-closed rule ``DEC-CA-03`` and the CA-21 lane contract both
    require (``ExternalAccess.quarantined()``'s pattern, applied to
    *classification* rather than *access*).
    """

    db: str
    table: str
    reason: str
    op: str = ""
    lsn: str = ""


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _debezium_payload(section: Any) -> dict[str, Any]:
    """Unwrap Kafka Connect's optional ``{"schema": ..., "payload": ...}``
    envelope (present when ``key.converter.schemas.enable``/
    ``value.converter.schemas.enable`` are ``true``; the CA-51 pilot
    connector runs with both set to ``"false"``, so most deployments see the
    payload directly — this tolerates either).
    """
    section = _as_dict(section)
    if "payload" in section and isinstance(section.get("payload"), dict):
        return section["payload"]
    return section


def _composite_key(key_payload: dict[str, Any]) -> str:
    """A deterministic string identity over every field of the Debezium
    message key (Debezium's key schema is exactly the source table's primary
    key column(s) — one field for a simple PK, several for a composite one).
    Never fabricated: an empty key payload returns ``""`` and the caller
    quarantines rather than inventing an id.
    """
    if not key_payload:
        return ""
    return "|".join(f"{k}={key_payload[k]!s}" for k in sorted(key_payload))


def _event_time(source: dict[str, Any], payload: dict[str, Any]) -> str | None:
    """ISO-8601 UTC event time from Debezium's ``source.ts_ms`` (falls back to
    the top-level ``ts_ms`` Kafka Connect stamps) — never left to default to
    the LSN string the way ``from_connector_record``'s own generic fallback
    would (an LSN is not a timestamp).
    """
    raw = (
        source.get("ts_ms") if source.get("ts_ms") is not None else payload.get("ts_ms")
    )
    if raw is None:
        return None
    try:
        millis = int(raw)
    except (TypeError, ValueError):
        return None
    try:
        return (
            datetime.fromtimestamp(millis / 1000, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
    except (OverflowError, OSError, ValueError):
        return None


def _resolve_ontology_class(db: str, table: str) -> tuple[str | None, str]:
    """``(ontology_class, schema_version)`` for ``db``.``table`` via the source
    database's own ``connector_manifest.yml``, or ``(None, "")`` when
    unresolved (the caller quarantines rather than guessing).

    Reuses the EXISTING manifest catalog/loader
    (:func:`~..ontology.connector_manifest_gate.find_connector_manifest` +
    :class:`~..ontology.connector_manifest.ConnectorManifest`) rather than
    inventing a second one — the same mechanism the connector-manifest compile-before-sync gate already trusts. ``db`` is treated as the
    manifest's "source" key (mirrors ``source_instance=f"{db}.{table}"`` in
    this lane's design: a Debezium source database is itself one connector
    identity) and ``table`` is matched case/separator-insensitively against
    each :class:`~..ontology.connector_manifest.ResourceSpec.name` in that
    manifest's ``resources`` — the manifest schema has no separate literal
    "source table name" field distinct from the resource's own OWL-class
    local name (``connector_manifest.py``'s ``ResourceSpec.name`` docstring:
    "OWL class local name"), so a matching resource name is the closest
    existing convention to bind a raw Debezium table to.

    **Measured note:** the lane doc's citation of ``MCP_TOOL_PRESETS[
    "sql-table"]``/``mcp_tool.py:106`` for this resolution was checked and
    found unrelated — that preset bootstraps a keyset-paginated ``sql_query``
    sweep for the LOAD/POLL connector path, not a table→ontology-class
    lookup. This function goes through the manifest catalog instead (see the
    module docstring's "measured correction" section for the other
    ``DEC-CA-03`` premise this lane found stale).
    """
    from ..ontology.connector_manifest import ConnectorManifest
    from ..ontology.connector_manifest_gate import find_connector_manifest

    path = find_connector_manifest(db)
    if path is None:
        return None, ""

    import yaml

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        manifest = ConnectorManifest.model_validate(raw)
    except Exception:  # noqa: BLE001 — a broken manifest fails closed (quarantine), never crashes the consumer
        logger.warning("debezium: connector_manifest.yml for %r failed to parse", db)
        return None, ""

    target = table.strip().casefold().replace("_", "").replace("-", "")
    if not target:
        return None, ""
    for resource in manifest.resources:
        candidate = resource.name.strip().casefold().replace("_", "").replace("-", "")
        if candidate != target:
            continue
        mapping = manifest.schema_mappings.get(resource.name)
        if mapping is not None and mapping.ontology_class:
            return mapping.ontology_class, manifest.schema_version
        return None, ""
    return None, ""


def map_debezium_event(
    record: dict[str, Any],
    *,
    tenant: str = "",
) -> ChangeEnvelope | QuarantinedRecord:
    """Translate one decoded Debezium Kafka message into a
    :class:`~.change_envelope.ChangeEnvelope`, or a :class:`QuarantinedRecord`
    when it cannot be safely classified.

    ``record`` is the shape a real Kafka Connect JSON-converter delivery
    decodes to — ``{"key": <PK column(s)>, "value": <Debezium change-event
    payload>}`` (matching what
    :meth:`~..streams.kafka_adapter.DebeziumKafkaConsumer.drain_once` builds
    from a real ``aiokafka`` ``ConsumerRecord``'s own ``.key``/``.value``).
    Both ``key`` and ``value`` tolerate the optional Kafka Connect
    ``{"schema": ..., "payload": ...}`` wrapper (:func:`_debezium_payload`).
    A caller that has only the bare Debezium value payload (no key) may pass
    ``{"value": <payload>}`` — that legitimately quarantines (no id, never
    guessed) unless the row's own fields happen to be all that's needed;
    this lane does not infer a primary key from ``after``/``before`` shape.
    """
    key_payload = _debezium_payload(record.get("key"))
    value_payload = _debezium_payload(record.get("value", record))

    op = str(value_payload.get("op") or "").strip().lower()
    source = _as_dict(value_payload.get("source"))
    db = str(source.get("db") or source.get("database") or "").strip()
    table = str(source.get("table") or "").strip()
    lsn_raw = source.get("lsn")
    lsn = "" if lsn_raw is None else str(lsn_raw)

    if not db or not table:
        return QuarantinedRecord(
            db=db, table=table, reason="missing source.db/source.table", op=op, lsn=lsn
        )

    mapped_op = DEBEZIUM_OP_MAP.get(op)
    if mapped_op is None:
        return QuarantinedRecord(
            db=db, table=table, reason=f"unmapped Debezium op {op!r}", op=op, lsn=lsn
        )

    pk = _composite_key(key_payload)
    if not pk:
        return QuarantinedRecord(
            db=db,
            table=table,
            reason="missing/empty Debezium message key (no primary key to identify the row)",
            op=op,
            lsn=lsn,
        )

    ontology_class, mapping_version = _resolve_ontology_class(db, table)
    if ontology_class is None:
        return QuarantinedRecord(
            db=db,
            table=table,
            reason=(
                f"no connector_manifest.yml resource for {db}.{table} — table not"
                " onboarded, never auto-classed"
            ),
            op=op,
            lsn=lsn,
        )

    row_field = "after" if mapped_op == "upsert" else "before"
    row = _as_dict(value_payload.get(row_field))
    if not row:
        return QuarantinedRecord(
            db=db,
            table=table,
            reason=f"missing Debezium {row_field!r} row for op={op!r}",
            op=op,
            lsn=lsn,
        )

    typed_payload = dict(row)
    typed_payload["node_type"] = ontology_class
    # P2's literal acceptance shape (DEC-CA-03): properties.provenance.lsn on
    # the committed node, matched against the WAL LSN. Setting it inside
    # typed_payload (rather than only ChangeEnvelope.provenance, which the
    # native write path does not automatically project onto node properties
    # — CONCEPT:AU-KG.ingest.debezium-changeenvelope, verified against
    # envelope_ingest.py's row-materialization path, see this module's
    # docstring) is what actually lands it there.
    typed_payload["provenance"] = {
        "connector": "debezium-pg",
        "db": db,
        "table": table,
        "lsn": lsn,
        "op": op,
    }

    envelope = ChangeEnvelope(
        connector="cdc",
        operation=mapped_op,
        tenant=tenant,
        source_instance=f"{db}.{table}",
        source_object_id=pk,
        source_version=lsn,
        event_time=_event_time(source, value_payload),
        schema_version=mapping_version or "1",
        typed_payload=typed_payload,
        checkpoint=lsn or None,
        provenance={"lsn": lsn, "db": db, "table": table, "connector": "debezium-pg"},
    )
    return envelope


# ── envelope-source registry ────────────────────────────────────────────────
#
# CA-22's exclusive path (the connector-delta-handler hub, FO-CA-001) never gets imported or
# edited here — CA-22 looks this handler up BY NAME and assigns it to its own
# `_DELTA_HANDLERS["cdc"]` entry once it lands (ordered CA-21 -> CA-22).
# Mirrors the existing `register_sink`/`get_sink` pattern
# (`enrichment/writeback/core.py:209-219`).
_ENVELOPE_SOURCES: dict[str, Callable[..., dict[str, Any]]] = {}


def register_envelope_source(name: str, handler: Callable[..., dict[str, Any]]) -> None:
    """Register a sync-shaped envelope-source handler under ``name`` (idempotent)."""
    _ENVELOPE_SOURCES[(name or "").strip().lower()] = handler


def get_envelope_source(name: str) -> Callable[..., dict[str, Any]] | None:
    return _ENVELOPE_SOURCES.get((name or "").strip().lower())


def list_envelope_sources() -> list[str]:
    return sorted(_ENVELOPE_SOURCES)


def run_cdc_catchup(
    engine: Any,
    *,
    mode: str = "delta",
    ids: list[str] | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """The ``"cdc"`` envelope-source handler — a bounded catch-up poll.

    Matches the connector-delta-handler registry's value signature
    (``Callable[[engine], dict[str, Any]]`` keyword-called as ``handler(
    engine, mode=mode, ids=ids, client=client)``) so CA-22's one-line
    ``_DELTA_HANDLERS["cdc"] = get_envelope_source("cdc")`` wiring needs no
    adapter. Consumes whatever is currently available on the subscribed
    ``cdc``/Debezium topics — up to one bounded batch — maps each record,
    commits through ``ingest_envelope``'s existing atomic boundary, and
    returns; it never blocks waiting for more messages (that's the
    always-on background daemon's job, not an on-demand sweep's).

    ``ids`` is accepted for signature parity with every other
    ``_DELTA_HANDLERS`` entry but unused: Kafka Connect redelivery, not a
    caller-supplied id list, drives what this handler sees.

    ``client``, when given, is an already-connected
    :class:`~..streams.kafka_adapter.DebeziumKafkaConsumer` (or an injected
    test double exposing the same ``drain_once``/``connect``/``disconnect``
    surface) — the same "inject to test without a broker" seam
    :class:`~..streams.kafka_adapter.KafkaStreamAdapter` already offers. When
    absent, a real one is constructed and connected here — gated by
    :func:`~..streams.kafka_adapter.debezium_consumer_enabled` (the
    ``KAFKA_CDC_DEBEZIUM_ENABLED`` flag, CA-21-W05's migration/rollback
    contract): disabled means this handler never touches Kafka at all and
    reports ``status="skipped"``, so unsetting the flag is a real, immediate
    rollback with no other state to unwind. A caller supplying its own
    ``client`` (a test double, or an operator who already holds a connected
    consumer) bypasses the flag deliberately — the same "explicit injection
    wins" convention :class:`~..streams.kafka_adapter.KafkaStreamAdapter`
    itself uses for its own ``consumer=`` parameter.
    """
    # `_run_async` (CONCEPT:AU-KG.ingest.mcp-tool-connector's shared helper) is the
    # SAME "sync surface over an async client, safe whether or not a loop is
    # already running" primitive every other `_DELTA_HANDLERS` entry that
    # wraps async I/O already uses (e.g. the Technitium delta handler) —
    # reused rather than a second implementation.
    from ...protocols.source_connectors.connectors.mcp_package import _run_async
    from ..streams.kafka_adapter import DebeziumKafkaConsumer, debezium_consumer_enabled

    del mode, ids  # accepted for _DELTA_HANDLERS signature parity only

    owns_client = client is None
    if owns_client and not debezium_consumer_enabled():
        return {
            "status": "skipped",
            "source": "cdc",
            "reason": "KAFKA_CDC_DEBEZIUM_ENABLED is unset — consumer disabled",
            "counts": {"succeeded": 0, "skipped": 0, "quarantined": 0, "failed": 0},
        }
    consumer = client or DebeziumKafkaConsumer(config=None)

    async def _drain() -> dict[str, Any]:
        if owns_client:
            await consumer.connect()
        try:
            return await consumer.drain_once(engine)
        finally:
            if owns_client:
                await consumer.disconnect()

    return _run_async(_drain())


register_envelope_source("cdc", run_cdc_catchup)
