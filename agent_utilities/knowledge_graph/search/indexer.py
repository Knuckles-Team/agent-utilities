"""``eg.cdc.<graph>`` -> OpenSearch. The CDC-fed indexer.

(CONCEPT:AU-KG.retrieval.opensearch-cdc-indexer, CA-24, DEC-CA-09, DEC-CA-03)

Consumes the real wire envelope eg's CA-11 Kafka sink emits on
``eg.cdc.<graph>`` and writes/deletes documents in the `DEC-CA-09` shape.

**Premise correction (report, don't paper over — the lane doc's own
instruction).** `DEC-CA-03`'s JSON schema names twelve envelope fields
including ``tenant``, ``marking``, ``actor``, ``lsn``. The lane doc for this
package assumed "the envelope already carries a marking field ... so the
indexer's primary marking source is the CDC message itself, with
MARKING_REGISTRY/permissioning.py as the read-time authority for any gap."
**That premise is false as measured against the real CA-11 sink** —
``crates/eg-stream/src/sink.rs``'s own module doc and its
``envelope_serializes_null_before_after_when_absent`` test assert these four
fields are *absent from the wire entirely*, not blank
(``assert!(v.get("tenant").is_none())`` etc.), because ``CdcHub::emit``'s
call site has no ``CarrierAuthority``/marking concept to source them from.
So this module NEVER reads ``marking``/``tenant`` off the envelope (there is
nothing to read): ``tenant`` is resolved as the eg **``graph`` id itself**
(eg's graph is already the tenant-scoping unit in this ecosystem — see the
program's own "Two graphs: __commons__ vs tenant" convention, and
`permissioning.py`'s own ``(tenant, node_id)`` keying comment), and
``marking`` is resolved from ``permissioning.markings_for`` on *every*
message, unconditionally — which is actually a stricter reading of this
lane's own "Prohibited fallback: never index a node without resolving its
marking set first" invariant than the lane doc's original "primary source is
the envelope, registry fills gaps" design, not a weaker one.

The wire envelope this module actually parses (8 fields, per
``eg_stream::sink::Envelope``): ``seq`` (int), ``graph`` (str), ``op``
(``"upsert"`` | ``"tombstone"`` — note: *not* ``"delete"``, the internal
``ChangeEnvelope.Operation`` literal a different lane's Debezium path uses;
this is eg's own wire vocabulary), ``node_id`` (str), ``edge_id``
(str|null), ``before``/``after`` (decoded JSON or null), ``ts`` (str).
``object_type`` (`DEC-CA-09`'s ``node_type`` field) is not on the wire
either — it is read off ``after``/``before``'s own ``"type"``/``"node_type"``
property (eg stores it as an ordinary node property, confirmed against
``sink.rs``'s own test fixture: ``json!({"type": "Person", "name": "a"})``).

**Fail-closed offset semantics**, matching
``knowledge_graph.streams.kafka_adapter.DebeziumKafkaConsumer``'s contract
exactly (``record.consumed = result.ok``, never unconditional): an
``applied`` write, a ``rejected_stale`` out-of-order rejection (P3's
negative case — *correctly refusing* to reorder IS successful handling, not
a failure), and a ``quarantined`` malformed/unclassifiable record all
consume the record (offset commits past it). A ``failed`` result (an
OpenSearch read/write exception — cluster down, timeout, ...) does NOT
commit; :meth:`EgCdcKafkaConsumer.drain_once` stops at the first ``failed``
record in a batch, leaving it and everything after it uncommitted so a
restart re-delivers from exactly there.

**Idempotency/ordering — ``(node_id, seq)``, checked against the index
itself, not local process state.** Before writing, the target document is
read back; a redelivered message carrying the SAME ``seq`` as what's already
indexed is a safe no-op overwrite (``existing.updated_lsn >= seq`` rejects,
covering both "same seq" and "older seq"), and a message with a strictly
newer ``seq`` applies. Checking against the index itself (rather than an
in-process ``dict``) means this is correct across process restarts with no
extra durable state to maintain — the derived index already carries its own
watermark per document.
"""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

from ..streams.kafka_adapter import KafkaStreamAdapter
from . import dls, doc_shape
from .client import OpenSearchClient

logger = logging.getLogger(__name__)

__all__ = [
    "EG_CDC_TOPIC_PATTERN",
    "ApplyResult",
    "DrainResult",
    "resolve_object_type",
    "derive_content",
    "apply_envelope",
    "EgCdcKafkaConsumer",
]


class ApplyResult(TypedDict, total=False):
    """Return shape of :func:`apply_envelope` and its private per-op
    helpers. Every field is optional -- WHICH are present depends on
    ``status`` (see :func:`apply_envelope`'s own docstring for the four
    branches: ``applied``/``rejected_stale``/``quarantined``/``failed``)."""

    status: str
    reason: str
    index: str | None
    seq: int
    op: str
    marking: list[str]
    indices: list[str | None]


class DrainResult(TypedDict):
    """Return shape of :meth:`EgCdcKafkaConsumer.drain_once`."""

    status: str
    source: str
    counts: dict[str, int]
    watermark: int | None
    details: dict[str, Any]


# eg.cdc.<graph> (DEC-CA-03; CA-11's real ENV_TOPIC_PREFIX default is
# "eg.cdc." — see epistemic-graph crates/eg-stream/src/sink.rs / server/
# cdc_sink/mod.rs::DEFAULT_TOPIC_PREFIX). One topic per graph, single
# partition (DEC-CA-03: "preserves seq order end to end").
EG_CDC_TOPIC_PATTERN = re.compile(r"^eg\.cdc\..+$")

_VALID_OPS = ("upsert", "tombstone")


def resolve_object_type(row: dict[str, Any] | None) -> str | None:
    """``row["type"]`` or ``row["node_type"]`` — never guessed. ``None`` when
    absent, so the caller can quarantine/fall back explicitly."""
    if not isinstance(row, dict):
        return None
    val = row.get("type") or row.get("node_type")
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def derive_content(row: dict[str, Any] | None) -> str:
    """A rudimentary full-text ``content`` field: every scalar property
    value, space-joined, excluding the type-discriminator keys (already
    carried structurally as ``node_type``). Not a relevance-tuned analyzer —
    `DEC-CA-09` only requires the field exist as full text; ranking quality
    is out of this lane's scope."""
    if not isinstance(row, dict):
        return ""
    parts = [
        str(v)
        for k, v in row.items()
        if k not in ("type", "node_type") and isinstance(v, (str, int, float, bool))
    ]
    return " ".join(parts)[:5000]


def apply_envelope(
    opensearch: OpenSearchClient, envelope: dict[str, Any]
) -> ApplyResult:
    """Apply one decoded ``eg.cdc.<graph>`` envelope. Returns
    ``{"status": "applied"|"rejected_stale"|"quarantined"|"failed", ...}``.

    Never raises for a data-shape problem (malformed envelope, unresolvable
    object_type) — those are ``quarantined`` (logged, not applied, but the
    record IS considered handled). Only an OpenSearch I/O exception produces
    ``"failed"`` (not handled — the caller must not advance past it).
    """
    seq = envelope.get("seq")
    graph = envelope.get("graph")
    op = str(envelope.get("op") or "").strip().lower()
    node_id = str(envelope.get("node_id") or "").strip()

    if seq is None or not graph or not node_id or op not in _VALID_OPS:
        return {
            "status": "quarantined",
            "reason": (
                "malformed eg CDC envelope: requires seq, graph, node_id, "
                f"op in {_VALID_OPS!r} (got seq={seq!r}, graph={graph!r}, "
                f"node_id={node_id!r}, op={op!r})"
            ),
        }
    try:
        seq_int = int(seq)
    except (TypeError, ValueError):
        return {"status": "quarantined", "reason": f"non-integer seq {seq!r}"}

    tenant = str(graph).strip()

    if op == "upsert":
        return _apply_upsert(
            opensearch, envelope, tenant=tenant, node_id=node_id, seq=seq_int
        )
    return _apply_tombstone(
        opensearch, envelope, tenant=tenant, node_id=node_id, seq=seq_int
    )


def _apply_upsert(
    opensearch: OpenSearchClient,
    envelope: dict[str, Any],
    *,
    tenant: str,
    node_id: str,
    seq: int,
) -> ApplyResult:
    after = envelope.get("after")
    row = after if isinstance(after, dict) else {}
    object_type = resolve_object_type(row)
    if not object_type:
        return {
            "status": "quarantined",
            "reason": (
                f"cannot resolve node_type for node_id={node_id!r} — 'after' "
                "carries no 'type'/'node_type' property, never guessed"
            ),
        }
    index = doc_shape.index_name(tenant, object_type)

    try:
        existing = opensearch.get_document(index, node_id)
    except Exception as exc:  # noqa: BLE001 - I/O failure, must not advance the offset
        return {
            "status": "failed",
            "reason": f"OpenSearch read failed: {exc}",
            "index": index,
        }

    if existing is not None and _existing_seq(existing) >= seq:
        return {
            "status": "rejected_stale",
            "reason": (
                f"incoming seq {seq} <= indexed seq {_existing_seq(existing)} "
                f"for node_id={node_id!r}"
            ),
            "index": index,
        }

    try:
        # Prohibited fallback (lane invariant): NEVER index a node without
        # resolving its marking set first — resolved fresh on every write,
        # never cached/derived locally.
        marking = dls.markings_for_node(node_id, tenant=tenant)
    except Exception as exc:  # noqa: BLE001 - marking-authority failure fails the write closed
        return {
            "status": "failed",
            "reason": f"marking resolution failed for node_id={node_id!r}: {exc}",
            "index": index,
        }

    document = doc_shape.build_document(
        node_id=node_id,
        node_type=object_type,
        tenant=tenant,
        marking=marking,
        properties=row,
        updated_lsn=seq,
        content=derive_content(row),
    )
    try:
        opensearch.ensure_index(index, mappings=doc_shape.INDEX_MAPPINGS)
        opensearch.index_document(index, node_id, dict(document))
    except Exception as exc:  # noqa: BLE001 - I/O failure, must not advance the offset
        return {
            "status": "failed",
            "reason": f"OpenSearch write failed: {exc}",
            "index": index,
        }

    return {
        "status": "applied",
        "op": "upsert",
        "index": index,
        "seq": seq,
        "marking": sorted(marking),
    }


def _apply_tombstone(
    opensearch: OpenSearchClient,
    envelope: dict[str, Any],
    *,
    tenant: str,
    node_id: str,
    seq: int,
) -> ApplyResult:
    before = envelope.get("before")
    row = before if isinstance(before, dict) else {}
    object_type = resolve_object_type(row)

    if object_type:
        index = doc_shape.index_name(tenant, object_type)
        try:
            existing = opensearch.get_document(index, node_id)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failed",
                "reason": f"OpenSearch read failed: {exc}",
                "index": index,
            }
        if existing is not None and _existing_seq(existing) >= seq:
            return {
                "status": "rejected_stale",
                "reason": (
                    f"incoming tombstone seq {seq} <= indexed seq "
                    f"{_existing_seq(existing)} for node_id={node_id!r}"
                ),
                "index": index,
            }
        try:
            opensearch.delete_document(index, node_id)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failed",
                "reason": f"OpenSearch delete failed: {exc}",
                "index": index,
            }
        return {"status": "applied", "op": "tombstone", "index": index, "seq": seq}

    # A tombstone whose ``before`` carries no type (or no before at all — a
    # bare delete) cannot resolve a single index deterministically. Rather
    # than guess, search every index this tenant owns for the document and
    # delete it there — this is the ONE place this lane relies on a
    # cross-index lookup instead of the direct-id path, and it is still a
    # bounded, exact `node_id` term query, not a scan.
    try:
        pattern = doc_shape.tenant_wildcard(tenant)
        response = opensearch.search(
            pattern, {"query": {"term": {"node_id": node_id}}, "size": 25}
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": f"OpenSearch tenant lookup failed: {exc}"}

    hits = response.get("hits", {}).get("hits", [])
    if not hits:
        return {
            "status": "applied",
            "op": "tombstone",
            "index": None,
            "seq": seq,
            "reason": "no matching document (already absent) — safe no-op",
        }

    applied_any = False
    for hit in hits:
        src = hit.get("_source", {}) or {}
        if _existing_seq(src) >= seq:
            continue
        hit_index = hit.get("_index")
        try:
            opensearch.delete_document(hit_index, node_id)
            applied_any = True
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failed",
                "reason": f"OpenSearch delete failed: {exc}",
                "index": hit_index,
            }

    return {
        "status": "applied" if applied_any else "rejected_stale",
        "op": "tombstone",
        "seq": seq,
        "indices": [h.get("_index") for h in hits],
    }


def _existing_seq(source: dict[str, Any]) -> int:
    try:
        return int(source.get("updated_lsn", -1))
    except (TypeError, ValueError):
        return -1


class EgCdcKafkaConsumer(KafkaStreamAdapter):
    """``eg.cdc.<graph>`` consumer entry point.

    Subscribes by topic PATTERN (:data:`EG_CDC_TOPIC_PATTERN`, overridable
    via ``config.topic_pattern``) rather than one fixed topic — a real
    deployment has one topic per graph. Reuses
    :class:`~..streams.kafka_adapter.KafkaStreamAdapter`'s injectable-
    consumer test seam (FO-CA-002: this lane rebases onto CA-21's consumer
    entry rather than adding a second aiokafka wrapper) instead of writing a
    second raw ``aiokafka`` connect path.

    Fail-closed offset semantics — see the module docstring.
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
            or EG_CDC_TOPIC_PATTERN
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
                "eg CDC indexer requires 'aiokafka'. Install agent-utilities[kafka]."
            ) from exc
        self._consumer = AIOKafkaConsumer(
            bootstrap_servers=self._servers(),
            group_id=getattr(self.config, "group_id", "au-cdc-opensearch-indexer"),
            # Fail-closed (module doc): explicit per-record offset commits in
            # drain_once, never a timer.
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        )
        await self._consumer.start()
        self._consumer.subscribe(pattern=self._topic_pattern.pattern)
        self._connected = True
        logger.info(
            "eg CDC indexer connected to %s, topic pattern %r",
            self._servers(),
            self._topic_pattern.pattern,
        )

    @staticmethod
    def _decode_json(value: Any) -> Any:
        import json

        if value is None:
            return None
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except Exception:  # noqa: BLE001 - undecodable bytes fail closed below
                return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (ValueError, json.JSONDecodeError):
                return None
        return value

    async def drain_once(
        self, opensearch: OpenSearchClient, *, batch_size: int = 500
    ) -> DrainResult:
        """One bounded poll + apply + commit pass. Never blocks indefinitely.

        Mirrors ``DebeziumKafkaConsumer.drain_once``'s per-partition
        stop-at-first-failure / commit-exact-next-offset discipline.
        """
        if not self._connected or self._consumer is None:
            raise RuntimeError("eg CDC indexer not connected")

        raw = await self._consumer.getmany(
            timeout_ms=getattr(self.config, "poll_timeout_ms", 1000),
            max_records=batch_size,
        )
        partitions = raw if isinstance(raw, dict) else {}

        counts = {"applied": 0, "rejected_stale": 0, "quarantined": 0, "failed": 0}
        stopped_early = False
        failure_detail: dict[str, Any] | None = None
        last_seq: int | None = None

        for topic_partition, records in partitions.items():
            commit_offset: int | None = None
            for rec in records:
                value = self._decode_json(getattr(rec, "value", rec))
                if not isinstance(value, dict):
                    counts["failed"] += 1
                    failure_detail = {
                        "status": "failed",
                        "reason": "undecodable eg CDC payload (not a JSON object)",
                    }
                    stopped_early = True
                    break

                result = apply_envelope(opensearch, value)
                status = result.get("status")
                if status in ("applied", "rejected_stale", "quarantined"):
                    counts[status] += 1
                    if status == "applied":
                        last_seq = result.get("seq", last_seq)
                    if status == "quarantined":
                        logger.warning(
                            "eg CDC indexer: quarantined record: %s",
                            result.get("reason"),
                        )
                    commit_offset = getattr(rec, "offset", commit_offset)
                    continue

                # "failed": an OpenSearch I/O exception. STOP — never
                # advance the offset past an unverified write.
                counts["failed"] += 1
                failure_detail = dict(result)
                stopped_early = True
                break

            if commit_offset is not None:
                await self._consumer.commit({topic_partition: commit_offset + 1})
            if stopped_early:
                break

        status = "failed" if stopped_early else "ok"
        return {
            "status": status,
            "source": "eg-cdc",
            "counts": counts,
            "watermark": last_seq,
            "details": {"failure": failure_detail} if failure_detail else {},
        }
