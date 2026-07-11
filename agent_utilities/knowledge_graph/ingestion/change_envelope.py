"""``ChangeEnvelope`` — the canonical unit-of-change contract for connector ingestion

(CONCEPT:AU-KG.ingest.change-envelope, AU-P1-6).

**The gap this closes.** Every connector today hands ``sync_source``/
``ingest_external_batch`` an ad hoc ``{"id": ..., "type": ..., **props}`` dict
(see :mod:`knowledge_graph.core.source_sync`'s ``_sync_leanix`` et al.) — each
connector independently decides what "id"/"updated"/"acl" keys mean, provenance
is stamped separately (:func:`knowledge_graph.enrichment.provenance.stamp_source`),
watermarks are tracked separately (``_read_watermark``/``_write_watermark``),
and ACL is a bespoke :class:`~agent_utilities.protocols.source_connectors.base.ExternalAccess`
bolted on per-connector. None of that is one typed, self-describing unit a
connector can emit and any consumer (the write layer, a future CDC/webhook
receiver, an audit trail, a replay/backfill tool) can reason about uniformly.

``ChangeEnvelope`` is that one unit — a single dataclass that carries:

* **identity** — ``envelope_id`` (this specific delivery), ``idempotency_key``
  (deterministic from connector+instance+object+version+operation, so redelivery
  of the same change is provably a no-op), ``tenant``.
* **provenance/lineage** — ``connector``, ``source_instance`` (multi-tenant
  instance name, e.g. a named GitLab/ServiceNow instance), ``source_object_id``,
  ``source_version`` (the connector's own watermark/version value for THIS
  object — not the whole source's watermark), ``schema_version``,
  ``ontology_mapping_version`` (which ``connector_manifest.yml``
  ``schema_mappings`` crosswalk revision this envelope was mapped under).
* **bitemporal timestamps** — ``event_time`` (when it happened upstream),
  ``valid_time`` (when the fact is true in the domain — may differ from
  ``event_time``, e.g. a backdated ServiceNow change), ``observed_time`` (when
  THIS system ingested it).
* **payload** — ``operation`` (``upsert``/``delete``/``snapshot_complete`` — the
  last is the reconcile-pass "authoritative snapshot ends here" marker
  ``source_sync._reconcile`` already needs), ``payload_type`` +
  ``typed_payload``/``blob_ref`` (exactly one set, matching the existing
  content-vs-blob split the fleet already has for large objects).
* **governance** — ``source_acl`` (reuses :class:`ExternalAccess` — the exact
  type :func:`permission_sync.sync_access` already consumes), ``classification``
  (reuses :class:`DataClassification` — the exact enum ``NodeACL`` already
  consumes), ``retention``, ``legal_hold``.
* **operational** — ``provenance`` (free-form dict: manifest hash, connector
  version, generated_by), ``confidence``, ``checkpoint`` (the watermark cursor
  to persist once this envelope is durably processed — the typed twin of
  ``source_sync._write_watermark``'s bare string), ``trace_context`` (W3C
  traceparent, mirroring :class:`~knowledge_graph.core.session.GraphSession`).

**Scope (deliberately limited, per AU-P1-6).** This module introduces the type
and ONE adapter each way — :meth:`ChangeEnvelope.from_connector_record` (bridge
IN from today's ``{"id", "type", **props}`` shape, mirroring
``GraphSession.from_ambient``'s "read today's ambient shape, produce the new
explicit object" pattern) and :meth:`ChangeEnvelope.to_entity_dict` (bridge OUT,
so a caller holding an envelope can still feed today's
``engine.ingest_external_batch``/``write_entities`` unchanged).

**AU-P1-5** (:mod:`~..ingestion.envelope_ingest`) is the ingestion-path
consolidation this scope note originally deferred: one ``ingest_envelope(engine,
envelope)`` atomic transaction (validate -> resolve identity -> write -> lineage
+checkpoint -> CDC -> watermark, crash-resume safe), a first wave of migrated
``source_sync`` handlers (``leanix``, ``claude_memory``), and 5 brand-new
envelope-native connectors closing the L27 mandatory-manifest gap. The
remaining ``source_sync`` handlers are enumerated as still-legacy right in
``source_sync``'s own module docstring — a scoped, stated follow-up, not a
silent gap.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal

from ...models.company_brain import DataClassification
from ...protocols.source_connectors.base import ExternalAccess

__all__ = [
    "ChangeEnvelope",
    "Operation",
    "OPERATIONS",
]

Operation = Literal["upsert", "delete", "snapshot_complete"]
OPERATIONS: frozenset[str] = frozenset({"upsert", "delete", "snapshot_complete"})


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _stable_key(*parts: str) -> str:
    """Deterministic sha256 over ordered string parts (idempotency-key derivation).

    Kept local (rather than importing :func:`materialization.content_hash`) so
    this module has no dependency on the write-layer/backends — it is a pure
    type module, importable from anywhere a connector lives.
    """
    blob = "\x1f".join(parts)  # unit-separator: parts can't collide across a join
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChangeEnvelope:
    """One self-describing unit of change from a connector (CONCEPT:AU-KG.ingest.change-envelope).

    All fields beyond ``connector``/``operation`` are optional/defaulted, so a
    connector can start by populating just identity + payload and add
    governance/bitemporal fields incrementally — mirrors
    :class:`~agent_utilities.knowledge_graph.core.session.GraphSession`'s "every
    field optional, nothing breaks" design.

    Attributes:
        envelope_id: Unique id for THIS delivery attempt (a redelivery of the
            same logical change gets a new ``envelope_id`` but the SAME
            ``idempotency_key`` — that's what makes redelivery a safe no-op).
        idempotency_key: Deterministic dedup key. Auto-derived in
            :meth:`__post_init__` from ``connector``/``tenant``/
            ``source_instance``/``source_object_id``/``source_version``/
            ``operation`` when not supplied explicitly.
        tenant: The KG tenant this change belongs to (mirrors
            ``GraphSession.tenant``). Empty means the default/single tenant.
        connector: The connector package/identifier that emitted this change
            (e.g. ``"servicenow-api"`` or a ``sync_source`` source key like
            ``"servicenow"`` — whichever this connector is known by).
        source_instance: The specific configured instance name for a
            multi-instance connector (e.g. one of ``gitlab_instances``), or
            ``""`` for a single-instance connector.
        source_object_id: The upstream object's own id (the connector's
            ``id_field`` value — see ``mcp_source_presets.json``).
        source_version: The upstream object's own version/updated marker (the
            connector's ``updated_field`` value for THIS object) — distinct
            from the source-wide watermark tracked by
            ``source_sync._read_watermark``.
        operation: ``"upsert"`` | ``"delete"`` | ``"snapshot_complete"``. The
            last is the reconcile "authoritative live-id snapshot ends here"
            marker (parallels ``source_sync._reconcile``'s ``fetch_ok``
            distinction — CONCEPT:AU-P0-4).
        event_time: ISO-8601 UTC — when the change happened upstream, if the
            source reports one (``None`` if unknown).
        valid_time: ISO-8601 UTC — when the fact is true in the domain
            (bitemporal; may predate/postdate ``event_time``, e.g. a backdated
            ticket). ``None`` when the source has no valid-time concept
            distinct from ``event_time``.
        observed_time: ISO-8601 UTC — when THIS system observed/ingested the
            change. Defaults to now at construction.
        schema_version: The connector's ``connector_manifest.yml``
            ``schema_version`` this envelope's payload shape matches.
        ontology_mapping_version: Which ``schema_mappings`` crosswalk revision
            (from the connector's manifest) this envelope was mapped under —
            lets a consumer detect a stale mapping after a manifest regenerate.
        payload_type: A short tag for what ``typed_payload``/``blob_ref`` holds
            (e.g. ``"json"``, ``"markdown"``, ``"blob"``).
        typed_payload: The structured record body, when it fits inline (most
            connectors). Mutually exclusive with ``blob_ref``.
        blob_ref: A reference (URI/blob-store key) to an out-of-line payload
            too large to inline (e.g. an attachment). Mutually exclusive with
            ``typed_payload``.
        source_acl: The connector-reported access descriptor — reuses
            :class:`ExternalAccess` verbatim so this envelope feeds
            ``permission_sync.sync_access`` unchanged.
        classification: The KG-2.46 :class:`DataClassification` this object
            should carry. Defaults to ``INTERNAL`` (fail-closed — mirrors
            ``ExternalAccess.quarantined()``'s conservative default; a
            connector must explicitly say ``PUBLIC`` for a document to be
            world-readable).
        retention: A retention-policy label/period (e.g. an ISO-8601 duration
            ``"P90D"`` or a named policy id). ``None`` means "no retention
            policy stated" — NOT "retain forever"; a consumer must not assume
            either way (see the 12 connector manifests' explicit
            ``policy.classification / policy.retention: UNKNOWN`` review_todos
            for why this must stay optional rather than defaulted).
        legal_hold: When ``True``, this object must not be purged by any
            retention policy regardless of ``retention``.
        provenance: Free-form lineage dict (e.g. ``{"manifest_hash": ...,
            "generated_by": ..., "connector_version": ...}``). Deliberately a
            plain dict (not the manifest's ``ProvenanceSpec``) — this is
            per-record lineage, not per-manifest.
        confidence: ``[0.0, 1.0]`` — how much this envelope's content should be
            trusted (1.0 = a direct, unprocessed read from the source; lower
            for a heuristically-derived/inferred field). Defaults to ``1.0``.
        checkpoint: The watermark cursor value to persist once this envelope
            is durably processed — the typed counterpart of
            ``source_sync._write_watermark``'s bare string argument.
        trace_context: The W3C ``traceparent``/correlation id this change
            should be attributed to (mirrors
            ``GraphSession.trace_context``).
        live_ids: For a ``snapshot_complete`` marker ONLY (AU-P1-5,
            :mod:`~..ingestion.envelope_ingest`) — the authoritative set of
            still-live upstream ids as of this snapshot, so the atomic ingest
            transaction can tombstone any previously-known node absent from it
            (the envelope-native counterpart of ``source_sync._reconcile``).
            Empty for every other operation.
    """

    connector: str
    operation: Operation = "upsert"

    envelope_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    idempotency_key: str = ""
    tenant: str = ""

    source_instance: str = ""
    source_object_id: str = ""
    source_version: str = ""

    event_time: str | None = None
    valid_time: str | None = None
    observed_time: str = field(default_factory=_now_iso)

    schema_version: str = "1"
    ontology_mapping_version: str = ""

    payload_type: str = "json"
    typed_payload: dict[str, Any] | None = None
    blob_ref: str | None = None

    source_acl: ExternalAccess | None = None
    classification: DataClassification = DataClassification.INTERNAL
    retention: str | None = None
    legal_hold: bool = False

    provenance: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    checkpoint: str | None = None
    trace_context: str | None = None

    live_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.operation not in OPERATIONS:
            raise ValueError(
                f"ChangeEnvelope.operation must be one of {sorted(OPERATIONS)}, "
                f"got {self.operation!r}"
            )
        if self.typed_payload is not None and self.blob_ref is not None:
            raise ValueError(
                "ChangeEnvelope: typed_payload and blob_ref are mutually exclusive "
                "(set at most one) — this envelope had both."
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"ChangeEnvelope.confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )
        if not self.idempotency_key:
            object.__setattr__(
                self,
                "idempotency_key",
                _stable_key(
                    self.connector,
                    self.tenant,
                    self.source_instance,
                    self.source_object_id,
                    self.source_version,
                    self.operation,
                ),
            )

    # ------------------------------------------------------------------
    # Bridge IN — from today's ``{"id", "type", **props}`` connector-record shape
    # ------------------------------------------------------------------
    @classmethod
    def from_connector_record(
        cls,
        record: dict[str, Any],
        *,
        connector: str,
        operation: Operation = "upsert",
        tenant: str = "",
        source_instance: str = "",
        id_field: str = "id",
        version_field: str = "updatedAt",
        schema_version: str = "1",
        ontology_mapping_version: str = "",
        session: Any | None = None,
        **overrides: Any,
    ) -> ChangeEnvelope:
        """Build an envelope from today's ad hoc connector-record dict.

        The back-compat bridge (mirrors ``GraphSession.from_ambient``'s "read
        today's shape, produce the new explicit object" pattern): every
        existing connector already builds ``{"id": ..., "type": ...,
        **props}`` rows (see ``source_sync._sync_leanix`` et al.) — this reads
        that same shape without requiring any connector to change yet.

        Args:
            record: One connector row, e.g. ``{"id": n.id, "type": n.type,
                **n.props}`` — matches ``_sync_leanix``'s ``entities`` shape.
            connector: The connector identifier (see :attr:`connector`).
            operation: Defaults to ``"upsert"``. Pass ``"delete"`` for a
                tombstone row (``source_sync._reconcile``'s "no longer live"
                case) or use :meth:`snapshot_complete` for the reconcile marker.
            id_field: Which key in ``record`` is the source object id.
                ``record[id_field]`` becomes :attr:`source_object_id`.
            version_field: Which key in ``record`` is the source's own
                version/updated marker for this object.
                ``record[version_field]`` becomes :attr:`source_version`.
            session: An optional ``GraphSession`` (or any object exposing
                ``.tenant``/``.trace_context``) to source ``tenant``/
                ``trace_context`` from when the caller didn't pass them
                explicitly — the same ambient-bridge idea
                ``GraphSession.from_ambient()`` itself uses. Import is deferred
                to avoid a hard dependency on the session module for callers
                that don't have one.
            **overrides: Any other :class:`ChangeEnvelope` field to set
                directly (e.g. ``classification=DataClassification.PUBLIC``,
                ``source_acl=ExternalAccess.public()``).

        Returns:
            A new :class:`ChangeEnvelope` wrapping ``record`` as
            ``typed_payload`` (with ``id``/``type`` kept in the payload too,
            so :meth:`to_entity_dict` round-trips losslessly).
        """
        if session is None and (not tenant or "trace_context" not in overrides):
            try:
                from ..core.session import GraphSession

                session = GraphSession.from_ambient()
            except Exception:  # noqa: BLE001 — session module is optional context
                session = None

        resolved_tenant = tenant or (getattr(session, "tenant", "") or "")
        resolved_trace = overrides.pop("trace_context", None) or (
            getattr(session, "trace_context", None) if session is not None else None
        )

        source_object_id = str(record.get(id_field) or "")
        source_version = str(record.get(version_field) or "")

        access = overrides.pop("source_acl", None)
        if access is None:
            raw_access = record.get("external_access")
            if isinstance(raw_access, ExternalAccess):
                access = raw_access
            elif isinstance(raw_access, dict):
                access = ExternalAccess.model_validate(raw_access)

        return cls(
            connector=connector,
            operation=operation,
            tenant=resolved_tenant,
            source_instance=source_instance,
            source_object_id=source_object_id,
            source_version=source_version,
            event_time=overrides.pop("event_time", None) or source_version or None,
            schema_version=schema_version,
            ontology_mapping_version=ontology_mapping_version,
            typed_payload=dict(record),
            source_acl=access,
            trace_context=resolved_trace,
            **overrides,
        )

    @classmethod
    def snapshot_complete(
        cls,
        *,
        connector: str,
        tenant: str = "",
        source_instance: str = "",
        checkpoint: str | None = None,
        live_ids: list[str] | tuple[str, ...] | set[str] | None = None,
        fetch_ok: bool = True,
        **overrides: Any,
    ) -> ChangeEnvelope:
        """The reconcile-pass marker: "the authoritative live-id snapshot for

        this source ends here" — the typed counterpart of
        ``source_sync._reconcile``'s ``fetch_ok=True`` branch. Carries no
        payload; a consumer sees this and knows any previously-seen object NOT
        re-asserted since the last ``snapshot_complete`` may be tombstoned.

        ``live_ids`` (AU-P1-5) is the authoritative still-live id set this
        snapshot asserts — :func:`~..ingestion.envelope_ingest.ingest_envelope`
        feeds it straight to ``source_sync._reconcile`` so an envelope-native
        connector gets the SAME fail-closed empty-snapshot handling as the
        legacy handlers (``fetch_ok=False`` when the live-id fetch itself
        failed/was skipped — never silently tombstone on an unverified empty
        snapshot; see ``_reconcile``'s CONCEPT:AU-P0-4 docstring).

        A marker carries no ``source_object_id`` (there's no single object), so
        its idempotency key would otherwise be IDENTICAL across every reconcile
        pass ever run for this connector — the second, third, ... pass would all
        look like a replay of the first and be silently skipped by
        ``ingest_envelope``'s dedup ledger. Defaulting ``source_version`` to
        ``checkpoint`` (when the caller doesn't override it) makes each
        DISTINCT pass (a new checkpoint/watermark cursor) get its own key while
        a genuine retry of the SAME pass (same checkpoint) still dedupes.
        """
        overrides.setdefault("source_version", checkpoint or "")
        return cls(
            connector=connector,
            operation="snapshot_complete",
            tenant=tenant,
            source_instance=source_instance,
            checkpoint=checkpoint,
            live_ids=tuple(live_ids or ()),
            provenance={**overrides.pop("provenance", {}), "fetch_ok": fetch_ok},
            **overrides,
        )

    # ------------------------------------------------------------------
    # Bridge OUT — back to today's ``{"id", "type", **props}`` shape
    # ------------------------------------------------------------------
    def to_entity_dict(self) -> dict[str, Any]:
        """Render this envelope back to the ``entities`` row shape

        ``engine.ingest_external_batch``/``materialization.write_entities``
        already accept — so a caller holding a :class:`ChangeEnvelope` can
        feed today's write path unchanged while more connectors are migrated.

        Raises:
            ValueError: if this envelope carries a ``blob_ref`` instead of a
                ``typed_payload`` (there's no lossless dict rendering of an
                out-of-line blob) or has no payload at all (e.g. a
                ``snapshot_complete`` marker, which is not a row).
        """
        if self.typed_payload is None:
            raise ValueError(
                "ChangeEnvelope.to_entity_dict(): no typed_payload to render "
                f"(operation={self.operation!r}, blob_ref={self.blob_ref!r}) — "
                "this envelope isn't a row-shaped change."
            )
        row = dict(self.typed_payload)
        row.setdefault("id", self.source_object_id)
        if self.event_time:
            row.setdefault("updatedAt", self.event_time)
        if self.source_acl is not None:
            row.setdefault("external_access", self.source_acl.model_dump())
        return row

    # ------------------------------------------------------------------
    # Immutable "with" helpers (mirrors GraphSession's with_* helpers)
    # ------------------------------------------------------------------
    def with_checkpoint(self, checkpoint: str) -> ChangeEnvelope:
        """Return a copy of this envelope carrying an updated ``checkpoint``."""
        return replace(self, checkpoint=checkpoint)

    def with_classification(self, classification: DataClassification) -> ChangeEnvelope:
        """Return a copy of this envelope reclassified (e.g. after a policy review)."""
        return replace(self, classification=classification)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly rendering (e.g. for an audit log or a future outbox

        table) — enums/``ExternalAccess`` are rendered to plain values.
        """
        return {
            "envelope_id": self.envelope_id,
            "idempotency_key": self.idempotency_key,
            "tenant": self.tenant,
            "connector": self.connector,
            "source_instance": self.source_instance,
            "source_object_id": self.source_object_id,
            "source_version": self.source_version,
            "operation": self.operation,
            "event_time": self.event_time,
            "valid_time": self.valid_time,
            "observed_time": self.observed_time,
            "schema_version": self.schema_version,
            "ontology_mapping_version": self.ontology_mapping_version,
            "payload_type": self.payload_type,
            "typed_payload": self.typed_payload,
            "blob_ref": self.blob_ref,
            "source_acl": self.source_acl.model_dump() if self.source_acl else None,
            "classification": str(self.classification),
            "retention": self.retention,
            "legal_hold": self.legal_hold,
            "provenance": self.provenance,
            "confidence": self.confidence,
            "checkpoint": self.checkpoint,
            "trace_context": self.trace_context,
            "live_ids": list(self.live_ids),
        }

    def to_json(self) -> str:
        """Stable JSON rendering of :meth:`as_dict` (sorted keys, UTF-8)."""
        return json.dumps(
            self.as_dict(), sort_keys=True, default=str, ensure_ascii=False
        )
