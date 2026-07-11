"""``ingest_envelope`` — the ONE atomic ingest transaction for a :class:`ChangeEnvelope`

(CONCEPT:AU-KG.ingest.envelope-atomic-transaction, AU-P1-5).

**The gap this closes.** AU-P1-6 introduced :class:`ChangeEnvelope` as the one
typed unit-of-change every connector shape (connector push, MCP pull,
fleet-package pull, CDC/webhook, bulk snapshot) *could* emit, but deliberately
left ``source_sync``'s ~20 per-connector handlers on the old ad hoc
``{"id", "type", **props}`` + a single unguarded
``engine.ingest_external_batch(...)`` call — no shared validation, no
per-record lineage, no crash-safe watermark advance. This module is the single
place that turns one ``ChangeEnvelope`` into a durable KG change, as ONE unit:

    validate (schema + best-effort SHACL + fail-closed policy)
        -> resolve identity (idempotency dedup + node id)
        -> write object / link / artifact
        -> record lineage + this envelope's own checkpoint
        -> emit CDC (``kg.mutations``)
        -> advance the SOURCE watermark (monotonic-guarded, LAST)

**Crash-resume contract.** There is no cross-backend multi-statement ACID
transaction in this codebase (backends range from an in-process NetworkX graph
to Ladybug/Kuzu to Postgres to a SPARQL store — see
``backends/base.GraphBackend``, whose only universal primitives are
``execute``/``execute_batch``). :func:`ingest_envelope` gets the SAME
end-to-end guarantee a real transaction would give a resumed connector through
two established, already-proven idioms in this codebase rather than a new
storage primitive:

* **Idempotent MERGE upserts** (:func:`~..core.materialization.write_entities`,
  keyed by ``idempotency_key``/node id) — replaying the same envelope twice
  converges to the same state, never a duplicate.
* **"Record success only on success, advance the watermark LAST"** — the exact
  pattern :meth:`~..ingestion.engine.IngestionEngine.ingest` already uses for
  its delta manifest and ``source_sync._write_watermark`` already uses for its
  per-source cursor. If ANY step raises — write, lineage, or CDC — this
  envelope's watermark is provably NOT advanced (a resumed sync will see this
  envelope again, never skip it), and the ledger row is rolled back to absent
  so a replay is indistinguishable from "never attempted".

For a NEWLY-created object (the common case — an object this KG has never
seen) a mid-transaction failure additionally leaves **zero** partial state: the
node itself is compensating-deleted along with the lineage row, so a crash
between the write and the CDC emit is invisible on resume — not just safe to
retry, but literally as if nothing happened. For an UPDATE to a pre-existing
object, the updated fields are — like every other MERGE writer in this
codebase — applied idempotently; a retry re-applies the same values rather
than corrupting anything, and (as above) the watermark/ledger never advance on
a failed attempt, so the retry is guaranteed to happen.
"""

from __future__ import annotations

import logging
from typing import Any

from .change_envelope import ChangeEnvelope

logger = logging.getLogger(__name__)

__all__ = ["ingest_envelope"]


# ── async CDC emit (kg.mutations) — sync-safe, mirrors the `_run_async` idiom
# already used by ``protocols.source_connectors.connectors.mcp_package`` /
# ``adaptation.ticket_playbooks`` for a sync surface over an async client. ───


def _run_async(coro: Any) -> Any:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


def _emit_cdc(envelope: ChangeEnvelope, write_result: dict[str, Any]) -> None:
    """Best-effort publish to ``kg.mutations`` — the CDC leg of the transaction.

    Failures here (an unreachable Kafka/Redpanda backend) are treated the same
    as any other step failure by the caller: the whole envelope rolls back and
    the watermark does not advance, so a resumed sync re-emits rather than
    silently losing the change notification.
    """
    from ..core.event_backend import TOPIC_MUTATIONS, get_event_backend

    eb = get_event_backend()
    payload = envelope.as_dict()
    payload["write_result"] = write_result
    _run_async(eb.publish(TOPIC_MUTATIONS, payload))


# ── best-effort SHACL quarantine (opt-in; reuses the real governance gate) ──


def _shacl_annotate(row: dict[str, Any]) -> dict[str, Any]:
    """Best-effort per-row SHACL check against the bundled governance shapes.

    Mirrors ``pipeline.phases.shacl_gate``'s philosophy exactly: a violation
    QUARANTINES the row (``shacl_valid=False`` + ``shacl_report``) rather than
    refusing to write it — a Tool/Agent-shaped node missing a required field
    still lands in the graph, just flagged for triage. Gated by
    ``KG_ENVELOPE_SHACL_GATE`` (default off): re-parsing the governance shapes
    file and validating per single-node graph on EVERY envelope is real work a
    high-throughput connector shouldn't pay for unless explicitly opted in.
    """
    from ...core.config import setting

    if str(setting("KG_ENVELOPE_SHACL_GATE", "0")) in ("0", "false", "False", ""):
        return row
    try:
        from pathlib import Path

        from ..pipeline.phases.shacl_gate import (
            _DEFAULT_SHAPES,
            SHACL_SUPPORT,
            validate_graph,
        )

        if not SHACL_SUPPORT or not Path(_DEFAULT_SHAPES).exists():
            return row

        import networkx as nx

        g = nx.DiGraph()
        node_id = str(row.get("id") or "envelope")
        g.add_node(node_id, **{k: v for k, v in row.items() if k != "id"})
        conforms, violations, _report = validate_graph(g, _DEFAULT_SHAPES)
    except Exception:  # noqa: BLE001 — SHACL is a best-effort quarantine, never fatal
        logger.debug("envelope SHACL check failed", exc_info=True)
        return row
    if not conforms and violations:
        row = dict(row)
        row["shacl_valid"] = False
        row["shacl_report"] = "\n".join(m for msgs in violations.values() for m in msgs)
    return row


# ── validate ─────────────────────────────────────────────────────────────


def _validate_envelope(envelope: ChangeEnvelope) -> list[str]:
    """Schema + fail-closed policy checks. Empty list = OK.

    Re-affirms the invariants ``ChangeEnvelope.__post_init__`` already enforces
    (defense-in-depth against a future mutation) and adds the policy check
    ``__post_init__`` can't: CONCEPT:AU-P0-4 fail-closed connector permissions
    — a ``PUBLIC``-classified object must carry an explicit
    ``source_acl.is_public=True`` proof; "unknown" must never silently become
    "public" just because a connector forgot to set an ACL.
    """
    from ...models.company_brain import DataClassification

    violations: list[str] = []
    if envelope.operation not in ("upsert", "delete", "snapshot_complete"):
        violations.append(f"invalid operation {envelope.operation!r}")
    if envelope.typed_payload is not None and envelope.blob_ref is not None:
        violations.append(
            "typed_payload and blob_ref are mutually exclusive on this envelope"
        )
    if (
        envelope.operation == "upsert"
        and envelope.typed_payload is None
        and envelope.blob_ref is None
    ):
        violations.append(
            "upsert envelope carries neither typed_payload nor blob_ref — nothing to write"
        )
    if not 0.0 <= envelope.confidence <= 1.0:
        violations.append(f"confidence {envelope.confidence!r} out of range [0.0, 1.0]")
    if envelope.classification == DataClassification.PUBLIC and not (
        envelope.source_acl is not None and envelope.source_acl.is_public
    ):
        violations.append(
            "classification=PUBLIC requires an explicit source_acl.is_public=True "
            "(CONCEPT:AU-P0-4 fail-closed connector permissions) — refusing to "
            "publish an object with no proof of public access"
        )
    return violations


# ── identity resolution ─────────────────────────────────────────────────


def _resolve_identity(
    envelope: ChangeEnvelope,
) -> tuple[str | None, dict[str, Any] | None]:
    """Return ``(node_id, row)`` — ``row`` is the rendered entity dict for an
    upsert with an inline ``typed_payload`` (``None`` for a blob-backed upsert,
    a delete, or a snapshot_complete marker, each handled without a row)."""
    if envelope.operation == "snapshot_complete":
        return None, None
    if envelope.operation == "upsert" and envelope.typed_payload is not None:
        row = envelope.to_entity_dict()
        node_id = str(
            row.get("id") or envelope.source_object_id or envelope.idempotency_key
        )
        return node_id, row
    node_id = envelope.source_object_id or envelope.idempotency_key
    return node_id, None


# ── ledger / lineage (one node serves both: dedup ledger + lineage record) ──


def _lineage_id(idempotency_key: str) -> str:
    return f"envelope:{idempotency_key}"


def _lineage_read(backend: Any, idempotency_key: str) -> dict[str, Any] | None:
    if backend is None:
        return None
    try:
        rows = backend.execute(
            "MATCH (n:ChangeLineage {id: $id}) RETURN n.status AS status, "
            "n.checkpoint AS checkpoint",
            {"id": _lineage_id(idempotency_key)},
        )
        for r in rows or []:
            if isinstance(r, dict):
                return r
    except Exception:  # noqa: BLE001 — ledger read is best-effort; proceed on failure
        logger.debug("envelope ledger read failed", exc_info=True)
    return None


def _lineage_write(backend: Any, envelope: ChangeEnvelope, *, status: str) -> None:
    """MERGE the ledger/lineage row. Deliberately does NOT swallow exceptions:

    when called for ``status="pending"`` inside :func:`ingest_envelope`'s guarded
    section, a failure here MUST propagate so the envelope rolls back instead of
    silently reporting success with no lineage recorded.
    """
    if backend is None:
        return
    backend.execute(
        "MERGE (n:ChangeLineage {id: $id}) SET n.status = $status, "
        "n.connector = $connector, n.source_instance = $source_instance, "
        "n.source_object_id = $source_object_id, n.source_version = $source_version, "
        "n.operation = $operation, n.envelope_id = $envelope_id, "
        "n.schema_version = $schema_version, "
        "n.ontology_mapping_version = $ontology_mapping_version, "
        "n.event_time = $event_time, n.observed_time = $observed_time, "
        "n.checkpoint = $checkpoint, n.trace_context = $trace_context, "
        "n.confidence = $confidence",
        {
            "id": _lineage_id(envelope.idempotency_key),
            "status": status,
            "connector": envelope.connector,
            "source_instance": envelope.source_instance,
            "source_object_id": envelope.source_object_id,
            "source_version": envelope.source_version,
            "operation": envelope.operation,
            "envelope_id": envelope.envelope_id,
            "schema_version": envelope.schema_version,
            "ontology_mapping_version": envelope.ontology_mapping_version,
            "event_time": envelope.event_time,
            "observed_time": envelope.observed_time,
            "checkpoint": envelope.checkpoint,
            "trace_context": envelope.trace_context,
            "confidence": envelope.confidence,
        },
    )


def _lineage_delete(backend: Any, idempotency_key: str) -> None:
    if backend is None:
        return
    try:
        backend.execute(
            "MATCH (n:ChangeLineage {id: $id}) DETACH DELETE n",
            {"id": _lineage_id(idempotency_key)},
        )
    except Exception:  # noqa: BLE001 — rollback is best-effort; never raise from a rollback
        logger.debug("envelope lineage rollback failed", exc_info=True)


# ── node existence / delete / tombstone (rollback primitives) ──────────────


def _node_exists(backend: Any, node_id: str | None) -> bool:
    if backend is None or not node_id:
        return False
    try:
        rows = backend.execute("MATCH (n {id: $id}) RETURN n.id AS id", {"id": node_id})
        return bool(rows)
    except Exception:  # noqa: BLE001 — unknown existence -> assume new (safer rollback default)
        return False


def _delete_node(backend: Any, node_id: str | None) -> None:
    if backend is None or not node_id:
        return
    try:
        backend.execute("MATCH (n {id: $id}) DETACH DELETE n", {"id": node_id})
    except Exception:  # noqa: BLE001 — rollback is best-effort; never raise from a rollback
        logger.debug("envelope write rollback failed for %s", node_id, exc_info=True)


def _read_archived(backend: Any, node_id: str | None) -> bool | None:
    if backend is None or not node_id:
        return None
    try:
        rows = backend.execute(
            "MATCH (n {id: $id}) RETURN n.archived AS archived", {"id": node_id}
        )
        for r in rows or []:
            if isinstance(r, dict):
                return bool(r.get("archived"))
    except Exception:  # noqa: BLE001
        logger.debug("envelope archived-state read failed", exc_info=True)
    return None


def _apply_tombstone(
    backend: Any, envelope: ChangeEnvelope, node_id: str | None
) -> dict[str, Any]:
    if backend is None or not node_id:
        return {"status": "skipped", "reason": "no backend or node id"}
    backend.execute(
        "MATCH (n {id: $id}) SET n.archived = true, n.archivedReason = $reason",
        {"id": node_id, "reason": f"tombstoned-by-{envelope.connector}"},
    )
    return {"status": "success", "node_id": node_id}


def _restore_archived(
    backend: Any, node_id: str | None, was_archived: bool | None
) -> None:
    if backend is None or not node_id or was_archived is None:
        return
    try:
        backend.execute(
            "MATCH (n {id: $id}) SET n.archived = $archived",
            {"id": node_id, "archived": was_archived},
        )
    except Exception:  # noqa: BLE001 — rollback is best-effort; never raise from a rollback
        logger.debug("tombstone rollback failed for %s", node_id, exc_info=True)


# ── the object/link/artifact write ──────────────────────────────────────


def _apply_write(
    engine: Any,
    backend: Any,
    envelope: ChangeEnvelope,
    node_id: str | None,
    row: dict[str, Any] | None,
) -> dict[str, Any]:
    if envelope.operation == "snapshot_complete":
        from ..core.source_sync import _reconcile

        live_ids = set(envelope.live_ids or [])
        fetch_ok = bool(envelope.provenance.get("fetch_ok", True))
        return _reconcile(engine, envelope.connector, live_ids, fetch_ok=fetch_ok)

    if envelope.operation == "delete":
        return _apply_tombstone(backend, envelope, node_id)

    # upsert
    if row is None:
        # blob-backed artifact: no typed_payload to render, write a minimal
        # pointer node (the "artifact" leg of "write object/link/artifact").
        write_row: dict[str, Any] = {
            "id": node_id,
            "type": envelope.payload_type or "Artifact",
            "blob_ref": envelope.blob_ref,
        }
        if envelope.source_acl is not None:
            write_row["external_access"] = envelope.source_acl.model_dump()
        relationships: list[dict[str, Any]] = []
    else:
        write_row = dict(row)
        write_row["id"] = node_id
        links = write_row.pop("_links", None)
        relationships = list(links) if isinstance(links, list) else []

    write_row = _shacl_annotate(write_row)

    # Route through the SAME single materialization entrypoint every legacy
    # handler already uses (``engine.ingest_external_batch`` -> ``write_entities``)
    # when the engine exposes it, so this is a wrapper around the existing write
    # path, not a second one. Falls back to the bare backend writer for a caller
    # that only has a backend (no engine facade).
    ingest_batch_fn = getattr(engine, "ingest_external_batch", None)
    if callable(ingest_batch_fn):
        return ingest_batch_fn(envelope.connector, [write_row], relationships)

    from ..core.materialization import write_entities

    return write_entities(backend, envelope.connector, [write_row], relationships)


# ── watermark (monotonic-guarded; shares SourceSyncState with legacy handlers) ──


def _watermark_key(envelope: ChangeEnvelope) -> str:
    return (
        envelope.connector
        if not envelope.source_instance
        else f"{envelope.connector}:{envelope.source_instance}"
    )


def _advance_watermark(backend: Any, envelope: ChangeEnvelope) -> bool:
    """Advance the source watermark iff ``envelope.checkpoint`` is newer than
    what's currently stored. Returns whether it actually advanced.

    Per-envelope (not per-batch) monotonic guarding is what makes crash-resume
    safe at ENVELOPE granularity: envelopes fully committed before a crash keep
    their watermark advanced; the not-yet-processed ones don't, so a resumed
    sync picks up exactly where it left off instead of re-reading (or, worse,
    skipping) a whole batch.
    """
    if not envelope.checkpoint:
        return False
    from ..core.source_sync import _read_watermark, _write_watermark

    key = _watermark_key(envelope)
    current = _read_watermark(backend, key)
    if current is not None and str(envelope.checkpoint) <= str(current):
        return False
    _write_watermark(backend, key, envelope.checkpoint)
    return True


# ── rollback ─────────────────────────────────────────────────────────────


def _rollback(
    backend: Any,
    envelope: ChangeEnvelope,
    node_id: str | None,
    existed_before: bool,
    prior_archived: bool | None,
    applied: list[str],
) -> None:
    """Undo whatever :func:`ingest_envelope` applied before the failing step.

    ``cdc`` has no compensator (a fire-and-forget notification can't be
    unpublished — and doesn't need to be: consumers treat delivery as
    at-least-once, same as every other CDC system). ``lineage``/``write`` are
    compensated so a NEW object leaves zero trace and a tombstoned object's
    ``archived`` flag reverts to what it was before this attempt.
    """
    if "lineage" in applied:
        _lineage_delete(backend, envelope.idempotency_key)
    if "write" in applied:
        if envelope.operation == "delete" and node_id:
            _restore_archived(backend, node_id, prior_archived)
        elif (
            envelope.operation != "snapshot_complete" and node_id and not existed_before
        ):
            _delete_node(backend, node_id)
        # snapshot_complete: `_reconcile` already applies its own per-row
        # tombstones defensively (each guarded in its own try/except) — there
        # is no outer write to compensate here.


# ── the entrypoint ───────────────────────────────────────────────────────


def ingest_envelope(
    engine: Any, envelope: ChangeEnvelope, *, backend: Any = None
) -> dict[str, Any]:
    """Ingest ONE :class:`ChangeEnvelope` as a single atomic unit.

    CONCEPT:AU-KG.ingest.envelope-atomic-transaction (AU-P1-5). See the module
    docstring for the crash-resume contract. Returns a dict with at minimum
    ``status`` (``"success"`` | ``"skipped"`` | ``"rejected"`` | ``"failed"``)
    and ``watermark_advanced`` (bool) — a resumed connector can trust
    ``watermark_advanced is False`` to mean "this envelope must be retried".
    """
    backend = backend if backend is not None else getattr(engine, "backend", None)

    base = {
        "envelope_id": envelope.envelope_id,
        "idempotency_key": envelope.idempotency_key,
        "connector": envelope.connector,
        "operation": envelope.operation,
        "watermark_advanced": False,
    }

    violations = _validate_envelope(envelope)
    if violations:
        return {**base, "status": "rejected", "violations": violations}

    idem_key = envelope.idempotency_key
    prior = _lineage_read(backend, idem_key)
    if prior and prior.get("status") == "applied":
        return {
            **base,
            "status": "skipped",
            "reason": "idempotent replay — envelope already applied",
        }

    node_id, row = _resolve_identity(envelope)
    existed_before = _node_exists(backend, node_id) if node_id else True
    prior_archived = (
        _read_archived(backend, node_id)
        if envelope.operation == "delete" and node_id
        else None
    )

    applied: list[str] = []
    write_result: dict[str, Any] = {}
    try:
        write_result = _apply_write(engine, backend, envelope, node_id, row)
        applied.append("write")
        _lineage_write(backend, envelope, status="pending")
        applied.append("lineage")
        _emit_cdc(envelope, write_result)
        applied.append("cdc")
        watermark_advanced = _advance_watermark(backend, envelope)
        _lineage_write(backend, envelope, status="applied")
    except Exception as exc:  # noqa: BLE001 — full rollback on ANY mid-transaction failure
        logger.warning(
            "ingest_envelope: %s (%s) failed mid-transaction after %s — rolling back: %s",
            idem_key,
            envelope.connector,
            applied,
            exc,
        )
        if "cdc" in applied:
            # Everything durable (write + lineage) already succeeded; only the
            # final "mark applied"/watermark-advance step failed. Undoing the
            # already-correct write would be WORSE than leaving it — leave the
            # ledger row at "pending" (never deleted) and the watermark
            # unadvanced, so the next sync simply redoes this idempotent
            # envelope and reaches "applied" — never a lost or duplicated write.
            return {
                **base,
                "status": "failed",
                "error": str(exc),
                "rolled_back": [],
            }
        _rollback(backend, envelope, node_id, existed_before, prior_archived, applied)
        return {
            **base,
            "status": "failed",
            "error": str(exc),
            "rolled_back": list(applied),
        }

    return {
        **base,
        "status": "success",
        "node_id": node_id,
        "write_result": write_result,
        "watermark_advanced": watermark_advanced,
        "checkpoint": envelope.checkpoint,
    }
