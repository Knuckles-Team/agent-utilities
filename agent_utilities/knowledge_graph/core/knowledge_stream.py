#!/usr/bin/python
from __future__ import annotations

"""``Method::KnowledgeStream`` consumer — the bulk Arrow-columnar streaming currency
(report §2 Query seam / §9 #3, CONCEPT:AU-KG.query.knowledge-stream-consumer).

The engine ships ``Method::KnowledgeStream`` (typed AU-side scaffolding already exists
in ``epistemic_graph.client``: ``KnowledgeStreamBatch``, ``projection="arrow_ipc_v1"``,
across the Graph/SQL/RDF/Vector/TimeSeries/Job/CrossModal families), but nothing under
``agent_utilities/`` called it — this module is the missing consumer.

**What the wire actually carries (verified against the engine's own Rust source,
not guessed).** A ``KnowledgeStreamBatch`` is the Arrow-columnar projection of a
``KnowledgeBatch`` (``epistemic-graph/crates/eg-plan/src/knowledge_batch.rs``):
``id`` (a keyed-opaque, privacy-safe reference — NOT the literal node id string),
``kind``, one ``score_<name>`` column per requested score, ``confidence``,
``evidence_kind``/``evidence_refs_json``, the bitemporal ``valid_from``/``valid_until``/
``tx_from``/``tx_to`` window, ``source_refs``/``policy_labels``/``transformation_ids``/
``proof_ids``/``alternative_ids``/``contradiction_ids`` (string lists), and a lazy
``blob_handle``/``has_payload`` pair. There is **no node-properties column** — this is
the same "epistemic columns" currency :mod:`.epistemic_row` already models
(CONCEPT:AU-KB-CURRENCY), not a bulk node-data export. So this module is the bulk,
label-scoped, cursor-resumable sibling of ``explain_provenance_by_ids`` (which is
id-seeded and row-major MessagePack) — useful for a bulk confidence/evidence sweep
over MANY-or-ALL nodes of one label (compliance rollups, claim/evidence review, a
future distillation export), never a substitute for a properties-bearing read like
``get_nodes_by_label``.

**Never a hard dependency.** Decoding ``projection=arrow_ipc_v1`` bytes needs
``pyarrow`` — an optional, lazily-imported dependency per this repo's Dependency
discipline (never added to core `dependencies`). :func:`stream_graph_confidence`
feature-detects BOTH the engine's ``.knowledge`` streaming surface and a local
``pyarrow`` install before returning a generator; either missing degrades to
``None`` (never raises) so a caller falls back to its existing bulk-read path.
"""

import logging
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "KNOWLEDGE_STREAM_MAX_BATCH_SIZE",
    "available",
    "pull_knowledge_stream",
    "stream_graph_confidence",
]

#: Mirrors the engine's own clamp (``request.batch_size.clamp(1, 65_536)``,
#: ``knowledge_stream.rs::serve_execution``).
KNOWLEDGE_STREAM_MAX_BATCH_SIZE = 65_536

#: Default page size — small enough to keep one Arrow frame modest, large enough
#: that a bulk sweep over thousands of rows is single-digit round trips.
_DEFAULT_BATCH_SIZE = 512

#: The ``score_<name>``/list-typed column names the engine's Arrow schema always
#: emits (``eg-plan::knowledge_batch::arrow_schema``) — read straight off the
#: decoded table so this stays correct even if the server adds a new named score.
_LIST_COLUMNS: tuple[str, ...] = (
    "evidence_refs_json",
    "source_refs",
    "policy_labels",
    "transformation_ids",
    "proof_ids",
    "alternative_ids",
    "contradiction_ids",
)


def _pyarrow() -> Any:
    """Lazy, optional ``pyarrow`` import — ``None`` when it isn't installed.

    A standalone accessor (rather than an inline ``try/import`` at each call
    site) so a test can monkeypatch it with a minimal fake to exercise the
    cursor-loop/dispatch logic without a real ``pyarrow`` wheel installed.
    """
    try:
        import pyarrow  # noqa: F401 (imported for its side effect: presence check)
        import pyarrow.ipc  # noqa: F401
    except ImportError:
        return None
    import pyarrow

    return pyarrow


def _resolve_knowledge_client(compute: Any) -> Any | None:
    """Resolve the sync ``.knowledge`` sub-client (``KnowledgeStreamClient``'s
    sync-wrapped twin) off ``compute`` (a ``GraphComputeEngine`` or anything
    exposing the same ``._client`` shape). ``None`` when unreachable — an
    older engine build without the surface, or a non-engine backend."""
    client = getattr(compute, "_client", None)
    if client is None:
        # Accept a facade-ish object exposing `.graph` -> the compute engine
        # (mirrors `graph.reactive.engine_subscription.resolve_streaming`).
        graph = getattr(compute, "graph", None)
        client = getattr(graph, "_client", None) if graph is not None else None
    if client is None:
        return None
    knowledge = getattr(client, "knowledge", None)
    if knowledge is None or not callable(getattr(knowledge, "pull", None)):
        return None
    return knowledge


def available(compute: Any) -> bool:
    """``True`` iff both a live engine ``.knowledge`` surface AND ``pyarrow`` are
    reachable — the exact precondition :func:`pull_knowledge_stream` needs to
    return a real generator instead of degrading to ``None``."""
    return _resolve_knowledge_client(compute) is not None and _pyarrow() is not None


def _row_from_arrow_dict(
    row: dict[str, Any], score_columns: tuple[str, ...]
) -> dict[str, Any]:
    """One decoded Arrow row -> the honest, wire-shaped Python row.

    Field-for-field off ``eg-plan::knowledge_batch::arrow_schema`` (see the
    module docstring) — never fabricates a field the wire didn't carry.
    """
    return {
        "id": row.get("id", ""),
        "kind": row.get("kind") or "",
        "scores": {name[len("score_") :]: row.get(name) for name in score_columns},
        "confidence": row.get("confidence", 1.0),
        "evidence_kind": row.get("evidence_kind"),
        "evidence_refs_json": list(row.get("evidence_refs_json") or []),
        "valid_time": (row.get("valid_from"), row.get("valid_until")),
        "tx_time": (row.get("tx_from"), row.get("tx_to")),
        "source_refs": list(row.get("source_refs") or []),
        "policy_labels": list(row.get("policy_labels") or []),
        "transformation_ids": list(row.get("transformation_ids") or []),
        "proof_ids": list(row.get("proof_ids") or []),
        "alternative_ids": list(row.get("alternative_ids") or []),
        "contradiction_ids": list(row.get("contradiction_ids") or []),
        "blob_handle": row.get("blob_handle"),
        "has_payload": bool(row.get("has_payload")),
    }


def _decode_batch(pa: Any, batch: dict[str, Any]) -> list[dict[str, Any]]:
    """Decode one ``KnowledgeStreamBatch``'s ``arrow_ipc_v1`` payload into rows."""
    payload = batch.get("payload") if isinstance(batch, dict) else None
    if not payload:
        return []
    table = pa.ipc.open_stream(payload).read_all()
    score_columns = tuple(c for c in table.column_names if c.startswith("score_"))
    return [_row_from_arrow_dict(r, score_columns) for r in table.to_pylist()]


def pull_knowledge_stream(
    compute: Any,
    query: dict[str, Any],
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> Iterator[dict[str, Any]] | None:
    """Stream every row a ``KnowledgeStreamQuery`` matches, one Arrow page at a
    time, resuming via the engine's own integrity-bound cursor.

    ``query`` is one of the seven ``KnowledgeStreamQuery`` family dicts
    (``{"family": "graph", "label": ..., "limit": ...}``, ``{"family": "sql",
    ...}``, ...) — see ``epistemic_graph.client``'s ``KnowledgeGraphQuery``/
    ``KnowledgeSqlQuery``/... TypedDicts for the exact per-family shape.
    ``batch_size`` bounds each Arrow page (the engine clamps to
    :data:`KNOWLEDGE_STREAM_MAX_BATCH_SIZE`); any total-result cap belongs
    IN the query itself (e.g. the "graph" family's own ``limit``), not here.

    Returns ``None`` — never raises, never returns a generator that raises
    on first use — when the engine has no ``.knowledge`` streaming surface or
    ``pyarrow`` isn't installed, so a caller can synchronously decide to fall
    back to its existing (non-streaming) bulk-read path. Once returned, the
    generator itself degrades to simply stopping (logged at debug) on a
    mid-stream RPC error — a caller sees a short, honest prefix rather than a
    crash.
    """
    knowledge = _resolve_knowledge_client(compute)
    if knowledge is None:
        return None
    pa = _pyarrow()
    if pa is None:
        return None
    return _stream(knowledge, pa, dict(query), batch_size=batch_size)


def _stream(
    knowledge: Any, pa: Any, query: dict[str, Any], *, batch_size: int
) -> Iterator[dict[str, Any]]:
    cursor: dict[str, Any] | None = None
    page_size = max(1, min(int(batch_size), KNOWLEDGE_STREAM_MAX_BATCH_SIZE))
    while True:
        try:
            batch = knowledge.pull(query, batch_size=page_size, cursor=cursor)
        except Exception as exc:  # noqa: BLE001 — a stream hiccup ends the stream, never raises
            logger.debug("knowledge_stream: pull failed for %r: %s", query, exc)
            return
        if not isinstance(batch, dict):
            return
        try:
            rows = _decode_batch(pa, batch)
        except Exception as exc:  # noqa: BLE001 — a malformed page ends the stream
            logger.debug("knowledge_stream: Arrow decode failed for %r: %s", query, exc)
            return
        yield from rows
        cursor = batch.get("cursor")
        if not isinstance(cursor, dict) or cursor.get("exhausted"):
            return


def stream_graph_confidence(
    compute: Any,
    label: str,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    limit: int = 0,
) -> Iterator[dict[str, Any]] | None:
    """Bulk, cursor-resumable confidence/evidence sweep over up to ``limit`` nodes
    of ``label`` (``limit=0`` — mirroring :meth:`GraphComputeEngine.get_nodes_by_label`'s
    own convention — means uncapped), via the ``"graph"`` KnowledgeStream family.

    Each yielded row is the dict :func:`_row_from_arrow_dict` shapes — ``id`` is a
    keyed-opaque per-request reference (privacy-safe; NOT the literal node id, see
    the module docstring), ``confidence``/``scores``/evidence+provenance columns are
    real engine-computed values. Use this when a caller wants a POPULATION-level
    epistemic signal for a label (contested rate, confidence distribution, a bulk
    evidence export) — never as a replacement for a properties-bearing bulk read.

    ``None`` when no engine streaming surface or ``pyarrow`` is reachable — see
    :func:`pull_knowledge_stream`.
    """
    query = {"family": "graph", "label": str(label), "limit": int(limit)}
    return pull_knowledge_stream(compute, query, batch_size=batch_size)
