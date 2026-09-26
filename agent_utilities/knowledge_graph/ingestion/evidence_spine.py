"""AU adapter for engine-owned evidence fragments and graph projection.

Artifact/Fragment identity, graph vocabulary, and pure source fragmentation live
in ``epistemic_graph.ingestion``. This adapter commits the resulting graph slice
through EG ``ApplyChangeEnvelope`` and reads committed fragment rows.
"""

from __future__ import annotations

import logging
from typing import Any

from epistemic_graph.ingestion.evidence_model import (
    ARTIFACT_NODE_TYPE,
    ARTIFACT_OF_EDGE,
    FRAGMENT_KINDS,
    FRAGMENT_NODE_TYPE,
    FRAGMENT_OF_EDGE,
    HAS_ARTIFACT_EDGE,
    HAS_CHILD_FRAGMENT_EDGE,
    HAS_FRAGMENT_EDGE,
    NEXT_FRAGMENT_EDGE,
    PARENT_FRAGMENT_EDGE,
    Artifact,
    Fragment,
    FragmentKind,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ARTIFACT_NODE_TYPE",
    "FRAGMENT_NODE_TYPE",
    "HAS_FRAGMENT_EDGE",
    "FRAGMENT_OF_EDGE",
    "HAS_CHILD_FRAGMENT_EDGE",
    "PARENT_FRAGMENT_EDGE",
    "NEXT_FRAGMENT_EDGE",
    "HAS_ARTIFACT_EDGE",
    "ARTIFACT_OF_EDGE",
    "FRAGMENT_KINDS",
    "ingest_artifact",
    "load_fragments",
    "FragmentKind",
    "Artifact",
    "Fragment",
]

# ── Ingest wiring ────────────────────────────────────────────────────────────


def ingest_artifact(
    engine: Any,
    artifact: Artifact,
    *,
    document_id: str = "",
    checkpoint: str | None = None,
) -> dict[str, Any]:
    """Commit an artifact and its whole fragment tree atomically.

    Rides the EXISTING ingestion path — :func:`..ingestion.envelope_ingest.ingest_graph_slice`,
    which wraps the slice in one ``ChangeEnvelope`` and one native
    ``ApplyChangeEnvelope`` transaction — rather than opening a second write
    route.  A half-written spine (an artifact whose fragments did not land, or
    fragments orphaned from their artifact) is not a state any reader should
    ever have to handle, so it is not a state this can produce.

    ``version_field="content_hash"`` makes re-ingesting an unchanged artifact
    dedupe on the envelope's idempotency ledger instead of rewriting the slice.
    """
    from .envelope_ingest import ingest_graph_slice

    entities, relationships = artifact.to_graph_slice(document_id=document_id)
    return ingest_graph_slice(
        engine,
        artifact.connector,
        entities,
        relationships,
        source_instance=artifact.source_instance,
        checkpoint=checkpoint,
        version_field="content_hash",
    )


def _fragment_read_query(
    *, artifact_id: str, document_id: str
) -> tuple[str, dict[str, str]]:
    """Build the one projection shared by both fragment lookup keys."""
    projection = (
        "RETURN f.id AS id, f.artifact_id AS artifact_id, "
        "f.fragment_kind AS fragment_kind, f.address AS address, "
        "f.text AS text, f.content_hash AS content_hash, "
        "f.ordinal AS ordinal, f.sequence AS sequence, f.depth AS depth, "
        "f.parent_fragment_id AS parent_fragment_id, "
        "f.char_start AS char_start, f.char_end AS char_end, "
        "f.label AS label, f.locus_kind AS locus_kind"
    )
    if artifact_id:
        return (
            f"MATCH (f:{FRAGMENT_NODE_TYPE}) WHERE f.artifact_id = $key {projection}",
            {"key": artifact_id},
        )
    return (
        f"MATCH (d:Document)-[:{HAS_ARTIFACT_EDGE}]->(a:{ARTIFACT_NODE_TYPE})"
        f"-[:{HAS_FRAGMENT_EDGE}]->(f:{FRAGMENT_NODE_TYPE}) "
        f"WHERE d.id = $key {projection}",
        {"key": document_id},
    )


def _read_fragment_rows(
    engine: Any, cypher: str, params: dict[str, str], *, target: str
) -> list[Any]:
    """Read fragment rows through the engine, with the backend fallback."""
    try:
        run = getattr(engine, "query_cypher", None)
        if callable(run):
            return list(run(cypher, params) or [])
        backend = getattr(engine, "backend", None)
        execute = getattr(backend, "execute", None)
        return list(execute(cypher, params) or []) if callable(execute) else []
    except Exception as exc:  # noqa: BLE001 — advisory reads report no stored spine
        logger.debug("fragment spine read failed for %r: %s", target, exc)
        return []


def _fragment_row_value(row: dict[str, Any], key: str, default: Any) -> Any:
    """Apply the stored-row defaults used by the materialized spine reader."""
    return row.get(key) or default


def _fragment_from_row(row: Any) -> Fragment | None:
    """Rehydrate one stored row, logging and dropping corrupt evidence."""
    if not isinstance(row, dict) or not row.get("address"):
        return None
    try:
        return Fragment(
            fragment_id=str(_fragment_row_value(row, "id", "")),
            artifact_id=str(_fragment_row_value(row, "artifact_id", "")),
            kind=str(_fragment_row_value(row, "fragment_kind", "span")),  # type: ignore[arg-type]
            path=tuple(str(row["address"]).split("/")),
            text=str(_fragment_row_value(row, "text", "")),
            content_hash=str(_fragment_row_value(row, "content_hash", "")),
            ordinal=int(_fragment_row_value(row, "ordinal", 0)),
            sequence=int(_fragment_row_value(row, "sequence", 0)),
            depth=int(_fragment_row_value(row, "depth", 0)),
            parent_fragment_id=_fragment_row_value(row, "parent_fragment_id", None),
            char_start=int(_fragment_row_value(row, "char_start", -1)),
            char_end=int(_fragment_row_value(row, "char_end", -1)),
            label=str(_fragment_row_value(row, "label", "")),
            locus_kind=str(_fragment_row_value(row, "locus_kind", "document_span")),
        )
    except (ValueError, TypeError) as exc:
        # A stored row that fails Fragment's own invariants (a mismatched
        # address/id pair, a missing digest) is CORRUPT evidence.  Dropping
        # it is right — returning it would let a citation resolve against a
        # fragment whose address no longer proves anything — but it is never
        # dropped silently.
        logger.warning("discarding corrupt Fragment row %r: %s", row.get("id"), exc)
        return None


def load_fragments(
    engine: Any, *, artifact_id: str = "", document_id: str = ""
) -> tuple[Fragment, ...]:
    """Read a materialized fragment spine back out of the graph, in document order.

    Either key works: ``artifact_id`` addresses the source object directly,
    ``document_id`` resolves through the ``HAS_ARTIFACT`` join the
    ``DocumentProcessor`` path writes.  Returns an empty tuple (never raises)
    when nothing is stored — an absent spine is a legitimate answer, and a
    citation check must not fail closed into "cannot tell".
    """
    if engine is None or not (artifact_id or document_id):
        return ()
    cypher, params = _fragment_read_query(
        artifact_id=artifact_id, document_id=document_id
    )
    rows = _read_fragment_rows(
        engine, cypher, params, target=artifact_id or document_id
    )
    fragments: list[Fragment] = []
    for row in rows:
        fragment = _fragment_from_row(row)
        if fragment is not None:
            fragments.append(fragment)
    fragments.sort(key=lambda f: f.sequence)
    return tuple(fragments)
