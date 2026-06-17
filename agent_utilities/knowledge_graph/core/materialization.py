"""The one KG materialization core (CONCEPT:KG-2.9).

A single write path for every connector. Both ingestion adapters delegate here:

* ``IngestionMixin.ingest_external_batch`` (dict entities, engine method) — the
  hub-and-spoke / delta-handler path (hydration, leanix, gitlab, …).
* ``enrichment.registry.write_batch`` (typed ``ExtractionBatch``) — the
  materialize / extractor path (camunda, egeria, okta, finance, …).

So provenance stamping, the content-hash write-delta, typed-label UNWIND
batching, and the Ladybug iterative fallback are implemented ONCE — there is no
second, thinner writer that silently misses delta or batching.

The two schema-aware helpers (label normalization + the SET-clause column filter)
live on the composed engine and have many non-write callers, so they are *not*
moved here; instead ``write_entities`` accepts them as injected callables. The
engine adapter passes its own (so its path is byte-identical to before); the
backend-only ``write_batch`` adapter uses the faithful free defaults below.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

# Backends whose writes must filter properties to declared columns (mirror of the
# engine's ``_SCHEMA_BACKED``) so the free default SET clause is correct on them.
_SCHEMA_BACKED = {"LadybugBackend", "PostgreSQLBackend", "TieredGraphBackend"}

# Properties that must NOT contribute to the content hash: the hash itself, and
# volatile provenance/observation timestamps that change every ingest.
_VOLATILE_HASH_KEYS = frozenset(
    {
        "content_hash",
        "_ingested_at",
        "ingested_at",
        "validFrom",
        "validTo",
        "observedAt",
        "lastSeen",
        "syncedAt",
        "_kg_ts",
    }
)

_LABEL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# ── content-hash write-delta ─────────────────────────────────────────────────


def content_hash(row: dict[str, Any]) -> str:
    """Stable SHA-256 over an entity's semantic properties (id + volatile
    timestamps excluded), so an unchanged upstream record hashes identically
    across ingests."""
    payload = {
        k: v for k, v in row.items() if k != "id" and k not in _VOLATILE_HASH_KEYS
    }
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def filter_unchanged(
    backend: Any, entities: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Stamp ``content_hash`` on each entity and drop the ones whose stored hash
    is unchanged (generic write-layer delta). Best-effort: any backend that can't
    answer the batched prefetch yields a full write (correct, just not deduped)."""
    if not entities or backend is None:
        return entities, 0
    for row in entities:
        if row.get("id") is not None:
            row["content_hash"] = content_hash(row)
    ids = [str(r["id"]) for r in entities if r.get("id") is not None]
    if not ids:
        return entities, 0
    stored: dict[str, str] = {}
    try:
        rows = backend.execute(
            "MATCH (n) WHERE n.id IN $ids RETURN n.id AS id, n.content_hash AS h",
            {"ids": ids},
        )
        for r in rows or []:
            if isinstance(r, dict) and r.get("id") is not None and r.get("h"):
                stored[str(r["id"])] = str(r["h"])
    except Exception:  # noqa: BLE001 — prefetch is an optimization, never fatal
        logger.debug("content-hash prefetch failed; writing full batch", exc_info=True)
        return entities, 0
    if not stored:
        return entities, 0
    changed = [
        r
        for r in entities
        if r.get("id") is None or stored.get(str(r["id"])) != r.get("content_hash")
    ]
    return changed, len(entities) - len(changed)


# ── free schema helpers (defaults for the backend-only write_batch path) ─────


def normalize_label(label: str) -> str:
    """Canonical case for a label from the schema (free default mirror of the
    engine's ``_normalize_label``)."""
    if not label:
        return label
    try:
        from ...models.schema_definition import SCHEMA

        for node_def in SCHEMA.nodes:
            if node_def.name.lower() == label.lower():
                return node_def.name
    except ImportError:
        pass
    return label


def schema_valid_keys(backend: Any, label: str | None) -> set[str] | None:
    """Declared columns for ``label`` on a schema-backed backend, else None (free
    default mirror of the engine's ``_schema_valid_keys``)."""
    if backend is None or backend.__class__.__name__ not in _SCHEMA_BACKED or not label:
        return None
    from ...models.schema_definition import GENERIC_NODE_COLUMNS, SCHEMA

    for node in SCHEMA.nodes:
        if node.name == label:
            return set(node.columns.keys())
    if backend.__class__.__name__ == "LadybugBackend":
        return set(GENERIC_NODE_COLUMNS)
    return None


def default_set_clause(
    data: dict[str, Any], backend: Any, alias: str = "n", label: str | None = None
) -> str:
    """Free default SET clause (mirror of the engine's ``_get_set_clause``) for the
    backend-only ``write_batch`` path."""
    if label:
        label = normalize_label(label)
    if alias == "r" and backend and backend.__class__.__name__ == "LadybugBackend":
        return ""
    valid_keys = schema_valid_keys(backend, label)
    sets = []
    for k in data:
        if k == "id":
            continue
        if valid_keys is not None and k not in valid_keys:
            continue
        sets.append(f"{alias}.`{k}` = ${k}")
    return " SET " + ", ".join(sets) if sets else ""


def safe_label(
    raw: Any,
    *,
    fallback: str = "DomainEntity",
    normalize: Callable[[str], str] | None = None,
) -> str:
    """A Cypher-safe label from a node/edge type: normalize via the schema, then
    require a bare identifier so it can be inlined into MERGE; else the generic
    fallback superclass (the real type survives as the ``type`` property)."""
    norm = normalize or normalize_label
    label = norm(str(raw or "")) if raw else str(raw or "")
    return label if _LABEL_RE.fullmatch(label) else fallback


def group_by_label(
    entities: list[dict[str, Any]], *, normalize: Callable[[str], str] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Bucket entities by their normalized, label-safe node type for per-type
    UNWIND MERGE (fallback ``DomainEntity``)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in entities:
        groups.setdefault(safe_label(row.get("type"), normalize=normalize), []).append(
            row
        )
    return groups


def group_by_rel(
    relationships: list[dict[str, Any]],
    *,
    normalize: Callable[[str], str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Bucket edges by their label-safe rel type (fallback ``EXTERNAL_LINK``)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in relationships:
        groups.setdefault(
            safe_label(row.get("type"), fallback="EXTERNAL_LINK", normalize=normalize),
            [],
        ).append(row)
    return groups


# ── the one writer ───────────────────────────────────────────────────────────


def write_entities(
    backend: Any,
    domain: str,
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]] | None = None,
    *,
    delta: bool = True,
    set_clause_fn: Callable[..., str] | None = None,
    normalize_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Persist standardized entity/relationship dicts to ``backend`` — the single
    materialization path (CONCEPT:KG-2.9).

    Stamps the shared provenance contract, applies the content-hash write-delta
    (unless ``KG_WRITE_DELTA=0`` or ``delta=False``), then writes via per-type
    UNWIND MERGE (typed labels preserved) — or the iterative Ladybug fallback.

    ``set_clause_fn``/``normalize_fn`` let the composed engine inject its own
    schema-aware helpers so its path is byte-identical; the backend-only caller
    omits them and gets the faithful free defaults above. Returns
    ``{status, nodes, edges, skipped_unchanged, backend}``.
    """
    from ..enrichment.provenance import stamp_source

    rels = relationships or []
    for row in entities:
        stamp_source(row, domain)
    for row in rels:
        stamp_source(row, domain)

    # Backend capability tiers — the one writer adapts to what the backend offers:
    #   1. execute_batch (+ not Ladybug) → high-throughput UNWIND MERGE (the norm).
    #   2. Ladybug / execute-only       → per-row execute MERGE (Kuzu has no UNWIND).
    #   3. add_node only                → lowest-common-denominator add_node/add_edge
    #      (node-only / SPARQL-style backends — the original write_batch path).
    is_ladybug = backend.__class__.__name__ == "LadybugBackend"
    has_execute = callable(getattr(backend, "execute", None))
    has_execute_batch = callable(getattr(backend, "execute_batch", None))

    # The content-hash delta needs ``execute`` for its prefetch; skip it on
    # add_node-only backends (they keep the original full-write behavior).
    skipped_unchanged = 0
    if delta and has_execute and str(setting("KG_WRITE_DELTA", "1")) != "0":
        entities, skipped_unchanged = filter_unchanged(backend, entities)
        if not entities and not rels:
            return {
                "status": "success",
                "nodes": 0,
                "edges": 0,
                "skipped_unchanged": skipped_unchanged,
                "backend": True,
            }

    if set_clause_fn is None:

        def set_clause_fn(  # noqa: ANN001
            data: dict[str, Any], alias: str = "n", label: str | None = None
        ) -> str:
            return default_set_clause(data, backend, alias, label)

    edges = 0

    if has_execute_batch and not is_ladybug:
        # Tier 1 — Neo4j/FalkorDB/AGE/epistemic high-throughput UNWIND batching,
        # grouped by the REAL node/rel type so each entity keeps its specific label.
        for label, rows in group_by_label(entities, normalize=normalize_fn).items():
            keys = sorted({k for row in rows for k in row} - {"id"})
            clause = (
                "SET " + ", ".join([f"n.`{k}` = row.`{k}`" for k in keys])
                if keys
                else ""
            )
            backend.execute_batch(
                f"UNWIND $batch AS row MERGE (n:{label} {{id: row.id}}) {clause}".rstrip(),
                rows,
            )
        for rel, rows in group_by_rel(rels, normalize=normalize_fn).items():
            r_keys = sorted(
                {k for row in rows for k in row} - {"source", "target", "type"}
            )
            clause = (
                "SET " + ", ".join([f"r.`{k}` = row.`{k}`" for k in r_keys])
                if r_keys
                else ""
            )
            backend.execute_batch(
                f"UNWIND $batch AS row MATCH (s {{id: row.source}}) "
                f"MATCH (t {{id: row.target}}) MERGE (s)-[r:{rel}]->(t) {clause}".rstrip(),
                rows,
            )
            edges += len(rows)
    elif has_execute:
        # Tier 2 — Ladybug (Kuzu) / execute-only backends: per-row MERGE.
        for row in entities:
            node_type = safe_label(row.get("type"), normalize=normalize_fn)
            clause = set_clause_fn(row, "n", node_type)
            backend.execute(f"MERGE (n:{node_type} {{id: $id}}){clause}", row)
        for row in rels:
            # Use the REAL rel type so Kuzu builds a typed table, not a generic
            # collapsed edge; the backend binds the endpoints' rel-pair (KG-2.74).
            rtype = safe_label(row.get("type") or "RELATED", normalize=normalize_fn)
            clause = set_clause_fn(row, "r", None)
            backend.execute(
                f"MATCH (s {{id: $source}}) MATCH (t {{id: $target}}) "
                f"MERGE (s)-[r:{rtype}]->(t){clause}",
                row,
            )
            edges += 1
    else:
        # Tier 3 — lowest-common-denominator: add_node/add_edge (node-only and
        # SPARQL-style backends). add_edge is optional (guarded), exactly as the
        # original write_batch did, so node-only stores still take the nodes.
        for row in entities:
            eid = row.get("id")
            if eid is None:
                continue
            props = {k: v for k, v in row.items() if k not in ("id", "type")}
            backend.add_node(eid, type=row.get("type"), **props)
        add_edge = getattr(backend, "add_edge", None)
        if callable(add_edge):
            for row in rels:
                props = {
                    k: v
                    for k, v in row.items()
                    if k not in ("source", "target", "type")
                }
                add_edge(
                    row.get("source"),
                    row.get("target"),
                    rel_type=row.get("type"),
                    **props,
                )
                edges += 1

    return {
        "status": "success",
        "nodes": len(entities),
        "edges": edges,
        "skipped_unchanged": skipped_unchanged,
        "backend": True,
    }
