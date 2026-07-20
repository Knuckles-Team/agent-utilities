# CONCEPT:AU-KG.ontology.external-graph-profiling - Profile / imprint / ontology-map a registered external graph.
"""Profile a registered external graph connection, imprint its schema into our KG,
and map its labels onto our ontology.

Extends the KG-2.63 multi-connection registry: once an external Neo4j / FalkorDB /
Postgres-AGE / Ladybug graph is registered (``graph_configure add_connection``),
this module lets an agent *discover what's in it and how to use it* without
re-introspecting every time:

* **profile** — read-only schema introspection (labels, relationship types,
  property keys, per-label counts + sample property shapes). Backend-portable:
  prefers the ``db.*`` procedures (Neo4j/FalkorDB), degrades to a bounded sampled
  scan where they're unavailable.
* **map** — deterministically map each external label onto the closest class in
  our ontology vocabulary (interfaces + our KG's own node types) by name; the
  unmatched ones are flagged ``novel`` (candidates for a new ontology class).
Raw discovery and mappings are stored only by the governed workflow in
:mod:`knowledge_graph.ingestion.external_graph_schema`; public/catalog state contains
only digests, counts, and pseudonyms.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


def _rows(result: Any) -> list[dict[str, Any]]:
    """Normalize ``engine.query_cypher`` output to a list of dict rows.

    The query surface may return a list of dicts, a JSON string, or a
    ``{"result": "[...]"}`` wrapper — accept all three.
    """
    if result is None:
        return []
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            return []
    if isinstance(result, dict):
        for k in ("result", "rows", "data"):
            if k in result:
                return _rows(result[k])
        return [result]
    if isinstance(result, list):
        return [r if isinstance(r, dict) else {"value": r} for r in result]
    return []


def _q(engine: Any, cypher: str) -> list[dict[str, Any]]:
    """Run one read-only introspection query, returning [] on any failure."""
    try:
        return _rows(engine.query_cypher(cypher))
    except Exception as exc:  # noqa: BLE001 - query material must not enter logs
        logger.debug("introspection query failed: error_type=%s", type(exc).__name__)
        return []


def _first(row: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    # fall back to the sole value if the row is single-column
    if len(row) == 1:
        return next(iter(row.values()))
    return None


def profile_connection(
    engine: Any, *, name: str = "", max_labels: int = 200, sample_props: bool = True
) -> dict[str, Any]:
    """Introspect an external graph's schema. Read-only, bounded, and transient.

    Do not persist or log this raw return. Public GraphOS actions use the
    metadata-only summary from ``discover_external_schema`` instead.
    """
    # ── labels ────────────────────────────────────────────────────────────
    labels: list[str] = []
    for r in _q(engine, "CALL db.labels() YIELD label RETURN label"):
        v = _first(r, "label")
        if v:
            labels.append(str(v))
    if not labels:  # backend without db.labels() → sample from nodes
        for r in _q(
            engine,
            "MATCH (n) WITH labels(n) AS ls UNWIND ls AS l "
            "RETURN DISTINCT l AS label LIMIT 200",
        ):
            v = _first(r, "label", "l")
            if v:
                labels.append(str(v))
    labels = sorted(set(labels))[:max_labels]

    # ── relationship types & property keys ────────────────────────────────
    rels = sorted(
        {
            str(_first(r, "relationshipType"))
            for r in _q(
                engine,
                "CALL db.relationshipTypes() YIELD relationshipType "
                "RETURN relationshipType",
            )
            if _first(r, "relationshipType")
        }
    )
    pkeys = sorted(
        {
            str(_first(r, "propertyKey"))
            for r in _q(
                engine, "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey"
            )
            if _first(r, "propertyKey")
        }
    )

    # ── per-label count + sample property keys ────────────────────────────
    per_label: dict[str, Any] = {}
    for lbl in labels:
        crows = _q(engine, f"MATCH (n:`{lbl}`) RETURN count(n) AS c")
        count = _first(crows[0], "c", "count(n)") if crows else None
        props: list[str] = []
        if sample_props:
            srows = _q(engine, f"MATCH (n:`{lbl}`) RETURN keys(n) AS k LIMIT 1")
            if srows:
                k = _first(srows[0], "k", "keys(n)")
                if isinstance(k, list):
                    props = [str(x) for x in k]
                elif isinstance(k, str):
                    try:
                        props = [str(x) for x in json.loads(k)]
                    except Exception:
                        props = []
        per_label[lbl] = {"count": count, "sample_property_keys": props}

    trows = _q(engine, "MATCH (n) RETURN count(n) AS c")
    total = _first(trows[0], "c") if trows else None

    return {
        "connection": name,
        "labels": labels,
        "label_count": len(labels),
        "relationship_types": rels,
        "property_keys": pkeys,
        "per_label": per_label,
        "total_nodes": total,
        "profiled_at": time.time(),
    }


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def map_labels_to_ontology(
    labels: list[str], our_classes: list[str]
) -> list[dict[str, Any]]:
    """Map each external label to the best ontology class by name (deterministic).

    Strategy per label: exact (normalized) → singular/plural → fuzzy (difflib,
    cutoff 0.85) → ``novel`` (no match; a candidate new ontology class).
    """
    norm_map: dict[str, str] = {}
    for c in our_classes:
        norm_map.setdefault(_norm(c), c)
    norm_keys = list(norm_map)

    out: list[dict[str, Any]] = []
    for lbl in labels:
        nl = _norm(lbl)
        if not nl:
            continue
        if nl in norm_map:
            out.append(_m(lbl, norm_map[nl], "exact", 1.0))
            continue
        alt = nl[:-1] if nl.endswith("s") else nl + "s"
        if alt in norm_map:
            out.append(_m(lbl, norm_map[alt], "plural", 0.9))
            continue
        close = difflib.get_close_matches(nl, norm_keys, n=1, cutoff=0.85)
        if close:
            out.append(_m(lbl, norm_map[close[0]], "fuzzy", 0.8))
        else:
            out.append(_m(lbl, None, "novel", 0.0))
    return out


def _m(label: str, mapped: str | None, method: str, conf: float) -> dict[str, Any]:
    return {
        "external_label": label,
        "mapped_to": mapped,
        "method": method,
        "confidence": conf,
    }


def _our_ontology_vocabulary(
    authority_engine: Any, interface_names: list[str] | None
) -> list[str]:
    """Our mapping target vocabulary: ontology interfaces + our KG node types."""
    vocab: list[str] = list(interface_names or [])
    if interface_names is None:
        try:
            from agent_utilities.knowledge_graph.ontology.interfaces import (
                DEFAULT_INTERFACE_REGISTRY,
            )

            vocab.extend(i.name for i in DEFAULT_INTERFACE_REGISTRY.list_interfaces())
        except Exception:  # noqa: BLE001 — interfaces optional
            pass
    if authority_engine is not None:
        for r in _q(
            authority_engine, "MATCH (n) RETURN DISTINCT n.node_type AS t LIMIT 500"
        ):
            t = _first(r, "t")
            if t:
                vocab.append(str(t))
    return sorted({v for v in vocab if v})
