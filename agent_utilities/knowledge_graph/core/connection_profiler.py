# CONCEPT:KG-2.63 - Profile / imprint / ontology-map a registered external graph.
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
* **imprint** — write a single ``ExternalGraphReference`` catalog node into our
  authority KG carrying the schema + mappings (no credentials), so the foreign
  graph becomes self-describing to the rest of the system. The schema is stored as
  a nested property — which now mirrors losslessly thanks to the KG-2.74
  Cypher-property coercion fix.
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
    except Exception as e:  # noqa: BLE001 — introspection is best-effort per query
        logger.debug("introspection query failed (%.60s): %s", cypher, e)
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
    """Introspect an external graph's schema. Read-only and bounded."""
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
            authority_engine, "MATCH (n) RETURN DISTINCT n.type AS t LIMIT 500"
        ):
            t = _first(r, "t")
            if t:
                vocab.append(str(t))
    return sorted({v for v in vocab if v})


def profile_and_imprint(
    external_engine: Any,
    *,
    name: str,
    spec_summary: dict[str, Any] | None = None,
    authority_engine: Any = None,
    interface_names: list[str] | None = None,
    max_labels: int = 200,
) -> dict[str, Any]:
    """Profile ``name``, map its labels to our ontology, and imprint a catalog node.

    Returns the profile + mappings; writes one ``ExternalGraphReference`` node into
    the authority KG (credentials are never stored — only the redacted summary).
    """
    spec_summary = spec_summary or {}
    profile = profile_connection(external_engine, name=name, max_labels=max_labels)
    vocab = _our_ontology_vocabulary(authority_engine, interface_names)
    mappings = map_labels_to_ontology(profile["labels"], vocab)
    mapped = sum(1 for m in mappings if m["mapped_to"])
    novel = sum(1 for m in mappings if not m["mapped_to"])

    node_id = f"extgraph:{name}"
    props = {
        "name": name,
        "backend": spec_summary.get("backend"),
        "endpoint": spec_summary.get("endpoint"),
        "schema": profile,
        "ontology_mappings": mappings,
        "mapped": mapped,
        "novel": novel,
        "profiled_at": profile["profiled_at"],
    }
    imprinted = False
    if authority_engine is not None:
        try:
            authority_engine.add_node(node_id, "ExternalGraphReference", props)
            imprinted = True
        except Exception as e:  # noqa: BLE001 — return the profile even if write fails
            logger.warning("imprint add_node failed for %s: %s", name, e)

    return {
        "status": "success",
        "connection": name,
        "imprint_node": node_id if imprinted else None,
        "label_count": profile["label_count"],
        "relationship_type_count": len(profile["relationship_types"]),
        "mapped": mapped,
        "novel": novel,
        "schema": profile,
        "ontology_mappings": mappings,
    }
