#!/usr/bin/python
# CONCEPT:KG-2.74 - Native cross-backend graph migration: copy a graph's nodes+edges (+embeddings) from any source (the L1 compute store or a full-cypher durable backend) into any target backend, writing through the engine's proven dialect-aware MERGE upserts so pg-age, Neo4j, FalkorDB and LadybugDB all receive a correct native write — the backfill/interchange primitive behind the fan-out mirror set.
"""Backend-agnostic graph data migration.

CONCEPT:KG-2.74 — Interchangeable storage across backends.

``copy_graph(source, target)`` reads every node + edge (+ node embeddings) from a
source and writes them into a target backend. The WRITE side reuses the engine's
**proven-portable** upsert path — :meth:`IntelligenceGraphEngine._upsert_node` and
:meth:`._upsert_edge` — which emit MERGE-on-``{id}`` cypher with per-backend dialect
handling (ad-hoc keys folded into the ``metadata`` JSON column, edge props into the
single ``properties`` JSON column for strict-schema Kuzu/Ladybug, nested values
JSON-encoded for map-rejecting drivers). Those upserts write ONLY to the backend
(no graph_compute / no shared L1), so a migration never pollutes or contends with
the live L1 engine.

This replaces the old reconcile path, which reconstructed its own
``CREATE (n:Label {`k`: $k, ...})`` cypher — fragile on native-cypher backends
(double-backticked / reserved keys → syntax errors) and edge-lossy. Because the
write is MERGE-based it is **idempotent and resumable**: re-running converges
rather than duplicating.

Source reading:
* an L1/compute source (anything exposing ``.graph`` with ``_get_all_nodes`` /
  ``_get_node_properties``) — used for the fan-out backfill (authority/L1 → mirrors);
* a full-cypher durable backend (Neo4j/FalkorDB/AGE) — read via ``MATCH``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from .backends.base import GraphBackend
from .backends.tiered_backend import _sanitize_label

logger = logging.getLogger(__name__)


def _as_backend(target: Any) -> GraphBackend:
    """Resolve a backend from a backend, an engine, or a tiered/fanout wrapper."""
    if isinstance(target, GraphBackend):
        return target
    be = getattr(target, "backend", None)
    if isinstance(be, GraphBackend):
        return be
    raise TypeError(f"copy_graph: cannot resolve a GraphBackend from {target!r}")


def _compute_graph(source: Any) -> Any | None:
    """The L1 compute graph for ``source`` (its ``.graph``, or itself), if it
    exposes ``_get_all_nodes`` — else ``None`` (a cypher source)."""
    for cand in (getattr(source, "graph", None), source):
        if cand is not None and hasattr(cand, "_get_all_nodes"):
            return cand
    return None


def _iter_source_nodes(source: Any) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield ``(node_id, label, props)`` from the source (L1 compute or cypher)."""
    graph = _compute_graph(source)
    if graph is not None:
        for nid in graph._get_all_nodes():
            try:
                props = dict(graph._get_node_properties(nid) or {})
            except Exception:  # noqa: BLE001
                props = {}
            label = _sanitize_label(props.get("type") or props.get("label") or "Node")
            clean = {
                k: v for k, v in props.items() if k != "embedding" and v is not None
            }
            clean.setdefault("type", label)
            yield str(nid), label, clean
        return
    # Full-cypher durable source.
    backend = _as_backend(source)
    nested_unsafe = getattr(backend, "cypher_support", "full") == "full"
    lbl = "labels(n)[0]" if nested_unsafe else "label(n)"
    rows = backend.execute(f"MATCH (n) RETURN n.id AS id, {lbl} AS label, n AS node")
    for r in rows:
        nid = r.get("id")
        if nid is None:
            continue
        node = r.get("node")
        props = dict(node) if isinstance(node, dict) else {}
        props.pop("embedding", None)
        label = _sanitize_label(r.get("label") or props.get("type") or "Node")
        props.setdefault("type", label)
        yield str(nid), label, props


def _iter_source_edges(source: Any) -> Iterator[tuple[str, str, str, dict[str, Any]]]:
    """Yield ``(source_id, target_id, rel_type, props)`` from the source."""
    graph = _compute_graph(source)
    if graph is not None:
        # Reuse the L1 edge enumeration shape from TieredGraphBackend._l1_edges.
        view = getattr(graph, "edges", None)
        seq: Any = None
        if view is not None:
            try:
                seq = view(data=True) if callable(view) else view
            except Exception:  # noqa: BLE001
                seq = None
        if seq is None:
            fn = getattr(graph, "_get_all_edges", None)
            if callable(fn):
                try:
                    seq = fn()
                except Exception:  # noqa: BLE001
                    seq = None
        if seq is None:
            try:
                seq = graph._client.edges.list()
            except Exception:  # noqa: BLE001
                seq = []
        for e in seq or []:
            if isinstance(e, tuple | list) and len(e) >= 2:
                data = e[2] if len(e) > 2 and isinstance(e[2], dict) else {}
                rel = str(data.get("type") or "RELATED_TO")
                props = {k: v for k, v in data.items() if k != "type" and v is not None}
                yield str(e[0]), str(e[1]), rel, props
        return
    # Full-cypher durable source.
    backend = _as_backend(source)
    rows = backend.execute(
        "MATCH (s)-[r]->(t) RETURN s.id AS sid, t.id AS tid, type(r) AS rel, r AS edge"
    )
    for r in rows:
        sid, tid = r.get("sid"), r.get("tid")
        if sid is None or tid is None:
            continue
        edge = r.get("edge")
        props = dict(edge) if isinstance(edge, dict) else {}
        yield str(sid), str(tid), str(r.get("rel") or "RELATED_TO"), props


def _iter_source_embeddings(source: Any) -> Iterator[tuple[str, list[float]]]:
    """Yield ``(node_id, embedding)`` for nodes that carry one (L1 source only)."""
    graph = _compute_graph(source)
    if graph is None:
        return
    for nid in graph._get_all_nodes():
        try:
            props = graph._get_node_properties(nid) or {}
        except Exception:  # noqa: BLE001
            continue
        emb = props.get("embedding")
        if isinstance(emb, list) and emb:
            yield str(nid), emb


def _portable_writer(backend: GraphBackend) -> Any:
    """A minimal IntelligenceGraphEngine subclass exposing only the portable
    backend-only upserts (``_upsert_node`` / ``_upsert_edge``) — constructed without
    the engine's heavy ``__init__`` (services, hooks, retriever, graph_compute,
    active-engine registration). The inherited helpers need only ``self.backend`` +
    class constants (``_NESTED_UNSAFE`` / ``_ARRAY_FIELDS`` / ``_SCHEMA_BACKED``) +
    the global ``SCHEMA``, so a bare ``self.backend`` is sufficient.
    """
    from .core.engine import IntelligenceGraphEngine

    class _PortableGraphWriter(IntelligenceGraphEngine):
        def __init__(self, be: GraphBackend) -> None:  # noqa: D401 — minimal ctor
            self.backend = be

    return _PortableGraphWriter(backend)


def copy_graph(
    source: Any,
    target: Any,
    *,
    copy_embeddings: bool = True,
) -> dict[str, int]:
    """Migrate all nodes + edges (+ embeddings) from ``source`` into ``target``.

    ``source`` / ``target`` may be a ``GraphBackend`` or anything exposing
    ``.backend`` (an engine, tiered/fanout wrapper). Writes go through the engine's
    portable backend-only upserts, so every backend gets a dialect-correct native
    write. Idempotent (MERGE). Returns counts + post-condition drift.
    """
    target_backend = _as_backend(target)
    # A minimal writer that reuses ONLY the engine's portable backend upserts
    # (_upsert_node/_upsert_edge) — no services, hooks, retriever, graph_compute, or
    # active-engine registration. So a migration never writes the shared L1 nor
    # triggers the engine's write-path side-effects.
    writer = _portable_writer(target_backend)

    summary = {
        "nodes": 0,
        "edges": 0,
        "embeddings": 0,
        "errors": 0,
        "nodes_missing": 0,
        "edges_missing": 0,
    }

    # --- nodes first (edges MATCH on existing nodes) ---
    labels_seen: dict[str, int] = {}
    for node_id, label, props in _iter_source_nodes(source):
        try:
            writer._upsert_node(label, node_id, {"id": node_id, **props})
            summary["nodes"] += 1
            labels_seen[label] = labels_seen.get(label, 0) + 1
        except Exception as exc:  # noqa: BLE001
            summary["errors"] += 1
            logger.debug("copy_graph: node %s (%s) failed: %s", node_id, label, exc)

    # --- edges ---
    n_src_edges = 0
    for src, dst, rel, props in _iter_source_edges(source):
        n_src_edges += 1
        try:
            writer._upsert_edge(src, dst, (rel or "RELATED_TO").upper(), props)
            summary["edges"] += 1
        except Exception as exc:  # noqa: BLE001
            summary["errors"] += 1
            logger.debug("copy_graph: edge %s->%s failed: %s", src, dst, exc)

    # --- embeddings ---
    if copy_embeddings:
        for node_id, emb in _iter_source_embeddings(source):
            try:
                target_backend.add_embedding(node_id, emb)
                summary["embeddings"] += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("copy_graph: embedding %s failed: %s", node_id, exc)

    # --- exact post-condition drift: per-label node counts + edge count ---
    for label, n_src in labels_seen.items():
        try:
            rows = target_backend.execute(f"MATCH (n:{label}) RETURN count(n) AS c")
            tgt = int(rows[0]["c"]) if rows and "c" in rows[0] else None
        except Exception:  # noqa: BLE001
            tgt = None
        if tgt is None:
            summary["nodes_missing"] += n_src
        else:
            summary["nodes_missing"] += max(0, n_src - tgt)
    tgt_edges = getattr(target_backend, "edge_count", lambda: None)()
    if tgt_edges is not None:
        summary["edges_missing"] = max(0, n_src_edges - int(tgt_edges))

    logger.info(
        "copy_graph: %d nodes, %d edges, %d embeddings written to %s; "
        "drift %d nodes / %d edges; %d errors",
        summary["nodes"],
        summary["edges"],
        summary["embeddings"],
        type(target_backend).__name__,
        summary["nodes_missing"],
        summary["edges_missing"],
        summary["errors"],
    )
    return summary
