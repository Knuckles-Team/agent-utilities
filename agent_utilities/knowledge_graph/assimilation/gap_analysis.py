#!/usr/bin/python
from __future__ import annotations

"""Auto gap analysis — the "stop rediscovering built features" engine (CONCEPT:AU-KG.query.vendor-agnostic-traversal).

The first evolution attempt repeatedly re-proposed features we had *already built*,
because "does this already exist?" was answered by re-reading. Here it is a graph
operation: every extracted feature is embedding-matched against our existing
``Concept`` nodes; a match above threshold writes a candidate
``feature -[SATISFIED_BY]-> concept`` edge. ``open_features`` is then the gap query —
features that are neither satisfied, superseded, nor closed by status — and that is
the *only* set the golden loop should propose against.

Backend-portable: closing edges are detected via the ``_rel`` property marker that
the assimilation engine stamps on its edges (``out_edges``/``in_edges`` expose
properties, not the relationship label), plus the node's own ``status`` field.

Concept: gap-analysis
"""

from dataclasses import dataclass, field
from typing import Any

from epistemic_graph.assimilation_lifecycle_derivation import (
    closed_feature_ids,
    relation_closes,
    status_is_closed,
)

from ...models.knowledge_graph import RegistryNodeType
from .dedup import iter_all_edges

_FEATURE_TYPES: tuple[str, ...] = (
    RegistryNodeType.SDD_FEATURE.value,
    RegistryNodeType.CAPABILITY.value,
    RegistryNodeType.ARTICLE.value,
)
_CONCEPT_TYPES: tuple[str, ...] = (RegistryNodeType.CONCEPT.value,)


@dataclass
class GapReport:
    features: int = 0
    concepts: int = 0
    satisfied: int = 0  # candidate SATISFIED_BY edges written
    candidates: list[tuple[str, str, float]] = field(default_factory=list)


def _node_data_by_id(graph: Any, nid: str) -> dict[str, Any] | None:
    """Fetch ONE node's full data by id without a whole-graph pull (CONCEPT:AU-KG.ingest.fetch-only-requested-ids).

    The live ``GraphComputeEngine`` facade exposes a per-id properties fetch
    (``get_node_properties`` / ``_get_node_properties``) that does a single engine
    round-trip — NOT a ``GetNodes`` whole-graph list, which on a large multi-tenant
    engine returns a huge payload and resets the socket. We prefer that; only a
    test-double dict graph falls through to the NX-style ``nodes()[id]`` view.
    Returns ``None`` on a miss so the caller can degrade to a full scan.
    """
    for meth in ("get_node_properties", "_get_node_properties"):
        fn = getattr(graph, meth, None)
        if callable(fn):
            try:
                data = fn(nid)
            except Exception:  # noqa: BLE001 — try the next access path
                continue
            # the engine returns {} for a missing id — treat as not-found
            return data if isinstance(data, dict) and data else None
    try:
        nodes = graph.nodes
        view = nodes() if callable(nodes) else nodes
        data = view[nid]
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001 — any view/lookup miss → caller decides
        return None


def _collect_rich(
    engine: Any,
    node_types: tuple[str, ...],
    restrict_to: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """id → full node ``data`` for the target types (case-insensitive label match).

    ``restrict_to`` collects ONLY those ids via per-id fetch (CONCEPT:AU-KG.ingest.fetch-only-requested-ids) so a
    per-cohort pass is O(cohort) not O(graph); falls back to a filtered full scan if
    the node view can't be indexed by id.
    """
    out: dict[str, dict[str, Any]] = {}
    graph = getattr(engine, "graph", None)
    if graph is None:
        return out
    wanted = {t.lower() for t in node_types}
    if restrict_to is not None:
        for nid in restrict_to:
            data = _node_data_by_id(graph, nid)
            if data is not None and str(data.get("type", "")).lower() in wanted:
                out[nid] = data
        if out or not restrict_to:
            return out
        # per-id view unavailable → filtered full scan (correctness over speed)
        try:
            return {
                nid: data
                for nid, data in graph.nodes(data=True)
                if nid in restrict_to
                and isinstance(data, dict)
                and str(data.get("type", "")).lower() in wanted
            }
        except TypeError:  # pragma: no cover - non-standard graph
            return out
    # Unrestricted: prefer the engine-side BOUNDED label fetch (CONCEPT:EG-KG.txn.per-graph-write-isolation) over a
    # whole-graph GetNodes dump — on a large multi-tenant engine the full node list is a
    # huge payload that resets the socket. Try each type's label across casings (live
    # labels are inconsistently cased, e.g. "article" vs "Concept").
    by_label = getattr(graph, "get_nodes_by_label", None)
    if callable(by_label):
        labels = {
            cased
            for t in node_types
            for cased in (t, t.lower(), t.capitalize(), t.upper(), t.title())
        }
        seen_any = False
        for lbl in labels:
            try:
                rows = by_label(lbl, 0) or []
            except Exception:  # noqa: BLE001 — try the next label casing
                continue
            for row in rows:
                if isinstance(row, list | tuple) and len(row) >= 2:
                    nid, data = str(row[0]), row[1]
                    if (
                        isinstance(data, dict)
                        and str(data.get("type", "")).lower() in wanted
                    ):
                        out[nid] = data
                        seen_any = True
        if seen_any:
            return out
    try:
        node_iter = graph.nodes(data=True)
    except TypeError:  # pragma: no cover - non-standard graph
        return out
    for nid, data in node_iter:
        if isinstance(data, dict) and str(data.get("type", "")).lower() in wanted:
            out[nid] = data
    return out


def _rel_of(props: Any) -> str:
    return str(props.get("_rel", "")) if isinstance(props, dict) else ""


def is_closed(engine: Any, feature_id: str, status: str = "") -> bool:
    """True if ``feature_id`` is satisfied/superseded or closed by status.

    When ``status`` is not supplied, the node's stored ``status`` is consulted via
    :func:`_node_data_by_id` (CONCEPT:AU-KG.ingest.fetch-only-requested-ids) — a bounded per-id fetch, NOT a
    whole-graph ``nodes(data=True)`` scan for one id — so ``is_closed(engine, fid)``
    stays self-sufficient without risking a ``RESULT_TOO_LARGE`` whole-graph dump on
    a large engine. IMPORTANT: once a bounded per-id surface exists on the graph, a
    "not found" answer from it is trusted as-is (no status) — it is NEVER treated as
    a signal to fall back to a full scan, since a live engine returns an empty dict
    for a genuinely missing id (not an error), and re-scanning on every miss would
    reopen the same ``RESULT_TOO_LARGE`` risk. The full per-node scan is reserved
    for graphs with NO bounded per-id surface at all (e.g. a minimal test double) —
    on a real engine this branch never fires.
    """
    graph = getattr(engine, "graph", None)
    if not status and graph is not None:
        has_bounded_lookup = any(
            callable(getattr(graph, m, None))
            for m in ("get_node_properties", "_get_node_properties")
        )
        if has_bounded_lookup:
            data = _node_data_by_id(graph, feature_id)
            if isinstance(data, dict):
                status = str(data.get("status", ""))
        else:
            try:
                for nid, d in graph.nodes(data=True):
                    if nid == feature_id and isinstance(d, dict):
                        status = str(d.get("status", ""))
                        break
            except TypeError:  # noqa: BLE001 — non-standard local graph has no data view
                pass
    if status_is_closed(status):
        return True
    if graph is None:
        return False
    try:
        for _s, _t, props in graph.out_edges(feature_id, data=True):
            if relation_closes(_rel_of(props), incoming=False):
                return True
        for _s, _t, props in graph.in_edges(feature_id, data=True):
            if relation_closes(_rel_of(props), incoming=True):
                return True
    except (TypeError, AttributeError):  # pragma: no cover - non-standard graph
        return False
    return False


def _closed_feature_index(
    engine: Any, feature_types: tuple[str, ...]
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """``(closed_ids, all_features)`` — feature ids closed by status or a closing edge.

    BATCHED: a BOUNDED per-label node collection (:func:`_collect_rich` — the same
    ``get_nodes_by_label``-preferring collector used elsewhere in this module,
    CONCEPT:EG-KG.txn.per-graph-write-isolation, never an unscoped whole-graph
    ``nodes(data=True)`` dump) plus one bulk edge scan
    (:func:`~assimilation.dedup.iter_all_edges`), instead of ``O(features)``
    per-node ``out_edges``/``in_edges`` round-trips — the live-backend scaling fix.
    Falls back to the per-node :func:`is_closed` when the graph has no bulk edge
    view (test doubles), preserving identical semantics.
    """
    graph = getattr(engine, "graph", None)
    closed: set[str] = set()
    if graph is None:
        return closed, {}
    feats = _collect_rich(engine, feature_types)
    edges = iter_all_edges(graph)
    if edges is not None:  # bulk path — one traversal
        closed = closed_feature_ids(
            feats, ((src, dst, _rel_of(props)) for src, dst, props in edges)
        )
    else:  # per-node fallback (no bulk edge view)
        for fid, data in feats.items():
            if fid not in closed and is_closed(
                engine, fid, str(data.get("status", "open"))
            ):
                closed.add(fid)
    return closed, feats


def open_features(
    engine: Any,
    *,
    feature_types: tuple[str, ...] = _FEATURE_TYPES,
) -> list[str]:
    """Return feature ids with no closing edge / closed status — the cycle's input.

    This is the durable, queryable answer to "what have we NOT already hit?" — the
    set the golden loop proposes against (everything else is excluded).
    """
    closed, feats = _closed_feature_index(engine, feature_types)
    return [fid for fid in feats if fid not in closed]


__all__ = ["GapReport", "open_features", "is_closed"]
