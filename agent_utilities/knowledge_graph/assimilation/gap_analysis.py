#!/usr/bin/python
from __future__ import annotations

"""Auto gap analysis — the "stop rediscovering built features" engine (CONCEPT:KG-2.7).

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

import re
from dataclasses import dataclass, field
from typing import Any

from ...models.knowledge_graph import RegistryNodeType
from .dedup import iter_all_edges

# A concept id like KG-2.7 / AHE-3.12 / ORCH-1.3b / OS-5 / KG-2.20g.
_CONCEPT_ID_RE = re.compile(r"\b([A-Z]{2,6}-\d+(?:\.\d+[a-z]?|-\d+)?)\b")

_FEATURE_TYPES: tuple[str, ...] = (
    RegistryNodeType.SDD_FEATURE.value,
    RegistryNodeType.CAPABILITY.value,
    RegistryNodeType.ARTICLE.value,
)
_CONCEPT_TYPES: tuple[str, ...] = (RegistryNodeType.CONCEPT.value,)
# A feature is "closed" (excluded from the cycle) by any of these statuses…
_CLOSED_STATUS = {"satisfied", "implemented", "rejected", "superseded", "done"}
# …or by these incident closing edges (matched on the `_rel` property marker).
_CLOSING_OUT = {"SATISFIED_BY", "DERIVED_FROM_RESEARCH"}
_CLOSING_IN = {"SUPERSEDES"}


@dataclass
class GapReport:
    features: int = 0
    concepts: int = 0
    satisfied: int = 0  # candidate SATISFIED_BY edges written
    candidates: list[tuple[str, str, float]] = field(default_factory=list)


def _canonical_id(raw: str) -> str:
    """Normalize a concept id token: ``kg-2-14`` / ``kg-2.14`` → ``KG-2.14``."""
    s = str(raw).strip()
    m = re.match(r"^([A-Za-z]{2,6})-(\d+)[.\-](\d+[a-z]?)$", s)
    if m:
        return f"{m.group(1).upper()}-{m.group(2)}.{m.group(3)}"
    m2 = re.match(r"^([A-Za-z]{2,6})-(\d+[a-z]?)$", s)  # bare major, e.g. OS-5
    return f"{m2.group(1).upper()}-{m2.group(2)}" if m2 else s.upper()


def _concept_key(nid: str, data: dict[str, Any]) -> str | None:
    """The canonical concept id a *concept node* represents (id field or its key)."""
    for cand in (data.get("concept_id"), data.get("id"), nid, data.get("name", "")):
        if cand:
            key = _canonical_id(str(cand))
            if re.match(r"^[A-Z]{2,6}-\d", key):
                return key
    return None


def _feature_refs(nid: str, data: dict[str, Any]) -> list[str]:
    """Canonical concept ids a feature **declares as its own identity**, ordered.

    Sourced — in priority order — from the curated ``concept_ids`` property, then
    the feature's own id, then its name/title. **Body/abstract prose is NOT
    scanned**: a research plan that *cites* ``ORCH-1.0`` as related work is not the
    same capability as ORCH-1.0, so scraping body text mis-marks new research as
    already-built. Matching on declared identity keeps precision high.
    """
    refs: list[str] = []

    def _add(s: Any) -> None:
        for m in _CONCEPT_ID_RE.findall(str(s).upper()):
            c = _canonical_id(m)
            if re.match(r"^[A-Z]{2,6}-\d", c) and c not in refs:
                refs.append(c)

    cids = data.get("concept_ids")
    if isinstance(cids, list | tuple):
        for c in cids:
            _add(c)
    _add(nid)
    _add(data.get("name", ""))
    _add(data.get("title", ""))
    return refs


def _node_data_by_id(graph: Any, nid: str) -> dict[str, Any] | None:
    """Fetch ONE node's data by id without a whole-graph pull (CONCEPT:KG-2.193).

    Uses the NX-compatible node view (``graph.nodes()[id]`` / ``graph.nodes[id]``);
    returns ``None`` on any miss so the caller can degrade to a full scan.
    """
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

    ``restrict_to`` collects ONLY those ids via per-id fetch (CONCEPT:KG-2.193) so a
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

    When ``status`` is not supplied, the node's stored ``status`` is consulted, so
    ``is_closed(engine, fid)`` is self-sufficient.
    """
    graph = getattr(engine, "graph", None)
    if not status and graph is not None:
        try:
            for nid, data in graph.nodes(data=True):
                if nid == feature_id and isinstance(data, dict):
                    status = str(data.get("status", ""))
                    break
        except TypeError:  # pragma: no cover
            pass
    if (status or "").lower() in _CLOSED_STATUS:
        return True
    if graph is None:
        return False
    try:
        for _s, _t, props in graph.out_edges(feature_id, data=True):
            if _rel_of(props) in _CLOSING_OUT:
                return True
        for _s, _t, props in graph.in_edges(feature_id, data=True):
            if _rel_of(props) in _CLOSING_IN:
                return True
    except (TypeError, AttributeError):  # pragma: no cover - non-standard graph
        return False
    return False


def _closed_feature_index(
    engine: Any, feature_types: tuple[str, ...]
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """``(closed_ids, all_features)`` — feature ids closed by status or a closing edge.

    BATCHED: one node scan + one bulk edge scan
    (:func:`~assimilation.dedup.iter_all_edges`), instead of ``O(features)``
    per-node ``out_edges``/``in_edges`` round-trips — the live-backend scaling fix.
    Falls back to the per-node :func:`is_closed` when the graph has no bulk edge
    view (test doubles), preserving identical semantics.
    """
    graph = getattr(engine, "graph", None)
    feats: dict[str, dict[str, Any]] = {}
    closed: set[str] = set()
    if graph is None:
        return closed, feats
    try:
        node_iter = graph.nodes(data=True)
    except TypeError:  # pragma: no cover
        return closed, feats
    wanted = {t.lower() for t in feature_types}  # case-insensitive (live labels)
    for nid, data in node_iter:
        if (
            not isinstance(data, dict)
            or str(data.get("type", "")).lower() not in wanted
        ):
            continue
        feats[nid] = data
        if str(data.get("status", "")).lower() in _CLOSED_STATUS:
            closed.add(nid)

    edges = iter_all_edges(graph)
    if edges is not None:  # bulk path — one traversal
        for src, dst, props in edges:
            rel = _rel_of(props)
            if rel in _CLOSING_OUT and src in feats:
                closed.add(src)
            elif rel in _CLOSING_IN and dst in feats:
                closed.add(dst)
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
