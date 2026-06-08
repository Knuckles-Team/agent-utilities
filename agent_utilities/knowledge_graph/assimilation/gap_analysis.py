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

from dataclasses import dataclass, field
from typing import Any

from ...models.knowledge_graph import RegistryEdgeType, RegistryNodeType
from .dedup import _collect, _cosine

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


def auto_satisfy(
    engine: Any,
    *,
    feature_types: tuple[str, ...] = _FEATURE_TYPES,
    concept_types: tuple[str, ...] = _CONCEPT_TYPES,
    threshold: float = 0.85,
    restrict_to: set[str] | None = None,
    write: bool = True,
) -> GapReport:
    """Write candidate ``SATISFIED_BY`` edges for features matching existing concepts.

    Args:
        engine: knowledge engine (``graph.nodes(data=True)`` + ``link_nodes``).
        feature_types / concept_types: node types to match between.
        threshold: cosine ≥ this → the feature is considered already satisfied.
        restrict_to: only evaluate these feature ids (incremental).
        write: persist edges (False = dry run).

    Returns:
        A :class:`GapReport` (the candidate ``(feature, concept, score)`` matches).
    """
    features = _collect(engine, feature_types)
    concepts = _collect(engine, concept_types)
    report = GapReport(features=len(features), concepts=len(concepts))
    if not features or not concepts:
        return report
    concept_items = list(concepts.items())
    for fid, fdata in features.items():
        if restrict_to and fid not in restrict_to:
            continue
        best_cid, best_s = None, 0.0
        for cid, cdata in concept_items:
            s = _cosine(fdata["vec"], cdata["vec"])
            if s > best_s:
                best_cid, best_s = cid, s
        if best_cid is not None and best_s >= threshold:
            report.satisfied += 1
            report.candidates.append((fid, best_cid, round(best_s, 6)))
            if write:
                engine.link_nodes(
                    fid,
                    best_cid,
                    RegistryEdgeType.SATISFIED_BY,
                    properties={
                        "_rel": "SATISFIED_BY",
                        "score": round(best_s, 6),
                        "auto": True,
                        "concept": "KG-2.7",
                    },
                )
    return report


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


def open_features(
    engine: Any,
    *,
    feature_types: tuple[str, ...] = _FEATURE_TYPES,
) -> list[str]:
    """Return feature ids with no closing edge / closed status — the cycle's input.

    This is the durable, queryable answer to "what have we NOT already hit?" — the
    set the golden loop proposes against (everything else is excluded).
    """
    out: list[str] = []
    graph = getattr(engine, "graph", None)
    if graph is None:
        return out
    try:
        node_iter = graph.nodes(data=True)
    except TypeError:  # pragma: no cover
        return out
    wanted = {t.lower() for t in feature_types}  # case-insensitive (live labels)
    for nid, data in node_iter:
        if (
            not isinstance(data, dict)
            or str(data.get("type", "")).lower() not in wanted
        ):
            continue
        if not is_closed(engine, nid, str(data.get("status", "open"))):
            out.append(nid)
    return out


__all__ = ["GapReport", "auto_satisfy", "open_features", "is_closed"]
