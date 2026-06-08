#!/usr/bin/python
from __future__ import annotations

"""Feature lifecycle ledger + assimilation close-out (CONCEPT:KG-2.7).

The durable record of *what we shipped*, so the loop never re-opens it. Three jobs:

* :func:`record_feature` / :func:`set_status` — create/update an `SDDFeature` node
  (the lifecycle ledger entry; status drives golden-loop exclusion via
  ``gap_analysis.open_features``).
* :func:`close_out` — when a feature is implemented, write
  ``feature -[DERIVED_FROM_RESEARCH]-> source`` and ``source -[ASSIMILATED_INTO]-> codebase``
  and flip the status to ``implemented`` (KG-2.7 US-1/3). This closes the research
  → code provenance loop and excludes the feature from future cycles.
* :func:`promote_feature_ledger` — lift the YAML feature/capability ledger
  (``scripts/build_feature_ledger.py``) into `SDDFeature` nodes so existing code is
  represented in the assimilation graph.
* :func:`ledger_state` — a queryable open/closed/by-status summary.

Nodes are written with ``engine.add_node`` (same path as the evolving-memory store)
so the stored ``type`` stays ``"sdd_feature"`` and matches the dedup/gap/synergy
filters. Edges carry the ``_rel`` marker for backend-portable lifecycle reads.

Concept: lifecycle-ledger
"""

from dataclasses import dataclass
from typing import Any

from ...models.knowledge_graph import RegistryEdgeType, RegistryNodeType
from .gap_analysis import _FEATURE_TYPES, is_closed

_SDD = RegistryNodeType.SDD_FEATURE.value


@dataclass
class CloseOutReport:
    feature_id: str
    status: str = "implemented"
    derived_from: int = 0
    assimilated: int = 0


def _get_node(engine: Any, node_id: str) -> dict[str, Any] | None:
    graph = getattr(engine, "graph", None)
    if graph is None:
        return None
    try:
        for nid, data in graph.nodes(data=True):
            if nid == node_id and isinstance(data, dict):
                return data
    except TypeError:  # pragma: no cover
        return None
    return None


def record_feature(
    engine: Any,
    *,
    feature_id: str,
    name: str,
    concept_ids: list[str] | tuple[str, ...] = (),
    research_sources: list[str] | tuple[str, ...] = (),
    status: str = "open",
    sdd_path: str = "",
    codebase: str = "",
) -> str:
    """Upsert an ``SDDFeature`` lifecycle node. Idempotent by ``feature_id``."""
    props = {
        "name": name,
        "concept_ids": list(concept_ids),
        "research_sources": list(research_sources),
        "status": status,
        "sdd_path": sdd_path,
        "codebase": codebase,
    }
    engine.add_node(feature_id, _SDD, properties=props)
    return feature_id


def set_status(engine: Any, feature_id: str, status: str) -> bool:
    """Update a feature's lifecycle status (re-upsert; idempotent)."""
    data = _get_node(engine, feature_id)
    if data is None:
        return False
    props = {k: v for k, v in data.items() if k != "type"}
    props["status"] = status
    engine.add_node(feature_id, data.get("type", _SDD), properties=props)
    return True


def close_out(
    engine: Any,
    feature_id: str,
    *,
    codebase: str | None = None,
    status: str = "implemented",
    assimilation_date: str | None = None,
) -> CloseOutReport:
    """Record a feature as assimilated: provenance edges + closed status (KG-2.7).

    For each of the feature's ``research_sources`` writes
    ``feature -[DERIVED_FROM_RESEARCH]-> source`` and (when a codebase is known)
    ``source -[ASSIMILATED_INTO]-> codebase``; then sets the feature ``status``.
    """
    report = CloseOutReport(feature_id=feature_id, status=status)
    data = _get_node(engine, feature_id) or {}
    sources = list(data.get("research_sources", []) or [])
    cb = codebase if codebase is not None else str(data.get("codebase", "") or "")
    assim_props: dict[str, Any] = {
        "_rel": "ASSIMILATED_INTO",
        "status": status,
        "concept": "KG-2.7",
    }
    if assimilation_date:
        assim_props["assimilation_date"] = assimilation_date
    for src in sources:
        engine.link_nodes(
            feature_id,
            src,
            RegistryEdgeType.DERIVED_FROM_RESEARCH,
            properties={"_rel": "DERIVED_FROM_RESEARCH", "concept": "KG-2.7"},
        )
        report.derived_from += 1
        if cb:
            engine.link_nodes(
                src, cb, RegistryEdgeType.ASSIMILATED_INTO, properties=dict(assim_props)
            )
            report.assimilated += 1
    set_status(engine, feature_id, status)
    return report


def promote_feature_ledger(engine: Any, rows: list[dict[str, Any]]) -> int:
    """Lift YAML feature-ledger rows into ``SDDFeature`` nodes. Returns count."""
    written = 0
    for row in rows:
        fid = str(row.get("id") or "").strip()
        if not fid:
            continue
        concept = str(row.get("concept", "") or "")
        concept_ids = [concept] if concept and concept != "UNKNOWN" else []
        source = str(row.get("source", "") or "")
        record_feature(
            engine,
            feature_id=fid,
            name=str(row.get("name", fid)),
            concept_ids=concept_ids,
            research_sources=[source] if source else [],
            status=str(row.get("status", "open") or "open"),
            sdd_path=str(row.get("target", "") or ""),
        )
        written += 1
    return written


def ledger_state(
    engine: Any, *, feature_types: tuple[str, ...] = _FEATURE_TYPES
) -> dict[str, Any]:
    """Open/closed/by-status summary over the feature nodes."""
    graph = getattr(engine, "graph", None)
    summary: dict[str, Any] = {"total": 0, "open": 0, "closed": 0, "by_status": {}}
    if graph is None:
        return summary
    try:
        node_iter = graph.nodes(data=True)
    except TypeError:  # pragma: no cover
        return summary
    wanted = {t.lower() for t in feature_types}  # case-insensitive (live labels)
    for nid, data in node_iter:
        if (
            not isinstance(data, dict)
            or str(data.get("type", "")).lower() not in wanted
        ):
            continue
        summary["total"] += 1
        st = str(data.get("status", "open"))
        summary["by_status"][st] = summary["by_status"].get(st, 0) + 1
        if is_closed(engine, nid, st):
            summary["closed"] += 1
        else:
            summary["open"] += 1
    return summary


__all__ = [
    "CloseOutReport",
    "record_feature",
    "set_status",
    "close_out",
    "promote_feature_ledger",
    "ledger_state",
]
