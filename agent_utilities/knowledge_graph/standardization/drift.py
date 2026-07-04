#!/usr/bin/python
from __future__ import annotations

"""Per-asset / per-org / per-domain conformance drift (CONCEPT:AU-KG.ontology.populated-at-import-real-3).

Scores every governed asset against its enterprise standard via
:func:`standards.drift_score`, writes an idempotent
``asset -[CONFORMS_TO {drift_score, gaps}]-> EnterpriseStandard`` edge, and rolls
the scores up per owning organization and per standard domain — the convergence
dashboard that shows how far each opinionated org is from the north-star.

The roll-up mirrors the egeria ``audit.py`` per-capability pattern; the edge
idempotency mirrors ``gap_analysis._clear_auto_satisfied`` (``_rel`` marker +
``auto`` flag, cleared via ``engine.delete_edge`` before re-write).
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ...models.knowledge_graph import RegistryEdgeType, RegistryNodeType
from ..assimilation.dedup import iter_all_edges
from .cohorts import read_assets
from .standards import drift_score, standard_node_id


@dataclass
class DomainDrift:
    """Aggregate drift for one enterprise-standard domain."""

    standard: str
    asset_count: int = 0
    conformant: int = 0
    mean_drift: float = 0.0


@dataclass
class OrgDrift:
    """Aggregate drift for one owning organization/business unit."""

    organization: str
    asset_count: int = 0
    mean_drift: float = 0.0


@dataclass
class DriftReport:
    """Outcome of a conformance-scoring pass."""

    assets_scored: int = 0
    edges_written: int = 0
    per_domain: dict[str, DomainDrift] = field(default_factory=dict)
    per_org: dict[str, OrgDrift] = field(default_factory=dict)


def _org_of(asset: dict[str, Any]) -> str:
    """Owning organization for roll-up (property fallbacks; ``unassigned``)."""
    for k in ("organization", "business_unit", "org", "owner_org", "cost_center"):
        v = asset.get(k)
        if isinstance(v, str) and v:
            return v
    return "unassigned"


def _clear_auto_conforms(engine: Any, asset_ids: set[str]) -> None:
    """Remove prior auto-written ``CONFORMS_TO`` edges (idempotent re-score)."""
    graph = getattr(engine, "graph", None)
    deleter = getattr(engine, "delete_edge", None)
    if graph is None or not callable(deleter):
        return
    edges = iter_all_edges(graph)
    if edges is None:
        return
    for src, dst, props in edges:
        if (
            src in asset_ids
            and isinstance(props, dict)
            and str(props.get("_rel", "")) == "CONFORMS_TO"
            and props.get("auto")
        ):
            try:
                deleter(src, dst, RegistryEdgeType.CONFORMS_TO.value)
            except Exception:  # noqa: BLE001 - best-effort reconcile
                pass


def score_conformance(
    engine: Any,
    *,
    assets: dict[str, dict[str, Any]] | None = None,
    write: bool = True,
) -> DriftReport:
    """Score governed assets against their enterprise standards.

    For each governed asset: compute ``(drift, gaps)``, write a ``CONFORMS_TO``
    edge to the standard's materialized node, and accumulate per-domain /
    per-org roll-ups. Optionally persist ``DRIFT_ROLLUP`` nodes for querying.

    Args:
        engine: knowledge engine (``graph.nodes`` + ``link_nodes``/``add_node``).
        assets: optional pre-read governed-asset map (avoids a second scan).
        write: persist CONFORMS_TO edges + roll-up nodes (False = analysis only).

    Returns:
        A :class:`DriftReport`.
    """
    assets = assets if assets is not None else read_assets(engine)
    report = DriftReport()
    if not assets:
        return report

    if write:
        _clear_auto_conforms(engine, set(assets))

    dom_acc: dict[str, list[float]] = defaultdict(list)
    dom_conform: dict[str, int] = defaultdict(int)
    org_acc: dict[str, list[float]] = defaultdict(list)
    link = getattr(engine, "link_nodes", None)

    for nid, asset in assets.items():
        std_name = asset["standard"]
        drift, gaps = drift_score(asset, std_name)
        report.assets_scored += 1
        dom_acc[std_name].append(drift)
        if drift == 0.0:
            dom_conform[std_name] += 1
        org_acc[_org_of(asset)].append(drift)

        if write and callable(link):
            try:
                link(
                    nid,
                    standard_node_id(std_name),
                    RegistryEdgeType.CONFORMS_TO,
                    properties={
                        "_rel": "CONFORMS_TO",
                        "auto": True,
                        "drift_score": drift,
                        "gaps": gaps[:20],
                        "concept": "AU-KG.ontology.populated-at-import-real-3",
                    },
                )
                report.edges_written += 1
            except Exception:  # noqa: BLE001 - best-effort edge write
                pass

    for std_name, drifts in dom_acc.items():
        report.per_domain[std_name] = DomainDrift(
            standard=std_name,
            asset_count=len(drifts),
            conformant=dom_conform.get(std_name, 0),
            mean_drift=round(sum(drifts) / len(drifts), 6) if drifts else 0.0,
        )
    for org, drifts in org_acc.items():
        report.per_org[org] = OrgDrift(
            organization=org,
            asset_count=len(drifts),
            mean_drift=round(sum(drifts) / len(drifts), 6) if drifts else 0.0,
        )

    if write:
        _persist_rollups(engine, report)
    return report


def asset_drift_map(
    engine: Any, *, assets: dict[str, dict[str, Any]] | None = None
) -> dict[str, float]:
    """Return ``id -> drift`` for governed assets (analysis-only, no writes).

    The consolidation engine consumes this to pick the lowest-drift north-star in
    a candidate group without re-reading the graph.
    """
    assets = assets if assets is not None else read_assets(engine)
    out: dict[str, float] = {}
    for nid, asset in assets.items():
        drift, _ = drift_score(asset, asset["standard"])
        out[nid] = drift
    return out


def _persist_rollups(engine: Any, report: DriftReport) -> None:
    """Persist per-domain / per-org drift as queryable ``DRIFT_ROLLUP`` nodes."""
    add = getattr(engine, "add_node", None)
    if not callable(add):
        return
    rollup_type = RegistryNodeType.DRIFT_ROLLUP.value
    for std_name, dd in report.per_domain.items():
        try:
            add(
                f"drift_rollup:domain:{std_name}",
                rollup_type,
                properties={
                    "scope": "domain",
                    "standard": std_name,
                    "asset_count": dd.asset_count,
                    "conformant": dd.conformant,
                    "mean_drift": dd.mean_drift,
                    "concept": "AU-KG.ontology.populated-at-import-real-3",
                },
            )
        except Exception:  # noqa: BLE001
            pass
    for org, od in report.per_org.items():
        try:
            add(
                f"drift_rollup:org:{org}",
                rollup_type,
                properties={
                    "scope": "org",
                    "organization": org,
                    "asset_count": od.asset_count,
                    "mean_drift": od.mean_drift,
                    "concept": "AU-KG.ontology.populated-at-import-real-3",
                },
            )
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "DriftReport",
    "DomainDrift",
    "OrgDrift",
    "score_conformance",
    "asset_drift_map",
]
