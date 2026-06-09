#!/usr/bin/python
from __future__ import annotations

"""Consolidation decision engine (CONCEPT:KG-2.49).

Turns the redundancy signals (:mod:`cohorts`) and the conformance drift map
(:mod:`drift`) into ranked, propose-only ``ConsolidationRecommendation`` nodes:
"collapse these N projects into one", "retire this licensed tool for that one",
"build a consolidator for these synergistic systems".

Two-axis scoring per candidate group:

  - **value** rises with cohort size, cross-vendor redundancy breadth, summed
    license cost, and mean conformance drift (high-drift redundant assets are the
    best collapse targets — they cost money *and* miss the standard).
  - **risk** rises with blast radius (``engine.get_blast_radius``) and downstream
    lineage count (how much depends on the members).

``priority = value / (1 + risk)`` floats high-value/low-risk collapses to the
top. The **north-star** within a group is the lowest-drift, most-depended-on
member — the survivor everything else is proposed to be ``ABSORBED_INTO``.

Everything is ``status="proposal"``: recommendation nodes + proposed
``ABSORBED_INTO`` edges + a ``REFERENCES`` edge to the governing standard. No
source asset is mutated. Idempotent: recommendations are keyed by member-set hash
and prior auto edges/nodes are cleared before re-write.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any

from ...models.knowledge_graph import RegistryEdgeType, RegistryNodeType
from ..assimilation.dedup import iter_all_edges
from .cohorts import CandidateGroup, read_assets, read_cohorts
from .drift import asset_drift_map
from .standards import applicable_standard, standard_node_id

# Scoring weights — module constants, opt-in overridable via score_group(weights=).
DEFAULT_WEIGHTS: dict[str, float] = {
    "size": 1.0,
    "breadth": 2.0,  # cross-vendor redundancy is the strongest collapse signal
    "cost": 3.0,  # license spend is the dollars-saved signal
    "drift": 2.0,  # high drift + redundancy = best collapse target
    "blast": 1.0,  # risk axis
    "lineage": 0.5,  # risk axis
}

# Edge types that count as "something depends on this asset" (downstream lineage).
_LINEAGE_RELS = {
    "WAS_DERIVED_FROM",
    "USES_DATASET",
    "DEPENDS_ON_SYSTEM",
    "DEPENDS_ON",
}


@dataclass
class ConsolidationRecommendation:
    """A ranked, propose-only consolidation recommendation."""

    rec_id: str
    kind: str
    members: list[str]
    north_star: str
    capability: str | None
    standard: str | None
    value_score: float
    risk_score: float
    priority: float
    rationale: str
    sources: list[str] = field(default_factory=list)


@dataclass
class ConsolidationReport:
    """Outcome of a consolidation-decision pass."""

    groups_considered: int = 0
    recommendations: list[ConsolidationRecommendation] = field(default_factory=list)
    nodes_written: int = 0
    edges_written: int = 0


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _lineage_counts(engine: Any, member_ids: set[str]) -> dict[str, int]:
    """Incoming-lineage edge count per member (one bulk traversal)."""
    counts: dict[str, int] = dict.fromkeys(member_ids, 0)
    graph = getattr(engine, "graph", None)
    edges = iter_all_edges(graph) if graph is not None else None
    if edges is None:
        return counts
    for _src, dst, props in edges:
        if dst in member_ids and isinstance(props, dict):
            if str(props.get("_rel", "") or "").upper() in _LINEAGE_RELS:
                counts[dst] = counts.get(dst, 0) + 1
    return counts


def _blast(engine: Any, node_id: str, depth: int = 2) -> int:
    """Blast-radius size for a node (0 when the engine can't compute it)."""
    fn = getattr(engine, "get_blast_radius", None)
    if not callable(fn):
        return 0
    try:
        return len(fn(node_id, depth) or [])
    except Exception:  # noqa: BLE001 - blast radius is a best-effort risk signal
        return 0


def _rec_id(members: list[str]) -> str:
    """Stable recommendation id from the sorted member set."""
    h = hashlib.sha256("|".join(sorted(members)).encode("utf-8")).hexdigest()[:12]
    return f"consolidation_recommendation:{h}"


def score_group(
    engine: Any,
    group: CandidateGroup,
    drift: dict[str, float],
    lineage: dict[str, int],
    *,
    max_cost: float,
    weights: dict[str, float] | None = None,
) -> ConsolidationRecommendation:
    """Score one candidate group and choose its north-star (no writes)."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    assets = getattr(engine, "_asset_cache", {})

    size = len(group.members)
    breadth = len(group.sources)
    cost = sum(
        _to_float(assets.get(m, {}).get("license_cost", 0)) for m in group.members
    )
    cost += sum(
        _to_float(assets.get(m, {}).get("annual_cost", 0)) for m in group.members
    )
    member_drift = [drift.get(m, 0.0) for m in group.members]
    mean_drift = sum(member_drift) / len(member_drift) if member_drift else 0.0

    blasts = {m: _blast(engine, m) for m in group.members}
    max_blast = max(blasts.values()) if blasts else 0
    total_lineage = sum(lineage.get(m, 0) for m in group.members)

    cost_norm = (cost / max_cost) if max_cost > 0 else 0.0
    value = (
        w["size"] * size
        + w["breadth"] * breadth
        + w["cost"] * cost_norm
        + w["drift"] * mean_drift * size
    )
    risk = w["blast"] * max_blast + w["lineage"] * total_lineage
    priority = round(value / (1.0 + risk), 6)

    # North-star: lowest drift, then most depended-on (lineage), then id.
    north_star = min(
        group.members,
        key=lambda m: (drift.get(m, 1.0), -lineage.get(m, 0), m),
    )

    standard = applicable_standard(assets.get(north_star, {})) or (
        assets.get(north_star, {}).get("standard")
    )
    rationale = (
        f"{group.kind}: {size} members across {breadth} source(s)"
        + (f" for capability '{group.capability}'" if group.capability else "")
        + f"; mean drift {round(mean_drift, 3)}, est. annual cost {round(cost, 2)}, "
        f"max blast radius {max_blast}, downstream lineage {total_lineage}. "
        f"North-star: {north_star} (lowest drift, most depended-on)."
    )
    return ConsolidationRecommendation(
        rec_id=_rec_id(group.members),
        kind=group.kind,
        members=sorted(group.members),
        north_star=north_star,
        capability=group.capability,
        standard=standard,
        value_score=round(value, 6),
        risk_score=round(risk, 6),
        priority=priority,
        rationale=rationale,
        sources=sorted(group.sources),
    )


def _clear_auto_absorbed(engine: Any, member_ids: set[str]) -> None:
    """Remove prior auto-written proposed ``ABSORBED_INTO`` edges (idempotent)."""
    graph = getattr(engine, "graph", None)
    deleter = getattr(engine, "delete_edge", None)
    if graph is None or not callable(deleter):
        return
    edges = iter_all_edges(graph)
    if edges is None:
        return
    for src, dst, props in edges:
        if (
            src in member_ids
            and isinstance(props, dict)
            and str(props.get("_rel", "")) == "ABSORBED_INTO"
            and props.get("auto")
        ):
            try:
                deleter(src, dst, RegistryEdgeType.ABSORBED_INTO.value)
            except Exception:  # noqa: BLE001
                pass


def recommend_consolidations(
    engine: Any,
    *,
    assets: dict[str, dict[str, Any]] | None = None,
    top_n: int = 20,
    min_sources: int = 2,
    weights: dict[str, float] | None = None,
    write: bool = True,
) -> ConsolidationReport:
    """Rank consolidation candidates and emit propose-only recommendations.

    Args:
        engine: knowledge engine.
        assets: optional pre-read governed-asset map.
        top_n: how many top-priority recommendations to persist.
        min_sources: minimum distinct vendors for a capability cohort.
        weights: optional scoring-weight overrides (defaults preserve behavior).
        write: persist recommendation nodes + proposed edges (False = analysis).

    Returns:
        A :class:`ConsolidationReport` with recommendations ranked by priority.
    """
    assets = assets if assets is not None else read_assets(engine)
    # Stash for score_group (avoids threading the map through every call).
    engine._asset_cache = assets  # type: ignore[attr-defined]

    groups = read_cohorts(engine, assets=assets, min_sources=min_sources)
    report = ConsolidationReport(groups_considered=len(groups))
    if not groups:
        return report

    drift = asset_drift_map(engine, assets=assets)
    all_members = {m for g in groups for m in g.members}
    lineage = _lineage_counts(engine, all_members)
    max_cost = 0.0
    for g in groups:
        c = sum(_to_float(assets.get(m, {}).get("license_cost", 0)) for m in g.members)
        c += sum(_to_float(assets.get(m, {}).get("annual_cost", 0)) for m in g.members)
        max_cost = max(max_cost, c)

    recs = [
        score_group(engine, g, drift, lineage, max_cost=max_cost, weights=weights)
        for g in groups
    ]
    recs.sort(key=lambda r: r.priority, reverse=True)
    report.recommendations = recs[:top_n]

    if write:
        _clear_auto_absorbed(engine, all_members)
        for rec in report.recommendations:
            _persist_recommendation(engine, rec)
            report.nodes_written += 1
            report.edges_written += _persist_edges(engine, rec)
    return report


def _persist_recommendation(engine: Any, rec: ConsolidationRecommendation) -> None:
    """Persist a ``ConsolidationRecommendation`` node (propose-only)."""
    add = getattr(engine, "add_node", None)
    if not callable(add):
        return
    try:
        add(
            rec.rec_id,
            RegistryNodeType.CONSOLIDATION_RECOMMENDATION.value,
            properties={
                "kind": rec.kind,
                "status": "proposal",
                "members": rec.members,
                "north_star": rec.north_star,
                "capability": rec.capability,
                "standard": rec.standard,
                "value_score": rec.value_score,
                "risk_score": rec.risk_score,
                "priority": rec.priority,
                "rationale": rec.rationale,
                "sources": rec.sources,
                "concept": "KG-2.49",
            },
        )
    except Exception:  # noqa: BLE001
        pass


def _persist_edges(engine: Any, rec: ConsolidationRecommendation) -> int:
    """Write proposed ABSORBED_INTO edges + a REFERENCES edge to the standard."""
    link = getattr(engine, "link_nodes", None)
    if not callable(link):
        return 0
    written = 0
    for member in rec.members:
        if member == rec.north_star:
            continue
        try:
            link(
                member,
                rec.north_star,
                RegistryEdgeType.ABSORBED_INTO,
                properties={
                    "_rel": "ABSORBED_INTO",
                    "auto": True,
                    "status": "proposed",
                    "recommendation": rec.rec_id,
                    "concept": "KG-2.49",
                },
            )
            written += 1
        except Exception:  # noqa: BLE001
            pass
    if rec.standard:
        try:
            link(
                rec.rec_id,
                standard_node_id(rec.standard),
                RegistryEdgeType.REFERENCES,
                properties={"_rel": "REFERENCES", "auto": True, "concept": "KG-2.49"},
            )
            written += 1
        except Exception:  # noqa: BLE001
            pass
    return written


__all__ = [
    "ConsolidationRecommendation",
    "ConsolidationReport",
    "DEFAULT_WEIGHTS",
    "score_group",
    "recommend_consolidations",
]
