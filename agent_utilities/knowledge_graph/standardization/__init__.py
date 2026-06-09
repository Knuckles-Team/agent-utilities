#!/usr/bin/python
from __future__ import annotations

"""Enterprise Standardization & Consolidation engine (CONCEPT:KG-2.49).

The decision brain that turns a fully-harvested enterprise knowledge graph into
(a) an enforceable enterprise-standard model and (b) ranked, propose-only
"collapse these things" recommendations. Built entirely on existing fabric:

  - **Standards as interface types** (:mod:`standards`) — the north-star contract
    per capability domain, reusing the Foundry-parity interface layer; orgs
    implement with their own concrete types + sanctioned extensions.
  - **Drift** (:mod:`drift`) — per-asset/org/domain conformance scoring from
    :meth:`Interface.gaps_for`, written as idempotent ``CONFORMS_TO`` edges.
  - **Cohorts** (:mod:`cohorts`) — capability cohorts (P22), dedup ``SUPERSEDES``
    clusters, and synergy bundles normalized into candidate groups.
  - **Consolidation** (:mod:`consolidation`) — two-axis (value/risk) ranking that
    emits ``ConsolidationRecommendation`` nodes + proposed ``ABSORBED_INTO`` edges.

Reached from the execution plane through ``kg.ontology`` (the OntologySystem binds
:data:`ENTERPRISE_STANDARD_REGISTRY` as ``ontology.standards``) and exposed over
the ``graph_orchestrate(action="standardize")`` MCP action. Mirrors
``research/golden_loop.run_assimilation_pass`` — a single propose-only pass the
daemon can schedule.
"""

from typing import Any

from .cohorts import CandidateGroup, capability_cohorts, read_assets, read_cohorts
from .consolidation import (
    ConsolidationRecommendation,
    ConsolidationReport,
    recommend_consolidations,
    score_group,
)
from .drift import (
    DomainDrift,
    DriftReport,
    OrgDrift,
    asset_drift_map,
    score_conformance,
)
from .standards import (
    ENTERPRISE_STANDARD_NODE,
    ENTERPRISE_STANDARD_REGISTRY,
    STANDARD_DOMAINS,
    applicable_standard,
    drift_score,
    materialize_standards,
    register_enterprise_standards,
    standard_names,
    standard_node_id,
)


def run_standardization_pass(
    engine: Any = None,
    *,
    top_n: int = 20,
    min_sources: int = 2,
    weights: dict[str, float] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Run one propose-only standardization + consolidation pass (CONCEPT:KG-2.49).

    materialize standards → score conformance drift → rank consolidation
    candidates. Idempotent (CONFORMS_TO / ABSORBED_INTO edges are cleared before
    re-write; recommendation nodes are keyed by member-set hash). Returns a
    JSON-able report consumed by ``graph_orchestrate(action="standardize")`` and
    the golden-loop ``standardize`` stage.

    Args:
        engine: knowledge engine (defaults to the active IntelligenceGraphEngine).
        top_n: number of top-priority consolidation recommendations to persist.
        min_sources: minimum distinct vendors for a capability cohort.
        weights: optional consolidation scoring-weight overrides.
        write: persist standards/edges/recommendations (False = analysis only).
    """
    if engine is None:
        from ..core.engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine.get_active() or IntelligenceGraphEngine()

    standards_written = materialize_standards(engine) if write else 0
    assets = read_assets(engine)
    drift = score_conformance(engine, assets=assets, write=write)
    consolidation = recommend_consolidations(
        engine,
        assets=assets,
        top_n=top_n,
        min_sources=min_sources,
        weights=weights,
        write=write,
    )

    return {
        "standards_materialized": standards_written,
        "assets_scored": drift.assets_scored,
        "conformance_edges": drift.edges_written,
        "drift_by_domain": {
            name: {
                "asset_count": dd.asset_count,
                "conformant": dd.conformant,
                "mean_drift": dd.mean_drift,
            }
            for name, dd in drift.per_domain.items()
        },
        "drift_by_org": {
            org: {"asset_count": od.asset_count, "mean_drift": od.mean_drift}
            for org, od in drift.per_org.items()
        },
        "groups_considered": consolidation.groups_considered,
        "recommendations": [
            {
                "rec_id": r.rec_id,
                "kind": r.kind,
                "north_star": r.north_star,
                "members": r.members,
                "capability": r.capability,
                "standard": r.standard,
                "priority": r.priority,
                "value_score": r.value_score,
                "risk_score": r.risk_score,
                "rationale": r.rationale,
            }
            for r in consolidation.recommendations
        ],
    }


__all__ = [
    # Orchestrator
    "run_standardization_pass",
    # Standards (KG-2.49)
    "ENTERPRISE_STANDARD_REGISTRY",
    "ENTERPRISE_STANDARD_NODE",
    "STANDARD_DOMAINS",
    "register_enterprise_standards",
    "standard_names",
    "applicable_standard",
    "drift_score",
    "materialize_standards",
    "standard_node_id",
    # Drift (KG-2.49)
    "DriftReport",
    "DomainDrift",
    "OrgDrift",
    "score_conformance",
    "asset_drift_map",
    # Cohorts (KG-2.49)
    "CandidateGroup",
    "read_assets",
    "read_cohorts",
    "capability_cohorts",
    # Consolidation (KG-2.49)
    "ConsolidationRecommendation",
    "ConsolidationReport",
    "recommend_consolidations",
    "score_group",
]
