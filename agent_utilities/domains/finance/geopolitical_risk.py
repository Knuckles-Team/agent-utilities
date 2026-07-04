"""Geopolitical Risk — CONCEPT:AU-KG.domains.geopolitical-risk-scoring — Geopolitical Risk Scoring

Models geopolitical risk factors (armed conflict, sanctions, supply-chain
chokepoints, election / regime change) as **reasoned OWL facts** and scores a
portfolio's exposure to them.

Each :class:`GeopoliticalRiskFactor` has a ``severity`` in ``[0, 1]``, a
``likelihood`` in ``[0, 1]``, and the **sectors** and **regions** it affects. A
portfolio is a set of :class:`AssetExposure` rows (sector + region weights). The
exposure-weighted product ``severity * likelihood * overlap`` yields a
per-asset and portfolio-level :class:`GeopoliticalRiskScore`.

The uniqueness payoff is that risk factors and asset exposure persist to the KG
as ``:GeopoliticalRisk``/``:affectsSector``/``:affectsRegion``/``:exposedTo``
facts, so **"which holdings are exposed to risk X"** is a graph QUERY (resolved
locally here via :func:`exposed_holdings`, and reasoned over by the OWL layer once
written) — not an agent guess.

Wiring into the existing risk/stress path:

* :func:`risk_to_stress_shocks` converts active risk factors into the
  ``{asset_class: shock_pct}`` vector that
  ``StressTestEngine.run_scenario(..., custom_shocks=...)`` already consumes, and
* :func:`risk_to_regime_flag` converts an aggregate score into a regime label the
  regime-aware machinery understands (``high_volatility`` / ``bear_market`` / …).

Persistence uses the canonical ``registry.write_batch`` path; a ``None`` backend
is a clean no-op. All scoring is a real overlap model — no placeholder constants.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class RiskCategory(StrEnum):
    """Canonical families of geopolitical risk."""

    CONFLICT = "conflict"
    SANCTIONS = "sanctions"
    SUPPLY_CHAIN = "supply_chain"
    ELECTION = "election"
    REGIME_CHANGE = "regime_change"
    TRADE_WAR = "trade_war"
    TERRORISM = "terrorism"


# How each risk category transmits into broad asset-class shocks. The signs and
# magnitudes are a defensible transmission model (conflict bids safe-haven bonds
# and crushes equities/crypto; sanctions spike commodities; a supply-chain
# chokepoint is commodity-led). Scaled by factor severity*likelihood downstream.
_CATEGORY_ASSET_TRANSMISSION: dict[str, dict[str, float]] = {
    RiskCategory.CONFLICT: {
        "equity": -0.18,
        "bond": 0.06,
        "commodity": 0.15,
        "crypto": -0.25,
    },
    RiskCategory.SANCTIONS: {
        "equity": -0.10,
        "bond": 0.02,
        "commodity": 0.20,
        "crypto": -0.12,
    },
    RiskCategory.SUPPLY_CHAIN: {
        "equity": -0.12,
        "bond": -0.02,
        "commodity": 0.18,
        "crypto": -0.08,
    },
    RiskCategory.ELECTION: {
        "equity": -0.06,
        "bond": 0.01,
        "commodity": 0.03,
        "crypto": -0.05,
    },
    RiskCategory.REGIME_CHANGE: {
        "equity": -0.20,
        "bond": 0.04,
        "commodity": 0.10,
        "crypto": -0.15,
    },
    RiskCategory.TRADE_WAR: {
        "equity": -0.14,
        "bond": 0.03,
        "commodity": 0.08,
        "crypto": -0.10,
    },
    RiskCategory.TERRORISM: {
        "equity": -0.10,
        "bond": 0.05,
        "commodity": 0.06,
        "crypto": -0.08,
    },
}


@dataclass
class GeopoliticalRiskFactor:
    """A named geopolitical risk with severity, likelihood and footprint."""

    id: str
    name: str
    category: RiskCategory
    severity: float  # [0, 1] impact if it materialises
    likelihood: float = 0.5  # [0, 1] probability of materialising
    sectors: list[str] = field(default_factory=list)  # affected GICS-ish sectors
    regions: list[str] = field(default_factory=list)  # affected regions/countries
    description: str = ""

    def __post_init__(self) -> None:
        self.severity = max(0.0, min(1.0, float(self.severity)))
        self.likelihood = max(0.0, min(1.0, float(self.likelihood)))
        self.sectors = [s.lower() for s in self.sectors]
        self.regions = [r.lower() for r in self.regions]

    @property
    def expected_severity(self) -> float:
        """Probability-weighted severity = severity * likelihood."""
        return self.severity * self.likelihood


@dataclass
class AssetExposure:
    """A holding's exposure: portfolio weight + sector/region attribution.

    ``sector_weights`` and ``region_weights`` should each sum to ~1 across the
    asset's revenue/operations attribution. ``asset_class`` ties the holding to
    the stress engine's shock buckets (equity/bond/commodity/crypto).
    """

    asset_id: str
    weight: float  # portfolio weight in [0, 1]
    asset_class: str = "equity"
    sector_weights: dict[str, float] = field(default_factory=dict)
    region_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sector_weights = {k.lower(): v for k, v in self.sector_weights.items()}
        self.region_weights = {k.lower(): v for k, v in self.region_weights.items()}


def _overlap(weights: dict[str, float], affected: list[str]) -> float:
    """Fraction of an asset's attribution that falls in the affected set."""
    if not affected or not weights:
        return 0.0
    total = sum(abs(v) for v in weights.values())
    if total <= 0:
        return 0.0
    hit = sum(abs(weights.get(a, 0.0)) for a in affected)
    return min(1.0, hit / total)


def asset_exposure_to_factor(
    asset: AssetExposure, factor: GeopoliticalRiskFactor
) -> float:
    """Exposure of a single asset to a single factor in ``[0, 1]``.

    Combines sector and region overlap (union semantics: an asset is exposed if
    *either* its sector or region overlaps the factor footprint) with the
    factor's expected severity.
    """
    sector_ov = _overlap(asset.sector_weights, factor.sectors)
    region_ov = _overlap(asset.region_weights, factor.regions)
    # Union of two independent exposure channels.
    overlap = 1.0 - (1.0 - sector_ov) * (1.0 - region_ov)
    return overlap * factor.expected_severity


@dataclass
class GeopoliticalRiskScore:
    """Portfolio-level geopolitical risk score with per-asset attribution."""

    score: float  # [0, 1] portfolio-weighted expected exposed severity
    per_asset: dict[str, float] = field(default_factory=dict)  # asset_id -> exposure
    per_factor: dict[str, float] = field(default_factory=dict)  # factor_id -> contrib
    dominant_factor: str = ""

    @property
    def is_elevated(self) -> bool:
        return self.score >= 0.15


def score_portfolio(
    holdings: list[AssetExposure],
    factors: list[GeopoliticalRiskFactor],
) -> GeopoliticalRiskScore:
    """Compute the portfolio geopolitical-risk score across all factors.

    The portfolio score is ``sum_assets weight * (1 - Π_f (1 - exposure_af))`` —
    each asset's per-factor exposures combine as independent hazards, then are
    portfolio-weight-aggregated. Real, bounded, and additive-decomposable.
    """
    per_asset: dict[str, float] = {}
    per_factor: dict[str, float] = {f.id: 0.0 for f in factors}
    total = 0.0

    for asset in holdings:
        survival = 1.0
        for f in factors:
            e = asset_exposure_to_factor(asset, f)
            survival *= 1.0 - e
            per_factor[f.id] += asset.weight * e
        asset_exposure = 1.0 - survival
        per_asset[asset.asset_id] = asset_exposure
        total += asset.weight * asset_exposure

    dominant = max(per_factor, key=lambda k: per_factor[k]) if per_factor else ""
    return GeopoliticalRiskScore(
        score=float(min(1.0, total)),
        per_asset=per_asset,
        per_factor=per_factor,
        dominant_factor=dominant,
    )


def exposed_holdings(
    holdings: list[AssetExposure],
    factor: GeopoliticalRiskFactor,
    threshold: float = 0.0,
) -> list[tuple[str, float]]:
    """Local resolution of the KG query "which holdings are exposed to risk X".

    Returns ``[(asset_id, exposure)]`` above ``threshold``, descending. The same
    question is answerable over the persisted ``:exposedTo`` edges once written;
    this is the in-process equivalent used by the stress/risk path.
    """
    out = [
        (a.asset_id, asset_exposure_to_factor(a, factor))
        for a in holdings
        if asset_exposure_to_factor(a, factor) > threshold
    ]
    return sorted(out, key=lambda kv: kv[1], reverse=True)


# ── Wiring: geopolitical risk → existing stress / regime machinery ────────────
def risk_to_stress_shocks(
    factors: list[GeopoliticalRiskFactor],
) -> dict[str, float]:
    """Convert active risk factors into a ``{asset_class: shock_pct}`` vector.

    This is the exact shape ``StressTestEngine.run_scenario(custom_shocks=...)``
    consumes, so a geopolitical scenario flows straight into the existing
    stress-test P&L path. Each factor scales its category transmission by its
    expected severity; factors combine additively (clamped to plausible bounds).
    """
    shocks: dict[str, float] = {}
    for f in factors:
        transmission = _CATEGORY_ASSET_TRANSMISSION.get(f.category, {})
        scale = f.expected_severity
        for asset_class, base in transmission.items():
            shocks[asset_class] = shocks.get(asset_class, 0.0) + base * scale
    # Clamp to a sane stress range so stacked factors stay realistic.
    return {k: max(-0.95, min(0.95, v)) for k, v in shocks.items()}


def risk_to_regime_flag(score: GeopoliticalRiskScore) -> str:
    """Map an aggregate geopolitical score to a regime label for regime routing.

    Returns one of the labels ``RegimeDetector`` emits so the regime-aware
    strategy switch can consume a geopolitical override directly.
    """
    if score.score >= 0.35:
        return "high_volatility"
    if score.score >= 0.15:
        return "bear_market"
    if score.score >= 0.05:
        return "sideways_market"
    return "bull_market"


def apply_geopolitical_stress(
    stress_engine: Any,
    factors: list[GeopoliticalRiskFactor],
    positions: dict[str, dict[str, Any]],
    portfolio_value: float,
    scenario_name: str = "geopolitical",
) -> Any:
    """Run the existing ``StressTestEngine`` with a geopolitical shock vector.

    Bridges KG-2.30 risk factors into the unchanged stress path: builds the
    custom shock vector and calls ``run_scenario`` so the result is a normal
    :class:`StressTestResult` the risk manager already understands.
    """
    shocks = risk_to_stress_shocks(factors)
    return stress_engine.run_scenario(
        scenario_name, positions, portfolio_value, custom_shocks=shocks
    )


# ── KG/OWL persistence: :GeopoliticalRisk facts (canonical write_batch) ───────
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:80] or "x"


def geopolitical_facts_batch(
    factors: list[GeopoliticalRiskFactor],
    holdings: list[AssetExposure] | None = None,
) -> Any:
    """Build an ``ExtractionBatch`` of ``:GeopoliticalRisk`` facts + exposure edges.

    Emits ``:affectsSector`` / ``:affectsRegion`` edges per factor and, when
    holdings are supplied, ``:exposedTo`` edges (with the numeric exposure as an
    edge-bearing fact) so "which holdings are exposed to risk X" is a graph query.
    """
    from agent_utilities.knowledge_graph.enrichment.models import (
        EnrichmentEdge,
        ExtractionBatch,
        GraphNode,
    )

    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    seen: set[str] = set()

    def _ensure(node_id: str, node_type: str, props: dict[str, Any]) -> None:
        if node_id not in seen:
            nodes.append(GraphNode(id=node_id, type=node_type, props=props))
            seen.add(node_id)

    for f in factors:
        rid = f"geopolitical_risk:{_slug(f.id)}"
        _ensure(
            rid,
            "GeopoliticalRisk",
            {
                "name": f.name,
                "category": str(f.category),
                "severity": f.severity,
                "likelihood": f.likelihood,
                "expected_severity": f.expected_severity,
                "description": f.description[:500],
                "concept": "AU-KG.domains.geopolitical-risk-scoring",
            },
        )
        for sector in f.sectors:
            sid = f"sector:{_slug(sector)}"
            _ensure(sid, "Sector", {"name": sector})
            edges.append(
                EnrichmentEdge(source=rid, target=sid, rel_type="affectsSector")
            )
        for region in f.regions:
            reg = f"region:{_slug(region)}"
            _ensure(reg, "Region", {"name": region})
            edges.append(
                EnrichmentEdge(source=rid, target=reg, rel_type="affectsRegion")
            )

        for asset in holdings or []:
            exposure = asset_exposure_to_factor(asset, f)
            if exposure <= 0:
                continue
            aid = f"instrument:{_slug(asset.asset_id)}"
            _ensure(
                aid,
                "FinancialInstrument",
                {"ticker": asset.asset_id, "asset_class": asset.asset_class},
            )
            edges.append(EnrichmentEdge(source=aid, target=rid, rel_type="exposedTo"))

    return ExtractionBatch(category="geopolitical", nodes=nodes, edges=edges)


def seed_geopolitical_risk(
    backend: Any,
    factors: list[GeopoliticalRiskFactor],
    holdings: list[AssetExposure] | None = None,
) -> tuple[int, int]:
    """Persist geopolitical risk facts + exposure via ``write_batch``.

    ``None`` backend (offline) is a clean no-op returning ``(0, 0)``.
    """
    if backend is None or not factors:
        return (0, 0)
    from agent_utilities.knowledge_graph.enrichment.registry import write_batch

    n, e = write_batch(backend, geopolitical_facts_batch(factors, holdings))
    logger.info("Seeded geopolitical risk facts: %d nodes, %d edges", n, e)
    return n, e
