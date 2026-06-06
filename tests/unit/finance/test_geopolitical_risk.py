"""Geopolitical Risk Scoring tests — CONCEPT:KG-2.30.

Overlap-based exposure scoring, the "which holdings are exposed to risk X" query,
KG/OWL fact persistence, and LIVE-PATH wiring into the existing StressTestEngine
and regime machinery.
"""

from __future__ import annotations

from agent_utilities.domains.finance.geopolitical_risk import (
    AssetExposure,
    GeopoliticalRiskFactor,
    RiskCategory,
    apply_geopolitical_stress,
    asset_exposure_to_factor,
    exposed_holdings,
    geopolitical_facts_batch,
    risk_to_regime_flag,
    risk_to_stress_shocks,
    score_portfolio,
    seed_geopolitical_risk,
)
from agent_utilities.domains.finance.risk_manager import StressTestEngine


class _FakeBackend:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node_id, type=None, **props):
        self.nodes.append((node_id, type, props))

    def add_edge(self, src, tgt, rel_type=None):
        self.edges.append((src, tgt, rel_type))


def _taiwan_conflict():
    return GeopoliticalRiskFactor(
        id="taiwan_strait",
        name="Taiwan Strait conflict",
        category=RiskCategory.CONFLICT,
        severity=0.8,
        likelihood=0.4,
        sectors=["semiconductors", "technology"],
        regions=["taiwan", "china"],
    )


def _sanctions():
    return GeopoliticalRiskFactor(
        id="russia_sanctions",
        name="Russia energy sanctions",
        category=RiskCategory.SANCTIONS,
        severity=0.7,
        likelihood=0.6,
        sectors=["energy"],
        regions=["russia", "europe"],
    )


def test_exposure_via_sector_and_region_overlap():
    factor = _taiwan_conflict()
    tsmc = AssetExposure(
        asset_id="TSM",
        weight=0.3,
        sector_weights={"semiconductors": 1.0},
        region_weights={"taiwan": 1.0},
    )
    unrelated = AssetExposure(
        asset_id="KO",
        weight=0.3,
        sector_weights={"consumer_staples": 1.0},
        region_weights={"usa": 1.0},
    )
    assert asset_exposure_to_factor(tsmc, factor) > 0.2
    assert asset_exposure_to_factor(unrelated, factor) == 0.0


def test_expected_severity_scales_with_likelihood():
    f = _taiwan_conflict()
    assert f.expected_severity == 0.8 * 0.4


def test_portfolio_score_and_attribution():
    holdings = [
        AssetExposure("TSM", 0.4, sector_weights={"semiconductors": 1.0}),
        AssetExposure("XOM", 0.3, sector_weights={"energy": 1.0}),
        AssetExposure("KO", 0.3, sector_weights={"consumer_staples": 1.0}),
    ]
    factors = [_taiwan_conflict(), _sanctions()]
    score = score_portfolio(holdings, factors)
    assert 0.0 < score.score <= 1.0
    assert score.per_asset["TSM"] > score.per_asset["KO"]
    assert score.dominant_factor in {"taiwan_strait", "russia_sanctions"}


def test_exposed_holdings_query():
    """The KG-query payoff: which holdings are exposed to risk X."""
    holdings = [
        AssetExposure("TSM", 0.4, sector_weights={"semiconductors": 1.0}),
        AssetExposure("NVDA", 0.3, sector_weights={"semiconductors": 1.0}),
        AssetExposure("KO", 0.3, sector_weights={"consumer_staples": 1.0}),
    ]
    exposed = exposed_holdings(holdings, _taiwan_conflict())
    ids = [a for a, _ in exposed]
    assert "TSM" in ids and "NVDA" in ids
    assert "KO" not in ids
    # sorted descending by exposure
    assert exposed == sorted(exposed, key=lambda kv: kv[1], reverse=True)


def test_facts_batch_emits_owl_shaped_edges():
    factors = [_taiwan_conflict()]
    holdings = [AssetExposure("TSM", 0.4, sector_weights={"semiconductors": 1.0})]
    batch = geopolitical_facts_batch(factors, holdings)
    types = {n.type for n in batch.nodes}
    assert "GeopoliticalRisk" in types
    assert "Sector" in types
    assert "Region" in types
    rels = {e.rel_type for e in batch.edges}
    assert "affectsSector" in rels
    assert "affectsRegion" in rels
    assert "exposedTo" in rels


def test_seed_to_kg_and_none_noop():
    backend = _FakeBackend()
    n, e = seed_geopolitical_risk(
        backend,
        [_taiwan_conflict()],
        [AssetExposure("TSM", 0.4, sector_weights={"semiconductors": 1.0})],
    )
    assert n > 0 and e > 0
    assert seed_geopolitical_risk(None, [_taiwan_conflict()]) == (0, 0)
    assert seed_geopolitical_risk(backend, []) == (0, 0)


def test_risk_to_stress_shocks_shape_matches_engine():
    shocks = risk_to_stress_shocks([_taiwan_conflict(), _sanctions()])
    # Exact shape StressTestEngine.run_scenario(custom_shocks=...) consumes.
    assert set(shocks) <= {"equity", "bond", "commodity", "crypto"}
    assert shocks["equity"] < 0  # conflict + sanctions hurt equities
    assert shocks["commodity"] > 0  # both bid commodities
    assert all(-0.95 <= v <= 0.95 for v in shocks.values())


def test_stress_engine_consumes_geopolitical_shocks_live_path():
    """LIVE-PATH: a geopolitical scenario flows into the unchanged StressTestEngine
    and produces a real P&L impact."""
    engine = StressTestEngine()
    positions = {
        "TSM": {"value": 40000.0, "asset_class": "equity"},
        "GLD": {"value": 20000.0, "asset_class": "commodity"},
    }
    result = apply_geopolitical_stress(
        engine, [_taiwan_conflict(), _sanctions()], positions, portfolio_value=60000.0
    )
    assert result.scenario_name == "geopolitical"
    # Equity leg is shocked down; result is a genuine StressTestResult.
    assert result.pnl_impact != 0.0
    assert result.portfolio_value_after != 60000.0


def test_risk_to_regime_flag_thresholds():
    holdings = [AssetExposure("TSM", 1.0, sector_weights={"semiconductors": 1.0})]
    high = GeopoliticalRiskFactor(
        id="x",
        name="severe",
        category=RiskCategory.CONFLICT,
        severity=1.0,
        likelihood=1.0,
        sectors=["semiconductors"],
    )
    score = score_portfolio(holdings, [high])
    flag = risk_to_regime_flag(score)
    # A maximal, fully-exposed factor lands in an elevated regime label the
    # RegimeDetector also emits.
    assert flag in {"high_volatility", "bear_market"}
    # And the label is one the regime-aware path understands.
    assert flag in {
        "high_volatility",
        "bear_market",
        "sideways_market",
        "bull_market",
    }
