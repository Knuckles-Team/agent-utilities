"""AU-CONTEXT-R007.1: boundary test pinning the current finance-math module set.

AU-CONTEXT-R007 moves finance math to epistemic-graph's finance core. Until
AU-CONTEXT-R007.2 lands that cutover and deletes the local modules, this
test pins the set of finance-math modules AU still owns and fails if a new
one is added -- the set may only shrink from here, never grow.
"""

from __future__ import annotations

from pathlib import Path

# The module set as of AU-CONTEXT-R007.1. AU-CONTEXT-R007.2 removes entries
# from this set as each calculation's parity-tested EG replacement lands; no
# entry is ever added back.
PINNED_FINANCE_MODULE_SET: frozenset[str] = frozenset(
    {
        # AU-CONTEXT-R008.1: calibrated, cited, informational-only
        # recommendation model -- AU-owned decision logic (not deterministic
        # finance math or a connector), so it stays local and is not part of
        # the AU-CONTEXT-R007 shrink-toward-EG set.
        "analysis_snapshot.py",
        "alpha_factors.py",
        "banking.py",
        "banking_models.py",
        "calibration_tracker.py",
        "composite_backtest.py",
        "copy_trade.py",
        "credit_quality.py",
        "cross_market_arb.py",
        "crypto_connector.py",
        "debate_engine.py",
        "engine_series.py",
        "errors.py",
        "execution.py",
        "features.py",
        "filing_diff.py",
        "forensic_screener.py",
        "geopolitical_risk.py",
        "__init__.py",
        "insider_equilibrium.py",
        "investor_debate.py",
        "kronos_forecaster.py",
        "market_data.py",
        "market_feeds.py",
        "microstructure.py",
        "pattern_classifier.py",
        "payments.py",
        "persona_heuristics.py",
        "portfolio_optimizer.py",
        "profit_attribution.py",
        "quant_mcp_tools.py",
        "quant_ontology.py",
        "regime_detector.py",
        "research_autopilot.py",
        "risk_manager.py",
        "sentiment_fusion.py",
        "signal_fusion.py",
        "strategy_export.py",
        "strategy_sharing.py",
        "streaming.py",
        "trade_journal.py",
        "trading_swarm.py",
        "versioned_orders.py",
        "visual_ta.py",
    }
)

# Roles the spec says AU retains permanently (not finance *math*): these
# never count against the shrink requirement.
RETAINED_AGENT_ROLE_MODULES: frozenset[str] = frozenset(
    {
        "trading_swarm.py",
        "debate_engine.py",
        "investor_debate.py",
        "research_autopilot.py",
        "persona_heuristics.py",
    }
)


def _finance_domain_dir() -> Path:
    # tests/unit/finance/test_r007_finance_module_boundary.py -> repo root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "agent_utilities" / "domains" / "finance"


def test_finance_module_set_has_not_grown_past_the_pinned_baseline() -> None:
    finance_dir = _finance_domain_dir()
    current = {p.name for p in finance_dir.glob("*.py")}

    unpinned_new_modules = current - PINNED_FINANCE_MODULE_SET
    assert not unpinned_new_modules, (
        "a new module was added to agent_utilities/domains/finance/ that is "
        "not in the AU-CONTEXT-R007.1 pinned baseline; the finance-math "
        f"module set may only shrink toward EG, not grow: {unpinned_new_modules}"
    )


def test_engine_finance_mixin_still_present_until_r007_2_cutover() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    engine_finance = (
        repo_root
        / "agent_utilities"
        / "knowledge_graph"
        / "orchestration"
        / "engine_finance.py"
    )
    assert engine_finance.is_file(), (
        "engine_finance.py is the live local finance-math module pinned by "
        "AU-CONTEXT-R007.1; AU-CONTEXT-R007.2 removes it only after EG's "
        "finance core has documented parity"
    )


def test_retained_agent_role_modules_are_a_subset_of_the_pinned_baseline() -> None:
    assert RETAINED_AGENT_ROLE_MODULES <= PINNED_FINANCE_MODULE_SET
