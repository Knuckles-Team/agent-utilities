"""Tests for CONCEPT:AU-KG.research.research-pipeline-runner — Risk Management Engine."""

import pytest

from agent_utilities.domains.finance.risk_manager import (
    PreTradeGuard,
    RiskLimits,
    RiskManager,
    StressTestEngine,
    VaRCalculator,
)
from agent_utilities.numeric import xp as np


class TestPreTradeGuard:
    def test_approved_trade(self):
        guard = PreTradeGuard()
        result = guard.validate(
            _order_side="buy",
            order_quantity=10,
            order_price=100.0,
            portfolio_value=100_000,
            current_drawdown=0.05,
        )
        assert result.approved is True
        assert len(result.violations) == 0

    def test_position_size_violation(self):
        guard = PreTradeGuard(RiskLimits(max_position_pct=0.05))
        result = guard.validate(
            _order_side="buy",
            order_quantity=100,
            order_price=100.0,
            portfolio_value=100_000,
        )
        assert result.approved is False
        assert any("Position size" in v for v in result.violations)

    def test_drawdown_violation(self):
        guard = PreTradeGuard(RiskLimits(max_drawdown_pct=0.10))
        result = guard.validate(
            _order_side="buy",
            order_quantity=1,
            order_price=10.0,
            portfolio_value=100_000,
            current_drawdown=0.15,
        )
        assert result.approved is False
        assert any("drawdown" in v.lower() for v in result.violations)

    def test_sector_concentration_violation(self):
        guard = PreTradeGuard(RiskLimits(max_sector_concentration=0.20))
        result = guard.validate(
            _order_side="buy",
            order_quantity=100,
            order_price=100.0,
            portfolio_value=100_000,
            sector_exposure=0.15,
        )
        assert result.approved is False

    def test_risk_score_increases_with_violations(self):
        guard = PreTradeGuard(RiskLimits(max_position_pct=0.01, max_drawdown_pct=0.01))
        result = guard.validate(
            _order_side="buy",
            order_quantity=100,
            order_price=100.0,
            portfolio_value=100_000,
            current_drawdown=0.15,
        )
        assert result.risk_score > 0.5


class TestVaRCalculator:
    @pytest.fixture
    def returns(self):
        rng = np.random.default_rng(42)
        return rng.normal(0.0005, 0.02, 500)

    def test_historical_var(self, returns):
        calc = VaRCalculator()
        result = calc.historical(returns)
        assert result.method == "historical"
        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.cvar_95 >= result.var_95
        assert result.n_observations == 500

    def test_parametric_var(self, returns):
        calc = VaRCalculator()
        result = calc.parametric(returns)
        assert result.method == "parametric"
        assert result.var_95 > 0
        assert result.var_99 > result.var_95

    def test_monte_carlo_var(self, returns):
        calc = VaRCalculator()
        result = calc.monte_carlo(returns, n_simulations=5000)
        assert result.method == "monte_carlo"
        assert result.var_95 > 0

    def test_insufficient_data(self):
        calc = VaRCalculator()
        result = calc.historical(np.array([0.01, -0.01]))
        assert result.var_95 == 0.0


class TestStressTestEngine:
    def test_market_crash_scenario(self):
        engine = StressTestEngine()
        positions = {
            "AAPL": {"value": 50_000, "asset_class": "equity"},
            "BTC": {"value": 30_000, "asset_class": "crypto"},
        }
        result = engine.run_scenario("market_crash_2008", positions, 100_000)
        assert result.pnl_impact < 0
        assert result.portfolio_value_after < 100_000

    def test_bull_market_scenario(self):
        engine = StressTestEngine()
        positions = {"AAPL": {"value": 50_000, "asset_class": "equity"}}
        result = engine.run_scenario("bull_market_rally", positions, 100_000)
        assert result.pnl_impact > 0

    def test_custom_shocks(self):
        engine = StressTestEngine()
        positions = {"GOLD": {"value": 20_000, "asset_class": "commodity"}}
        result = engine.run_scenario(
            "custom",
            positions,
            100_000,
            custom_shocks={"commodity": -0.15},
        )
        assert result.pnl_impact == pytest.approx(-3000, rel=0.01)

    def test_unknown_scenario_returns_empty(self):
        engine = StressTestEngine()
        result = engine.run_scenario("nonexistent", {}, 100_000)
        assert result.pnl_impact == 0.0


class TestRiskManagerFacade:
    def test_check_order(self):
        rm = RiskManager()
        result = rm.check_order(
            _order_side="buy",
            order_quantity=5,
            order_price=100.0,
            portfolio_value=100_000,
        )
        assert result.approved is True

    def test_compute_var(self):
        rm = RiskManager()
        returns = np.random.default_rng(42).normal(0, 0.02, 200)
        result = rm.compute_var(returns, method="historical")
        assert result.var_95 > 0

    def test_run_stress_test(self):
        rm = RiskManager()
        positions = {"SPY": {"value": 80_000, "asset_class": "equity"}}
        result = rm.run_stress_test("covid_march_2020", positions, 100_000)
        assert result.pnl_impact < 0


class TestVaRCalculatorEngineLivePath:
    """Live-path test (CONCEPT:AU-KG.memory.mementified-context): VaRCalculator.historical routes to the
    Rust epistemic-graph engine when reachable. Auto-skips when the engine is
    not running so offline/unit environments are unaffected."""

    def test_historical_uses_engine_when_available(self):
        import agent_utilities.domains.finance.risk_manager as rm

        # Reset the cached probe so engine availability is re-evaluated here.
        rm._ENGINE_PROBED = False
        rm._ENGINE_CLIENT = None
        client = rm._risk_engine()
        if client is None:
            pytest.skip("epistemic-graph engine not reachable")

        returns = np.random.default_rng(0).normal(0, 0.01, 500)
        direct = client.finance.risk_metrics(returns.tolist(), 0.0)
        result = VaRCalculator().historical(returns)

        # The VaRCalculator result must equal the engine's own computation,
        # proving the live call path actually invoked the engine.
        assert abs(result.var_95 - float(direct["var_95"])) < 1e-9
        assert abs(result.var_99 - float(direct["var_99"])) < 1e-9
        assert abs(result.cvar_95 - float(direct["cvar_95"])) < 1e-9
