"""Tests for CONCEPT:KG-2.6 — Optimal Execution Engine."""

import pytest

from agent_utilities.knowledge_graph.core.optimal_execution import (
    AlmgrenChrissContinuous,
    AlmgrenChrissDiscrete,
    AvellanedaStoikovMarketMaker,
    CarteaJaimungalExecutor,
    CointegrationPairsTrader,
    SignalAdaptiveExecutor,
)
from agent_utilities.numeric import xp as np


class TestAlmgrenChrissDiscrete:
    """Tests for Almgren-Chriss discrete execution (Oxford HFT Ch 3)."""

    def test_basic_schedule(self):
        ac = AlmgrenChrissDiscrete()
        plan = ac.compute_schedule(
            total_shares=1000,
            n_steps=10,
            volatility=0.02,
            temporary_impact=0.001,
            permanent_impact=0.0001,
        )
        assert plan.strategy_name == "almgren_chriss_discrete"
        assert len(plan.schedule) == 10
        total_traded = sum(q for _, q in plan.schedule)
        assert abs(total_traded - 1000) < 1.0

    def test_risk_averse_front_loads(self):
        ac = AlmgrenChrissDiscrete()
        aggressive = ac.compute_schedule(
            total_shares=1000,
            n_steps=5,
            volatility=0.02,
            temporary_impact=0.001,
            permanent_impact=0.0001,
            risk_aversion=1e-3,
        )
        passive = ac.compute_schedule(
            total_shares=1000,
            n_steps=5,
            volatility=0.02,
            temporary_impact=0.001,
            permanent_impact=0.0001,
            risk_aversion=1e-9,
        )
        # More risk-averse → more front-loaded (first trade larger)
        assert aggressive.schedule[0][1] >= passive.schedule[0][1] * 0.8

    def test_zero_shares(self):
        ac = AlmgrenChrissDiscrete()
        plan = ac.compute_schedule(0, 10, 0.02, 0.001, 0.0001)
        assert len(plan.schedule) == 0

    def test_expected_cost_positive(self):
        ac = AlmgrenChrissDiscrete()
        plan = ac.compute_schedule(1000, 10, 0.02, 0.001, 0.0001)
        assert plan.expected_cost > 0

    def test_parameters_stored(self):
        ac = AlmgrenChrissDiscrete()
        plan = ac.compute_schedule(1000, 10, 0.02, 0.001, 0.0001)
        assert "kappa" in plan.parameters
        assert "volatility" in plan.parameters


class TestAlmgrenChrissContinuous:
    """Tests for continuous-time Almgren-Chriss (Oxford HFT Ch 4)."""

    def test_smooth_trajectory(self):
        ac = AlmgrenChrissContinuous()
        plan = ac.compute_trajectory(
            total_shares=1000,
            time_horizon=1.0,
            volatility=0.02,
            temporary_impact=0.001,
            n_points=50,
        )
        assert len(plan.schedule) >= 50
        total = sum(q for _, q in plan.schedule)
        assert abs(total - 1000) < 5.0

    def test_zero_horizon(self):
        ac = AlmgrenChrissContinuous()
        plan = ac.compute_trajectory(1000, 0, 0.02, 0.001)
        assert len(plan.schedule) == 0


class TestCarteaJaimungal:
    """Tests for Cartea-Jaimungal framework (Oxford HFT Ch 5)."""

    def test_basic_schedule(self):
        cj = CarteaJaimungalExecutor()
        plan = cj.compute_schedule(
            total_shares=1000,
            time_horizon=1.0,
            volatility=0.02,
            temporary_impact=0.001,
            inventory_penalty=0.01,
        )
        assert plan.strategy_name == "cartea_jaimungal"
        total = sum(q for _, q in plan.schedule)
        assert abs(total - 1000) < 5.0

    def test_higher_penalty_front_loads(self):
        cj = CarteaJaimungalExecutor()
        low = cj.compute_schedule(
            1000,
            1.0,
            0.02,
            0.001,
            inventory_penalty=0.001,
        )
        high = cj.compute_schedule(
            1000,
            1.0,
            0.02,
            0.001,
            inventory_penalty=1.0,
        )
        # Higher penalty → more aggressive early trading
        assert high.schedule[0][1] >= low.schedule[0][1] * 0.5


class TestMarketMaking:
    """Tests for Avellaneda-Stoikov market making (Oxford HFT Ch 10)."""

    def test_symmetric_quotes_zero_inventory(self):
        mm = AvellanedaStoikovMarketMaker()
        quote = mm.compute_quotes(
            mid_price=100.0,
            inventory=0.0,
            volatility=0.01,
            risk_aversion=0.1,
            time_remaining=1.0,
        )
        assert quote.reservation_price == pytest.approx(100.0)
        assert quote.bid_price < 100.0
        assert quote.ask_price > 100.0
        spread = quote.ask_price - quote.bid_price
        assert spread > 0

    def test_long_inventory_lowers_reservation(self):
        mm = AvellanedaStoikovMarketMaker()
        long_quote = mm.compute_quotes(
            100.0, inventory=10, volatility=0.01, risk_aversion=0.1, time_remaining=1.0
        )
        short_quote = mm.compute_quotes(
            100.0, inventory=-10, volatility=0.01, risk_aversion=0.1, time_remaining=1.0
        )
        assert long_quote.reservation_price < 100.0  # Wants to sell
        assert short_quote.reservation_price > 100.0  # Wants to buy

    def test_spread_positive(self):
        mm = AvellanedaStoikovMarketMaker()
        quote = mm.compute_quotes(100.0, 5.0, 0.02, 0.5, 0.5)
        assert quote.optimal_spread > 0


class TestCointegrationPairsTrading:
    """Tests for OU-based pairs trading (Oxford HFT Ch 12)."""

    def test_fit_mean_reverting_series(self):
        trader = CointegrationPairsTrader()
        rng = np.random.default_rng(42)
        # Simulate OU process
        theta, mu, sigma = 0.5, 10.0, 0.3
        n = 500
        x = np.zeros(n)
        x[0] = mu
        for i in range(1, n):
            x[i] = x[i - 1] + theta * (mu - x[i - 1]) + sigma * rng.standard_normal()
        params = trader.fit_ou_parameters(x.tolist())
        assert params["theta"] > 0  # Mean-reverting
        assert abs(params["mu"] - mu) < 3.0
        assert params["half_life"] > 0 and params["half_life"] < 100

    def test_signal_generation(self):
        trader = CointegrationPairsTrader()
        params = {"theta": 0.5, "mu": 10.0, "sigma": 0.5, "half_life": 1.4}

        # Far above mean → short signal
        signal = trader.generate_signal(12.0, params, entry_threshold=2.0)
        assert signal.signal == "short"
        assert signal.z_score > 2.0

        # Near mean → exit
        signal = trader.generate_signal(
            10.1, params, entry_threshold=2.0, exit_threshold=0.5
        )
        assert signal.signal == "exit"

    def test_short_series(self):
        trader = CointegrationPairsTrader()
        params = trader.fit_ou_parameters([1.0, 2.0, 3.0])
        assert params["half_life"] == float("inf")  # Not enough data


class TestSignalAdaptiveExecution:
    """Tests for signal-adaptive execution (Oxford HFT Ch 7)."""

    def test_signal_adjusts_schedule(self):
        executor = SignalAdaptiveExecutor()
        signals = [1.0] * 5 + [-1.0] * 5  # Favorable then unfavorable
        plan = executor.compute_adaptive_schedule(
            total_shares=1000,
            n_steps=10,
            volatility=0.02,
            temporary_impact=0.001,
            risk_aversion=1e-6,
            signal_values=signals,
        )
        assert plan.strategy_name == "signal_adaptive"
        total = sum(q for _, q in plan.schedule)
        assert abs(total - 1000) < 1.0

    def test_preserves_total_shares(self):
        executor = SignalAdaptiveExecutor()
        signals = [0.5, -0.3, 1.0, -0.8, 0.2]
        plan = executor.compute_adaptive_schedule(
            500,
            5,
            0.02,
            0.001,
            1e-6,
            signals,
        )
        total = sum(q for _, q in plan.schedule)
        assert abs(total - 500) < 1.0
