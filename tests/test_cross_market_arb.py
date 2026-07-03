"""Tests for CONCEPT:KG-2.6 — Cross Market Arbitrage."""

from agent_utilities.numeric import xp as np
import pytest

from agent_utilities.domains.finance.cross_market_arb import (
    CointegrationAnalyzer,
    CostAwareThresholdFilter,
    EventArbitrageEngine,
    OrnsteinUhlenbeckModel,
)


class TestCointegrationAnalyzer:
    def test_insufficient_data(self):
        a = np.array([1, 2])
        b = np.array([1, 2])
        assert not CointegrationAnalyzer.is_cointegrated(a, b)

    def test_mismatched_lengths(self):
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert not CointegrationAnalyzer.is_cointegrated(a, b)


class TestOrnsteinUhlenbeckModel:
    def test_optimal_thresholds(self):
        x_long, x_short = OrnsteinUhlenbeckModel.optimal_thresholds(
            theta=0.5, mu=0.0, c=0.01
        )
        # variance_buffer = 1 / 0.5 = 2.0
        # mu +/- (c + variance_buffer) = 0 +/- (0.01 + 2.0)
        assert np.isclose(x_long, -2.01)
        assert np.isclose(x_short, 2.01)

    def test_optimal_thresholds_zero_theta(self):
        x_long, x_short = OrnsteinUhlenbeckModel.optimal_thresholds(
            theta=0.0, mu=0.0, c=0.01
        )
        assert x_long == 0.0
        assert x_short == 0.0

    def test_calibrate_insufficient_data(self):
        with pytest.raises(ValueError):
            OrnsteinUhlenbeckModel.calibrate(np.array([1.0]), 1.0)

    def test_calibrate_constant_spread(self):
        # A constant spread will give b_hat = 1.0 roughly, returning 0 theta
        spread = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        res = OrnsteinUhlenbeckModel.calibrate(spread, 1.0)
        assert res["theta"] == 0.0


class TestPredictionMarketArbitrage:
    def test_cost_aware_threshold(self):
        # Model: 0.60, Market: 0.50, Cost: 0.08
        # Edge = 0.10 >= 0.08 (True)
        assert CostAwareThresholdFilter.passes_threshold(0.60, 0.50, 0.08) is True

        # Model: 0.55, Market: 0.50, Cost: 0.08
        # Edge = 0.05 < 0.08 (False)
        assert CostAwareThresholdFilter.passes_threshold(0.55, 0.50, 0.08) is False

    def test_event_arbitrage_engine(self):
        # Model: 0.70
        # Market A: 0.60 (Edge 0.10) -> Valid
        # Market B: 0.65 (Edge 0.05) -> Invalid (below 0.08 threshold)
        ops = EventArbitrageEngine.evaluate_dual_markets(
            model_probability=0.70,
            market_a_price=0.60,
            market_b_price=0.65,
            execution_costs=0.08,
        )
        assert "market_a" in ops
        assert "market_b" not in ops
        assert np.isclose(ops["market_a"], 0.10)
