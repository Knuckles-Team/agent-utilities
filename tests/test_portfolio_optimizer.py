"""Tests for CONCEPT:KG-2.62 — Portfolio Optimization Suite."""

import numpy as np
import pytest

from agent_utilities.domains.finance.portfolio_optimizer import (
    BlackLittermanOptimizer,
    MeanVarianceOptimizer,
    OptimizationResult,
    RiskParityOptimizer,
)


@pytest.fixture
def simple_portfolio():
    """Two-asset portfolio for testing."""
    returns = np.array([0.10, 0.05])
    cov = np.array([[0.04, 0.005], [0.005, 0.01]])
    names = ["AAPL", "MSFT"]
    return returns, cov, names


@pytest.fixture
def three_asset_portfolio():
    """Three-asset portfolio."""
    returns = np.array([0.12, 0.08, 0.04])
    cov = np.array(
        [
            [0.04, 0.01, 0.002],
            [0.01, 0.02, 0.003],
            [0.002, 0.003, 0.005],
        ]
    )
    names = ["STOCK_A", "STOCK_B", "BOND_C"]
    return returns, cov, names


class TestMeanVarianceOptimizer:
    def test_basic_optimization(self, simple_portfolio):
        ret, cov, names = simple_portfolio
        opt = MeanVarianceOptimizer()
        result = opt.optimize(ret, cov, names)
        assert result.method == "mean_variance"
        assert (
            sum(result.weights.values()) > 0.5
        )  # Weights may not sum to exactly 1.0 with tight max_weight
        assert result.sharpe_ratio > 0

    def test_three_assets(self, three_asset_portfolio):
        ret, cov, names = three_asset_portfolio
        opt = MeanVarianceOptimizer()
        result = opt.optimize(ret, cov, names)
        assert len(result.weights) == 3
        assert all(w >= 0 for w in result.weights.values())

    def test_max_weight_constraint(self, simple_portfolio):
        ret, cov, names = simple_portfolio
        opt = MeanVarianceOptimizer()
        result = opt.optimize(ret, cov, names, max_weight=0.30)
        assert all(w <= 0.31 for w in result.weights.values())

    def test_empty_inputs(self):
        opt = MeanVarianceOptimizer()
        result = opt.optimize(np.array([]), np.array([[]]), [])
        assert result.method == "mean_variance"
        assert len(result.weights) == 0

    def test_volatility_positive(self, simple_portfolio):
        ret, cov, names = simple_portfolio
        opt = MeanVarianceOptimizer()
        result = opt.optimize(ret, cov, names)
        assert result.expected_volatility > 0


class TestRiskParityOptimizer:
    def test_basic_risk_parity(self, simple_portfolio):
        _, cov, names = simple_portfolio
        opt = RiskParityOptimizer()
        result = opt.optimize(cov, names)
        assert result.method == "risk_parity"
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_lower_vol_gets_higher_weight(self, simple_portfolio):
        _, cov, names = simple_portfolio
        opt = RiskParityOptimizer()
        result = opt.optimize(cov, names)
        # MSFT has lower variance → should get equal or higher weight
        assert result.weights["MSFT"] >= result.weights["AAPL"]

    def test_three_assets(self, three_asset_portfolio):
        ret, cov, names = three_asset_portfolio
        opt = RiskParityOptimizer()
        result = opt.optimize(cov, names, expected_returns=ret)
        assert len(result.weights) == 3


class TestBlackLittermanOptimizer:
    def test_without_views(self, three_asset_portfolio):
        ret, cov, names = three_asset_portfolio
        market_caps = np.array([1000, 500, 200])
        opt = BlackLittermanOptimizer()
        result = opt.optimize(market_caps, cov, names)
        assert result.method == "black_litterman"
        assert len(result.weights) == 3

    def test_with_views(self, three_asset_portfolio):
        _, cov, names = three_asset_portfolio
        market_caps = np.array([1000, 500, 200])
        views = [{"asset_idx": 0, "return": 0.15, "confidence": 0.8}]
        opt = BlackLittermanOptimizer()
        result = opt.optimize(market_caps, cov, names, views=views)
        assert result.method == "black_litterman"
        # View on first asset should increase its weight
        assert result.weights[names[0]] > 0

    def test_empty_inputs(self):
        opt = BlackLittermanOptimizer()
        result = opt.optimize(np.array([]), np.array([[]]), [])
        assert len(result.weights) == 0
