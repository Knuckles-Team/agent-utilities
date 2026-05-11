"""Tests for CONCEPT:KG-2.6 — Profit Attribution Engine."""

import numpy as np
import pytest

from agent_utilities.domains.finance.profit_attribution import (
    BenchmarkComparison,
    PerformanceReport,
    ProfitAttributor,
    compare_to_benchmark,
    compute_performance_report,
)


class TestProfitAttributor:
    def test_perfect_tracking(self):
        """Strategy that perfectly tracks benchmark → alpha=0, beta=1."""
        rng = np.random.default_rng(42)
        bench = rng.normal(0.001, 0.02, 200)
        strat = bench.copy()
        attr = ProfitAttributor()
        result = attr.attribute(strat, bench)
        assert result.beta_coefficient == pytest.approx(1.0, abs=0.01)
        assert result.r_squared > 0.99

    def test_alpha_generation(self):
        """Strategy with consistent alpha → positive alpha component."""
        rng = np.random.default_rng(42)
        bench = rng.normal(0.0, 0.02, 200)
        strat = bench + 0.001  # Add daily alpha
        attr = ProfitAttributor()
        result = attr.attribute(strat, bench)
        assert result.alpha_return > 0

    def test_market_neutral(self):
        """Uncorrelated strategy → beta near 0."""
        rng = np.random.default_rng(42)
        bench = rng.normal(0, 0.02, 200)
        strat = rng.normal(0.001, 0.01, 200)
        attr = ProfitAttributor()
        result = attr.attribute(strat, bench)
        assert abs(result.beta_coefficient) < 0.5

    def test_insufficient_data(self):
        attr = ProfitAttributor()
        result = attr.attribute(np.array([0.01]), np.array([0.02]))
        assert result.total_return == 0.0


class TestPerformanceReport:
    @pytest.fixture
    def positive_returns(self):
        rng = np.random.default_rng(42)
        return rng.normal(0.001, 0.015, 252)

    def test_basic_metrics(self, positive_returns):
        report = compute_performance_report(positive_returns)
        assert isinstance(report, PerformanceReport)
        assert report.n_trades == 252
        assert report.volatility > 0
        assert report.sharpe_ratio != 0

    def test_positive_returns_positive_sharpe(self, positive_returns):
        report = compute_performance_report(positive_returns)
        assert report.sharpe_ratio > 0

    def test_max_drawdown_negative(self, positive_returns):
        report = compute_performance_report(positive_returns)
        assert report.max_drawdown <= 0

    def test_win_rate_reasonable(self, positive_returns):
        report = compute_performance_report(positive_returns)
        assert 0.3 < report.win_rate < 0.8

    def test_sortino_positive_for_uptrend(self, positive_returns):
        report = compute_performance_report(positive_returns)
        assert report.sortino_ratio > 0

    def test_best_worst_day(self, positive_returns):
        report = compute_performance_report(positive_returns)
        assert report.best_day > report.worst_day

    def test_insufficient_data(self):
        report = compute_performance_report(np.array([0.01]))
        assert report.n_trades == 0


class TestBenchmarkComparison:
    def test_outperformance(self):
        rng = np.random.default_rng(42)
        bench = rng.normal(0.0005, 0.02, 200)
        strat = bench + 0.001  # Consistent alpha
        result = compare_to_benchmark(strat, bench)
        assert result.excess_return > 0
        assert result.information_ratio > 0

    def test_underperformance(self):
        rng = np.random.default_rng(42)
        bench = rng.normal(0.001, 0.02, 200)
        strat = bench - 0.002
        result = compare_to_benchmark(strat, bench)
        assert result.excess_return < 0

    def test_tracking_error_positive(self):
        rng = np.random.default_rng(42)
        bench = rng.normal(0, 0.02, 200)
        strat = bench + rng.normal(0, 0.005, 200)
        result = compare_to_benchmark(strat, bench)
        assert result.tracking_error > 0

    def test_high_correlation(self):
        rng = np.random.default_rng(42)
        bench = rng.normal(0, 0.02, 200)
        strat = bench * 1.1  # Levered benchmark
        result = compare_to_benchmark(strat, bench)
        assert result.correlation > 0.95

    def test_insufficient_data(self):
        result = compare_to_benchmark(np.array([0.01]), np.array([0.02]))
        assert result.strategy_return == 0.0
