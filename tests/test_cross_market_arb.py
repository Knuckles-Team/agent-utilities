"""Tests for CONCEPT:KG-2.6 — Cross Market Arbitrage."""

import numpy as np
import pytest

from agent_utilities.domains.finance.cross_market_arb import (
    CointegrationAnalyzer,
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
