"""Tests for CONCEPT:KG-2.6 — Signal Fusion and Alpha Combination Engine."""

import numpy as np

from agent_utilities.domains.finance.signal_fusion import (
    AlphaCombinationEngine,
    BayesianSignalFusion,
    LaplaceEnsembleFusion,
)


class TestBayesianSignalFusion:
    def test_basic_fusion(self):
        fusion = BayesianSignalFusion(prior=0.5)
        fusion.register_source("MACD", weight=1.0, accuracy=0.6)

        # MACD signals UP (1)
        # Prior is 0.5. Likelihood up=0.6, down=0.4
        # Posterior = (0.6 * 0.5) / (0.6 * 0.5 + 0.4 * 0.5) = 0.3 / 0.5 = 0.6
        post = fusion.fuse({"MACD": 1})
        assert post > 0.5

    def test_conflicting_signals(self):
        fusion = BayesianSignalFusion(prior=0.5)
        fusion.register_source("MACD", weight=1.0, accuracy=0.6)
        fusion.register_source("RSI", weight=1.0, accuracy=0.7)

        # MACD says UP, RSI says DOWN
        post = fusion.fuse({"MACD": 1, "RSI": -1})
        # RSI has higher accuracy, so we expect the probability of UP to be less than 0.5
        assert post < 0.5


class TestAlphaCombinationEngine:
    def test_basic_computation(self):
        engine = AlphaCombinationEngine(lookback_d=10)
        # N=3 signals, M=20 periods
        returns = np.random.randn(3, 20)

        weights = engine.compute_weights(returns)

        assert len(weights) == 3
        assert np.isclose(np.abs(weights).sum(), 1.0)


class TestLaplaceEnsembleFusion:
    def test_basic_smoothing(self):
        # 16 / 31 models say UP
        # Expected: (16 + 1) / (31 + 2) = 17 / 33 ~= 0.515
        prob = LaplaceEnsembleFusion.compute_probability(16, 31)
        assert np.isclose(prob, 17 / 33)

    def test_zero_total_members(self):
        prob = LaplaceEnsembleFusion.compute_probability(0, 0)
        assert prob == 0.5

    def test_extreme_smoothing(self):
        # 31/31 say UP
        # Expected: 32 / 33 ~= 0.9697 (not 1.0)
        prob = LaplaceEnsembleFusion.compute_probability(31, 31)
        assert prob < 1.0
        assert np.isclose(prob, 32 / 33)
