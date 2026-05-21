"""Tests for CONCEPT:KG-2.6 — Signal Fusion and Alpha Combination Engine."""

import numpy as np
import pytest

from agent_utilities.domains.finance.signal_fusion import (
    AlphaCombinationEngine,
    BayesianSignalFusion,
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

    def test_not_enough_periods(self):
        engine = AlphaCombinationEngine(lookback_d=10)
        # N=3 signals, M=5 periods (needs at least 12)
        returns = np.random.randn(3, 5)

        with pytest.raises(ValueError):
            engine.compute_weights(returns)
