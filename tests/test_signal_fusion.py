"""Tests for CONCEPT:KG-2.6 — Signal Fusion and Alpha Combination Engine."""

from agent_utilities.domains.finance.signal_fusion import (
    AlphaCombinationEngine,
    BayesianSignalFusion,
    LaplaceEnsembleFusion,
)
from agent_utilities.numeric import xp as np


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

    def test_seed_from_kg_filters_and_weights(self):
        """CONCEPT:EE-033 — seed fusion from stored MicrostructureSignal priors,
        dropping overfit (pbo>0.5) and edgeless (sharpe<=0) signals; weight =
        directional_accuracy * standalone_sharpe."""
        from agent_utilities.models.domains.finance import MicrostructureSignalNode

        sigs = [
            MicrostructureSignalNode(
                id="sig:ofi",
                name="ofi",
                directional_accuracy=0.57,
                standalone_sharpe=1.8,
                pbo=0.2,
            ),
            MicrostructureSignalNode(
                id="sig:overfit",
                name="overfit",
                directional_accuracy=0.6,
                standalone_sharpe=2.0,
                pbo=0.7,  # dropped: overfit
            ),
            MicrostructureSignalNode(
                id="sig:noedge",
                name="noedge",
                directional_accuracy=0.5,
                standalone_sharpe=0.0,
                pbo=0.1,  # dropped: no edge
            ),
        ]
        fusion = BayesianSignalFusion()
        registered = fusion.seed_from_kg(sigs)
        assert registered == 1
        assert set(fusion.sources) == {"ofi"}
        assert np.isclose(fusion.sources["ofi"].weight, 0.57 * 1.8)
        # Bullish surviving signal lifts the posterior above the prior.
        assert fusion.fuse({"ofi": 1}) > 0.5

    def test_seed_from_kg_accepts_plain_mappings(self):
        fusion = BayesianSignalFusion()
        n = fusion.seed_from_kg(
            [
                {
                    "name": "queue",
                    "directional_accuracy": 0.55,
                    "standalone_sharpe": 1.0,
                    "pbo": 0.1,
                }
            ]
        )
        assert n == 1 and "queue" in fusion.sources


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
