"""Comprehensive tests for Markov Chain Regime Detection.

CONCEPT:KG-2.6 — Markov Regime Detection Tests

Tests cover:
- MarketRegimeDetector state labeling
- MarkovRegimeModel end-to-end fit and forecast
- Chapman-Kolmogorov multi-step transitions
- Signal generation
- Walk-forward backtest structure
- predict_next_states (PreemptiveCacheEngine contract)
- Bug regression for double-negation fix
"""

from __future__ import annotations

import numpy as np
import pytest

from agent_utilities.knowledge_graph.core.formal_reasoning_core import (
    MarkovTransitionModel,
)
from agent_utilities.knowledge_graph.core.markov_regime import (
    ASSET_CLASS_DEFAULTS,
    AssetClass,
    BacktestResult,
    MarketRegimeDetector,
    MarkovRegimeModel,
    RegimeState,
    RegimeThresholds,
)


# --- Fixtures ---


@pytest.fixture
def synthetic_returns() -> np.ndarray:
    """Synthetic returns with clear regime structure.

    - Days 0-49:  Bull regime (positive drift)
    - Days 50-99: Bear regime (negative drift)
    - Days 100-149: Sideways (near-zero)
    """
    rng = np.random.default_rng(42)
    bull = rng.normal(0.005, 0.01, 50)   # ~0.5% daily, mild vol
    bear = rng.normal(-0.005, 0.01, 50)  # ~-0.5% daily
    sideways = rng.normal(0.0, 0.005, 50)  # ~0% daily, low vol
    return np.concatenate([bull, bear, sideways])


@pytest.fixture
def long_returns() -> np.ndarray:
    """Longer returns series for backtest testing (500 days)."""
    rng = np.random.default_rng(123)
    returns = np.zeros(500)
    # Regime switching every ~100 days
    for i in range(500):
        cycle = (i // 100) % 3
        if cycle == 0:
            returns[i] = rng.normal(0.003, 0.01)
        elif cycle == 1:
            returns[i] = rng.normal(-0.003, 0.01)
        else:
            returns[i] = rng.normal(0.0, 0.005)
    return returns


# --- MarketRegimeDetector Tests ---


class TestMarketRegimeDetector:
    def test_detect_rolling_sum(self, synthetic_returns: np.ndarray):
        """Test that rolling_sum detection produces valid labels."""
        detector = MarketRegimeDetector(
            asset_class=AssetClass.EQUITIES,
            window=10,
        )
        states = detector.detect(synthetic_returns)

        assert len(states) == len(synthetic_returns)
        valid_states = {RegimeState.BULL, RegimeState.BEAR, RegimeState.SIDEWAYS}
        for s in states:
            assert s in valid_states

    def test_detect_compounding(self, synthetic_returns: np.ndarray):
        """Test that compounding detection also works."""
        detector = MarketRegimeDetector(
            asset_class=AssetClass.EQUITIES,
            window=10,
            method="compounding",
        )
        states = detector.detect(synthetic_returns)
        assert len(states) == len(synthetic_returns)

    def test_detect_short_series(self):
        """Series shorter than window → all SIDEWAYS."""
        detector = MarketRegimeDetector(window=20)
        states = detector.detect(np.random.randn(10))
        assert all(s == RegimeState.SIDEWAYS for s in states)

    def test_detect_with_custom_thresholds(self, synthetic_returns: np.ndarray):
        """Custom thresholds should override defaults."""
        detector = MarketRegimeDetector(
            asset_class=AssetClass.EQUITIES,
            bull_threshold=0.001,
            bear_threshold=-0.001,
            window=5,
        )
        states = detector.detect(synthetic_returns)
        # With very tight thresholds, we should see more bull/bear labels
        bull_count = sum(1 for s in states if s == RegimeState.BULL)
        bear_count = sum(1 for s in states if s == RegimeState.BEAR)
        assert bull_count > 0
        assert bear_count > 0


class TestAssetClassDefaults:
    def test_all_asset_classes_have_defaults(self):
        """Every AssetClass enum value must have default thresholds."""
        for ac in AssetClass:
            assert ac in ASSET_CLASS_DEFAULTS
            thresholds = ASSET_CLASS_DEFAULTS[ac]
            assert isinstance(thresholds, RegimeThresholds)
            assert thresholds.bull_threshold > 0
            assert thresholds.bear_threshold < 0
            assert thresholds.window > 0

    def test_crypto_wider_thresholds(self):
        """Crypto thresholds should be wider than equities."""
        equities = ASSET_CLASS_DEFAULTS[AssetClass.EQUITIES]
        crypto = ASSET_CLASS_DEFAULTS[AssetClass.CRYPTO]
        assert crypto.bull_threshold > equities.bull_threshold
        assert abs(crypto.bear_threshold) > abs(equities.bear_threshold)


# --- MarkovRegimeModel Tests ---


class TestMarkovRegimeModel:
    def test_fit(self, synthetic_returns: np.ndarray):
        """Model should fit without error and set _fitted=True."""
        model = MarkovRegimeModel(
            asset_class=AssetClass.EQUITIES, window=10
        )
        result = model.fit(synthetic_returns)
        assert result is model  # Method chaining
        assert model.is_fitted
        assert model.regime_states is not None
        assert len(model.regime_states) == len(synthetic_returns)

    def test_unfitted_raises(self):
        """Calling forecast/signal on unfitted model should raise."""
        model = MarkovRegimeModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.forecast("bull", 5)

    def test_forecast(self, synthetic_returns: np.ndarray):
        """Forecast should return probabilities that sum to ~1.0."""
        model = MarkovRegimeModel(window=10)
        model.fit(synthetic_returns)
        forecast = model.forecast(RegimeState.BULL, n_steps=5)

        assert len(forecast) > 0
        total = sum(forecast.values())
        assert abs(total - 1.0) < 0.01  # Probabilities sum to 1

    def test_stationary_distribution(self, synthetic_returns: np.ndarray):
        """Stationary distribution should sum to 1.0."""
        model = MarkovRegimeModel(window=10)
        model.fit(synthetic_returns)
        stat_dist = model.stationary_distribution()

        assert len(stat_dist) > 0
        total = sum(stat_dist.values())
        assert abs(total - 1.0) < 0.01

    def test_generate_signal(self, synthetic_returns: np.ndarray):
        """Signal should be in [-1, 1] range."""
        model = MarkovRegimeModel(window=10)
        model.fit(synthetic_returns)
        sig = model.generate_signal(RegimeState.BULL)

        assert "signal" in sig
        assert "bull_prob" in sig
        assert "bear_prob" in sig
        assert "sideways_prob" in sig
        assert -1.0 <= sig["signal"] <= 1.0
        assert abs(sig["bull_prob"] + sig["bear_prob"] + sig["sideways_prob"] - 1.0) < 0.01

    def test_transition_matrix_rows_sum_to_one(self, synthetic_returns: np.ndarray):
        """Transition matrix rows must sum to 1.0."""
        model = MarkovRegimeModel(window=10)
        model.fit(synthetic_returns)
        trans = model.get_transition_matrix_dict()

        for src, row in trans.items():
            row_sum = sum(row.values())
            assert abs(row_sum - 1.0) < 0.001, f"Row {src} sums to {row_sum}"

    def test_to_kg_properties(self, synthetic_returns: np.ndarray):
        """KG serialization should contain all required fields."""
        model = MarkovRegimeModel(window=10)
        model.fit(synthetic_returns)
        props = model.to_kg_properties()

        assert "asset_class" in props
        assert "transition_matrix" in props
        assert "stationary_distribution" in props
        assert "n_states" in props
        assert props["n_states"] > 0


class TestWalkForwardBacktest:
    def test_backtest_structure(self, long_returns: np.ndarray):
        """Walk-forward backtest should return correct structure."""
        model = MarkovRegimeModel(window=10)
        model.fit(long_returns[:50])  # Initial fit required

        result = model.walk_forward_backtest(long_returns, lookback=100)

        assert isinstance(result, BacktestResult)
        assert len(result.signals) == len(long_returns)
        assert len(result.returns) == len(long_returns)
        assert isinstance(result.cumulative_return, float)
        assert result.n_regime_changes >= 0

    def test_no_lookahead_bias(self, long_returns: np.ndarray):
        """Signals before lookback period should be zero (no lookahead)."""
        model = MarkovRegimeModel(window=10)
        model.fit(long_returns[:50])

        result = model.walk_forward_backtest(long_returns, lookback=200)

        # All signals before the lookback window should be zero
        assert all(result.signals[:200] == 0.0)


# --- Core MarkovTransitionModel Enhancement Tests ---


class TestMarkovTransitionModelEnhancements:
    def test_predict_next_states(self):
        """predict_next_states should return sorted (state, prob) tuples."""
        model = MarkovTransitionModel()
        model.ingest_trace(["A", "B", "A", "B", "A", "C"])

        preds = model.predict_next_states("A", k=2)
        assert len(preds) <= 2
        # Probabilities should be descending
        if len(preds) >= 2:
            assert preds[0][1] >= preds[1][1]

    def test_predict_next_states_unknown_state(self):
        """Unknown state should return empty list."""
        model = MarkovTransitionModel()
        model.ingest_trace(["A", "B", "C"])
        assert model.predict_next_states("UNKNOWN") == []

    def test_multi_step_transition(self):
        """Chapman-Kolmogorov: P^n should have rows summing to 1."""
        model = MarkovTransitionModel()
        model.ingest_trace(["A", "B", "A", "B", "C", "A", "B"])

        p_n = model.multi_step_transition(5)
        assert p_n is not None
        for row in p_n:
            assert abs(sum(row) - 1.0) < 0.001

    def test_multi_step_transition_converges_to_stationary(self):
        """P^n should converge to stationary distribution as n → ∞."""
        model = MarkovTransitionModel()
        trace = ["A", "B", "C", "A", "B", "A", "C", "B", "A", "B", "C"] * 10
        model.ingest_trace(trace)

        stat_dist = model.stationary_distribution()
        p_large = model.multi_step_transition(100)
        assert p_large is not None

        # All rows of P^100 should be approximately the stationary distribution
        for i in range(len(model.states)):
            for j, state in enumerate(model.states):
                assert abs(p_large[i][j] - stat_dist.get(state, 0)) < 0.01

    def test_forecast_from_state(self):
        """forecast_from_state should return valid probability dict."""
        model = MarkovTransitionModel()
        model.ingest_trace(["X", "Y", "X", "Y", "Z", "X"])

        forecast = model.forecast_from_state("X", 3)
        assert len(forecast) > 0
        assert abs(sum(forecast.values()) - 1.0) < 0.01

    def test_forecast_from_unknown_state(self):
        """Unknown starting state should return empty dict."""
        model = MarkovTransitionModel()
        model.ingest_trace(["A", "B"])
        assert model.forecast_from_state("UNKNOWN", 3) == {}

    def test_multi_step_transition_invalid_steps(self):
        """n_steps < 1 should return None."""
        model = MarkovTransitionModel()
        model.ingest_trace(["A", "B"])
        assert model.multi_step_transition(0) is None
        assert model.multi_step_transition(-1) is None


class TestBugFixRegression:
    """Regression test for the double-negation bug in get_transition_probability."""

    def test_get_transition_probability_works(self):
        """After the fix, get_transition_probability should return nonzero values."""
        model = MarkovTransitionModel()
        model.ingest_trace(["A", "B", "A", "B", "A"])

        # A→B should have high probability
        prob = model.get_transition_probability("A", "B")
        assert prob > 0.0, "Bug regression: double-negation returns 0.0 always"

    def test_get_transition_probability_unknown_returns_zero(self):
        """Unknown states should still return 0.0."""
        model = MarkovTransitionModel()
        model.ingest_trace(["A", "B"])
        assert model.get_transition_probability("A", "UNKNOWN") == 0.0
        assert model.get_transition_probability("UNKNOWN", "A") == 0.0
