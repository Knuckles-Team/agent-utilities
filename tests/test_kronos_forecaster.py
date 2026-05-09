"""Tests for CONCEPT:KG-2.70 — Kronos Foundation Model Forecaster."""

import numpy as np
import pytest

from agent_utilities.domains.finance.kronos_forecaster import (
    CandleType,
    ForecastResult,
    KLineToken,
    KLineTokenizer,
    KronosForecaster,
    KronosPredictor,
)


@pytest.fixture
def sample_ohlcv():
    rng = np.random.default_rng(42)
    n = 200
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    opens = close * (1 + rng.normal(0, 0.005, n))
    highs = np.maximum(opens, close) * (1 + rng.uniform(0, 0.01, n))
    lows = np.minimum(opens, close) * (1 - rng.uniform(0, 0.01, n))
    volumes = rng.integers(100_000, 10_000_000, n).astype(float)
    return opens, highs, lows, close, volumes


class TestKLineToken:
    def test_vocab_size(self):
        assert KLineToken.vocab_size() == 1200

    def test_to_int_range(self):
        token = KLineToken(
            body_direction=1,
            body_size_bucket=4,
            upper_wick_bucket=3,
            lower_wick_bucket=3,
            volume_bucket=4,
        )
        assert 0 <= token.to_int() < KLineToken.vocab_size()

    def test_deterministic(self):
        t1 = KLineToken(1, 2, 1, 1, 3)
        t2 = KLineToken(1, 2, 1, 1, 3)
        assert t1.to_int() == t2.to_int()


class TestKLineTokenizer:
    def test_fit_and_tokenize(self, sample_ohlcv):
        opens, highs, lows, closes, volumes = sample_ohlcv
        tokenizer = KLineTokenizer()
        tokenizer.fit(opens, highs, lows, closes, volumes)
        tokens = tokenizer.tokenize_series(opens, highs, lows, closes, volumes)
        assert len(tokens) == len(opens)
        assert all(0 <= t.token_id < KLineToken.vocab_size() for t in tokens)

    def test_tokenize_before_fit_raises(self):
        tokenizer = KLineTokenizer()
        with pytest.raises(RuntimeError):
            tokenizer.tokenize_bar(100, 105, 95, 102, 1000)

    def test_candle_classification(self, sample_ohlcv):
        opens, highs, lows, closes, volumes = sample_ohlcv
        tokenizer = KLineTokenizer()
        tokenizer.fit(opens, highs, lows, closes, volumes)
        tokens = tokenizer.tokenize_series(opens, highs, lows, closes, volumes)
        types = {t.candle_type for t in tokens}
        assert len(types) > 1  # Multiple candle types should be detected

    def test_body_direction(self, sample_ohlcv):
        opens, highs, lows, closes, volumes = sample_ohlcv
        tokenizer = KLineTokenizer()
        tokenizer.fit(opens, highs, lows, closes, volumes)
        tokens = tokenizer.tokenize_series(opens, highs, lows, closes, volumes)
        directions = {t.body_direction for t in tokens}
        assert -1 in directions or 1 in directions


class TestKronosPredictor:
    def test_fit_and_predict(self):
        token_ids = [10, 20, 10, 20, 10, 20, 10, 20]
        predictor = KronosPredictor()
        predictor.fit(token_ids)
        next_tok, conf = predictor.predict_next([10])
        assert next_tok == 20
        assert conf > 0.5

    def test_predict_sequence(self):
        token_ids = list(range(50)) * 5  # Repeating pattern
        predictor = KronosPredictor()
        predictor.fit(token_ids)
        result = predictor.predict_sequence(token_ids[:10], horizon=5)
        assert isinstance(result, ForecastResult)
        assert result.horizon == 5

    def test_unfitted_returns_zero(self):
        predictor = KronosPredictor()
        tok, conf = predictor.predict_next([1, 2, 3])
        assert tok == 0 and conf == 0.0


class TestKronosForecaster:
    def test_full_pipeline(self, sample_ohlcv):
        opens, highs, lows, closes, volumes = sample_ohlcv
        forecaster = KronosForecaster()
        forecaster.fit(opens, highs, lows, closes, volumes)
        result = forecaster.forecast(horizon=5)
        assert result.horizon == 5
        assert result.predicted_direction in (-1, 0, 1)
        assert 0.0 <= result.confidence <= 1.0

    def test_vocab_size(self):
        f = KronosForecaster()
        assert f.vocab_size == 1200

    def test_empty_forecast(self):
        f = KronosForecaster()
        result = f.forecast()
        assert result.horizon == 0
