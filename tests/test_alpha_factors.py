"""Tests for CONCEPT:KG-2.6 — Alpha Factor Library."""

import numpy as np
import pandas as pd
import pytest

from agent_utilities.domains.finance.alpha_factors import (
    AlphaFactorLibrary,
    FACTOR_REGISTRY,
    compute_factor_ic,
    compute_factor_ir,
    momentum_1d,
    momentum_5d,
    volatility_ratio,
    volume_zscore,
    rsi,
    macd_signal,
    bollinger_position,
    rank_factors,
)


@pytest.fixture
def sample_ohlcv():
    """Generate realistic OHLCV data for testing."""
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.bdate_range(end="2024-12-31", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.005, n)),
            "High": close * (1 + rng.uniform(0, 0.02, n)),
            "Low": close * (1 - rng.uniform(0, 0.02, n)),
            "Close": close,
            "Volume": rng.integers(100_000, 10_000_000, n).astype(float),
        },
        index=dates,
    )


class TestIndividualFactors:
    def test_momentum_1d(self, sample_ohlcv):
        result = momentum_1d(sample_ohlcv["Close"])
        assert len(result) == len(sample_ohlcv)
        assert result.iloc[0] != result.iloc[0]  # First value is NaN

    def test_momentum_5d(self, sample_ohlcv):
        result = momentum_5d(sample_ohlcv["Close"])
        assert result.dropna().shape[0] == len(sample_ohlcv) - 5

    def test_volatility_ratio(self, sample_ohlcv):
        result = volatility_ratio(sample_ohlcv["Close"])
        clean = result.dropna()
        assert len(clean) > 0
        assert all(clean > 0)

    def test_volume_zscore(self, sample_ohlcv):
        result = volume_zscore(sample_ohlcv["Volume"])
        clean = result.dropna()
        assert abs(clean.mean()) < 1.0  # Should be roughly centered

    def test_rsi_range(self, sample_ohlcv):
        result = rsi(sample_ohlcv["Close"])
        clean = result.dropna()
        assert all(clean >= 0)
        assert all(clean <= 100)

    def test_macd_signal(self, sample_ohlcv):
        result = macd_signal(sample_ohlcv["Close"])
        clean = result.dropna()
        assert len(clean) > 0

    def test_bollinger_position(self, sample_ohlcv):
        result = bollinger_position(sample_ohlcv["Close"])
        clean = result.dropna()
        # Most values should be within [-1, 1] but extremes can exceed
        assert abs(clean.mean()) < 1.0


class TestAlphaFactorLibrary:
    def test_compute_all(self, sample_ohlcv):
        library = AlphaFactorLibrary()
        result = library.compute_all(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert len(result.columns) >= 15  # Most factors should compute

    def test_compute_subset(self, sample_ohlcv):
        library = AlphaFactorLibrary(factor_names=["momentum_1d", "rsi"])
        result = library.compute_all(sample_ohlcv)
        assert set(result.columns) == {"momentum_1d", "rsi"}

    def test_unknown_factor_raises(self):
        with pytest.raises(ValueError):
            AlphaFactorLibrary(factor_names=["nonexistent_factor"])

    def test_available_factors(self):
        library = AlphaFactorLibrary()
        assert len(library.available_factors) == len(FACTOR_REGISTRY)

    def test_case_insensitive_columns(self, sample_ohlcv):
        df = sample_ohlcv.rename(columns=str.lower)
        library = AlphaFactorLibrary(factor_names=["momentum_1d"])
        result = library.compute_all(df)
        assert "momentum_1d" in result.columns

    def test_missing_columns_skips_factor(self):
        df = pd.DataFrame({"Close": [100, 101, 102] * 30})
        library = AlphaFactorLibrary(factor_names=["momentum_1d", "volume_zscore"])
        result = library.compute_all(df)
        # volume_zscore should be skipped (no Volume column)
        assert "volume_zscore" not in result.columns


class TestICIRAnalysis:
    def test_compute_factor_ic_positive(self):
        rng = np.random.default_rng(42)
        n = 100
        factor = pd.Series(rng.standard_normal(n))
        returns = factor * 0.5 + pd.Series(rng.standard_normal(n) * 0.1)
        ic = compute_factor_ic(factor, returns)
        assert ic > 0.3  # Strong positive IC expected

    def test_compute_factor_ic_small_sample(self):
        factor = pd.Series([1.0, 2.0])
        returns = pd.Series([0.01, 0.02])
        ic = compute_factor_ic(factor, returns)
        assert ic == 0.0  # Too few observations

    def test_compute_factor_ir(self):
        ic_series = pd.Series([0.05, 0.03, 0.04, 0.06, 0.02, 0.05, 0.04])
        ir = compute_factor_ir(ic_series)
        assert ir > 0  # Consistently positive IC → positive IR

    def test_compute_factor_ir_insufficient_data(self):
        ic_series = pd.Series([0.05])
        ir = compute_factor_ir(ic_series)
        assert ir == 0.0

    def test_rank_factors(self):
        rng = np.random.default_rng(42)
        n = 200
        returns = pd.Series(rng.standard_normal(n) * 0.01)
        factors = {
            "good_factor": pd.Series(returns * 5 + rng.standard_normal(n) * 0.01),
            "bad_factor": pd.Series(rng.standard_normal(n)),
        }
        ranking = rank_factors(factors, returns)
        assert ranking.iloc[0]["factor"] == "good_factor"  # Should rank first
