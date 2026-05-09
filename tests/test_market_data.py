"""Tests for CONCEPT:KG-2.64 — Market Data Abstraction Layer."""

import numpy as np
import pandas as pd
import pytest

from agent_utilities.domains.finance.market_data import (
    DataFetchResult,
    DataRegistry,
    SyntheticProvider,
    normalize_ohlcv,
)


class TestSyntheticProvider:
    def test_basic_fetch(self):
        provider = SyntheticProvider()
        df = provider.fetch("SYNTH", n_bars=100)
        assert len(df) == 100
        assert set(df.columns) == {"Open", "High", "Low", "Close", "Volume"}

    def test_all_values_positive(self):
        provider = SyntheticProvider()
        df = provider.fetch("SYNTH", n_bars=50)
        assert (df > 0).all().all()

    def test_high_above_low(self):
        provider = SyntheticProvider()
        df = provider.fetch("SYNTH", n_bars=100)
        assert (df["High"] >= df["Low"]).all()

    def test_deterministic_with_seed(self):
        p1 = SyntheticProvider()
        p2 = SyntheticProvider()
        df1 = p1.fetch("X", n_bars=50, seed=123)
        df2 = p2.fetch("X", n_bars=50, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_supports_all_symbols(self):
        provider = SyntheticProvider()
        assert provider.supports("AAPL") is True
        assert provider.supports("ANYTHING") is True

    def test_name(self):
        assert SyntheticProvider().name == "synthetic"


class TestDataRegistry:
    def test_synthetic_fallback(self):
        registry = DataRegistry(providers=[SyntheticProvider()])
        result = registry.fetch("SYNTH")
        assert isinstance(result, DataFetchResult)
        assert result.provider == "synthetic"
        assert result.row_count > 0

    def test_fallback_chain(self):
        """A provider that always fails should fall through to the next."""

        class FailingProvider:
            @property
            def name(self):
                return "failing"

            def supports(self, symbol):
                return True

            def fetch(self, symbol, **kwargs):
                raise ConnectionError("Simulated failure")

        registry = DataRegistry(providers=[FailingProvider(), SyntheticProvider()])
        result = registry.fetch("TEST")
        assert result.provider == "synthetic"
        assert any("failing" in w for w in result.warnings)

    def test_all_providers_fail(self):
        class EmptyProvider:
            @property
            def name(self):
                return "empty"

            def supports(self, symbol):
                return True

            def fetch(self, symbol, **kwargs):
                return pd.DataFrame()

        registry = DataRegistry(providers=[EmptyProvider()])
        result = registry.fetch("TEST")
        assert result.provider == "none"
        assert result.row_count == 0

    def test_provider_names(self):
        registry = DataRegistry(providers=[SyntheticProvider()])
        assert "synthetic" in registry.provider_names

    def test_add_provider(self):
        registry = DataRegistry(providers=[])
        registry.add_provider(SyntheticProvider())
        assert len(registry.provider_names) == 1

    def test_fetched_at_populated(self):
        registry = DataRegistry(providers=[SyntheticProvider()])
        result = registry.fetch("SYNTH")
        assert result.fetched_at != ""


class TestNormalizeOHLCV:
    def test_lowercase_columns(self):
        df = pd.DataFrame(
            {
                "open": [100],
                "high": [105],
                "low": [95],
                "close": [102],
                "volume": [1000],
            }
        )
        normalized = normalize_ohlcv(df)
        assert set(normalized.columns) == {"Open", "High", "Low", "Close", "Volume"}

    def test_mixed_case_columns(self):
        df = pd.DataFrame(
            {
                "OPEN": [100],
                "High": [105],
                "low": [95],
                "Close": [102],
                "vol": [1000],
            }
        )
        normalized = normalize_ohlcv(df)
        assert "Open" in normalized.columns
        assert "Volume" in normalized.columns

    def test_adj_close(self):
        df = pd.DataFrame({"adj close": [100]})
        normalized = normalize_ohlcv(df)
        assert "Close" in normalized.columns
