"""Tests for CONCEPT:AU-KG.research.research-pipeline-runner — Market Data Abstraction Layer."""

import pandas as pd
import pytest

# The compiled epistemic_graph.numeric kernel must be built for these tests; skip the whole module cleanly when it isn't, rather than erroring out collection (CONCEPT:AU-KG.compute.numeric-kernel).
pytest.importorskip("epistemic_graph.numeric")

from agent_utilities.domains.finance.market_data import (
    DataFetchResult,
    DataRegistry,
    normalize_ohlcv,
)


class _FakeOHLCVProvider:
    """A deterministic, in-memory provider for exercising ``DataRegistry``'s
    fallback-chain mechanics without a real (or fabricated) market feed. This
    test double stays under ``tests/`` rather than in the production
    market-data module."""

    def __init__(self, name: str = "fake", n_bars: int = 10):
        self._name = name
        self._n_bars = n_bars

    @property
    def name(self) -> str:
        return self._name

    def supports(self, symbol: str) -> bool:
        return True

    def fetch(self, symbol, **kwargs) -> pd.DataFrame:
        n = kwargs.get("n_bars", self._n_bars)
        return pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [101.0] * n,
                "Low": [99.0] * n,
                "Close": [100.5] * n,
                "Volume": [1000.0] * n,
            }
        )


class TestDataRegistry:
    def test_fallback_provider_result(self):
        registry = DataRegistry(providers=[_FakeOHLCVProvider()])
        result = registry.fetch("TEST")
        assert isinstance(result, DataFetchResult)
        assert result.provider == "fake"
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

        registry = DataRegistry(providers=[FailingProvider(), _FakeOHLCVProvider()])
        result = registry.fetch("TEST")
        assert result.provider == "fake"
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
        registry = DataRegistry(providers=[_FakeOHLCVProvider()])
        assert "fake" in registry.provider_names

    def test_add_provider(self):
        registry = DataRegistry(providers=[])
        registry.add_provider(_FakeOHLCVProvider())
        assert len(registry.provider_names) == 1

    def test_fetched_at_populated(self):
        registry = DataRegistry(providers=[_FakeOHLCVProvider()])
        result = registry.fetch("TEST")
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
