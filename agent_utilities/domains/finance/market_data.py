"""
Unified Market Data Abstraction Layer — CONCEPT:AU-KG.research.research-pipeline-runner

Provides a protocol-based data provider system with auto-fallback chains,
OHLCV normalization, and KG data provenance tracking.

Sources: Qlib Data Server, Vibe-Trading Data Sources
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from agent_utilities.numeric import xp as np

try:
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "Finance extra dependencies missing. Please install agent-utilities[finance]"
    ) from e

logger = logging.getLogger(__name__)


@dataclass
class OHLCVBar:
    """A single OHLCV bar with normalized field names."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class DataFetchResult:
    """Result of a market data fetch including provenance metadata."""

    data: pd.DataFrame
    provider: str
    symbol: str
    interval: str
    fetched_at: str = ""
    row_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.fetched_at:
            self.fetched_at = datetime.now(UTC).isoformat()
        self.row_count = len(self.data) if self.data is not None else 0


@runtime_checkable
class MarketDataProvider(Protocol):
    """Protocol for market data providers — any class implementing this interface is a valid provider."""

    @property
    def name(self) -> str:
        return "protocol"

    def fetch(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        return pd.DataFrame()

    def supports(self, symbol: str) -> bool:
        return False


class YFinanceProvider:
    """
    yfinance-based market data provider for global equities, ETFs, indices, and crypto.
    """

    @property
    def name(self) -> str:
        return "yfinance"

    def supports(self, symbol: str) -> bool:
        """yfinance supports most ticker formats."""
        return bool(symbol and len(symbol) <= 20)

    def fetch(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from yfinance.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'BTC-USD').
            period: Data period ('1mo', '3mo', '1y', '5y', 'max').
            interval: Bar interval ('1d', '1h', '5m').
        """
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is required. Install agent-utilities[finance]"
            ) from e

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {symbol} from yfinance")
            return pd.DataFrame()

        # Normalize columns
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        return df


class AKShareProvider:
    """
    AKShare-based market data provider for Chinese equities (A-shares) and index data.
    """

    @property
    def name(self) -> str:
        return "akshare"

    def supports(self, symbol: str) -> bool:
        """AKShare supports 6-digit Chinese tickers."""
        return bool(symbol and symbol.isdigit() and len(symbol) == 6)

    def fetch(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        try:
            import akshare as ak
        except ImportError as e:
            raise ImportError(
                "akshare is required for Chinese markets. Install agent-utilities[finance]"
            ) from e

        try:
            # Default to A-share daily
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
            if df.empty:
                return pd.DataFrame()

            # Rename columns from Chinese to standard OHLCV
            df = df.rename(
                columns={
                    "日期": "Date",
                    "开盘": "Open",
                    "收盘": "Close",
                    "最高": "High",
                    "最低": "Low",
                    "成交量": "Volume",
                }
            )
            df.set_index("Date", inplace=True)
            df.index = pd.to_datetime(df.index)
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            logger.warning(f"AKShare fetch failed for {symbol}: {e}")
            return pd.DataFrame()


class CryptoProvider:
    """
    CCXT-based provider for global cryptocurrency exchange data (Binance, Coinbase, etc.).
    """

    @property
    def name(self) -> str:
        return "crypto"

    def supports(self, symbol: str) -> bool:
        """Crypto usually formatted as BASE/QUOTE (e.g. BTC/USDT)."""
        return "/" in symbol

    def fetch(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        try:
            import ccxt
        except ImportError as e:
            raise ImportError(
                "ccxt is required for crypto markets. Install agent-utilities[finance]"
            ) from e

        # Map '1d' to CCXT timeframe format
        tf_map = {"1d": "1d", "1h": "1h", "5m": "5m"}
        tf = tf_map.get(interval, "1d")

        try:
            exchange = ccxt.binance({"enableRateLimit": True})
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=1000)

            if not ohlcv:
                return pd.DataFrame()

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.warning(f"CCXT fetch failed for {symbol}: {e}")
            return pd.DataFrame()


class PredictionMarketProvider:
    """
    Polymarket/Kalshi provider for prediction market probability data.
    Maps prediction odds to 'price' and liquidity to 'volume'.
    """

    @property
    def name(self) -> str:
        return "prediction_market"

    def supports(self, symbol: str) -> bool:
        """Prediction markets use specific question IDs or prefixes."""
        return symbol.startswith("POLY:") or symbol.startswith("KALSHI:")

    def fetch(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        # Provider API mapping for prediction market (requires Py-Polymarket or Kalshi SDK)
        # Returns synthetic probability data for now to satisfy the abstraction
        logger.info(f"Using synthetic prediction market data for {symbol}")
        provider = SyntheticProvider()
        df = provider.fetch(
            symbol=symbol,
            period=period,
            interval=interval,
            initial_price=0.5,
            volatility=0.05,
        )
        # Bound between 0 and 1 (0% to 100% probability)
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].clip(0.01, 0.99)
        return df


class SyntheticProvider:
    """
    Synthetic data provider for testing — generates realistic OHLCV data
    using geometric Brownian motion.
    """

    @property
    def name(self) -> str:
        return "synthetic"

    def supports(self, symbol: str) -> bool:
        return True

    def fetch(
        self,
        symbol: str = "SYNTH",
        period: str = "1y",
        interval: str = "1d",
        n_bars: int = 252,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data using GBM."""
        rng = np.random.default_rng(seed)

        dates = pd.bdate_range(end=datetime.now(), periods=n_bars)
        returns = rng.normal(0.0005, volatility, n_bars)

        close = initial_price * np.exp(np.cumsum(returns))
        high = close * (1 + rng.uniform(0, 0.02, n_bars))
        low = close * (1 - rng.uniform(0, 0.02, n_bars))
        open_price = close * (1 + rng.normal(0, 0.005, n_bars))
        volume = rng.integers(100_000, 10_000_000, n_bars).astype(float)

        df = pd.DataFrame(
            {
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=dates,
        )

        return df


class DataRegistry:
    """
    Auto-fallback data registry — tries providers in priority order
    and returns the first successful result.
    """

    def __init__(self, providers: list | None = None):
        if providers is not None:
            self._providers = providers
        else:
            # Default chain: yfinance → akshare → crypto → synthetic
            self._providers = [
                YFinanceProvider(),
                AKShareProvider(),
                CryptoProvider(),
                PredictionMarketProvider(),
                SyntheticProvider(),
            ]

    def add_provider(self, provider, priority: int | None = None):
        """Add a data provider. Lower priority index = higher priority."""
        if priority is not None:
            self._providers.insert(priority, provider)
        else:
            self._providers.append(provider)

    def fetch(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> DataFetchResult:
        """
        Fetch data using the fallback chain. Returns the first successful result.
        """
        warnings = []

        for provider in self._providers:
            if not provider.supports(symbol):
                warnings.append(f"{provider.name}: does not support {symbol}")
                continue

            try:
                df = provider.fetch(symbol, period=period, interval=interval)
                if df is not None and not df.empty:
                    logger.info(
                        f"Fetched {len(df)} bars for {symbol} from {provider.name}"
                    )
                    return DataFetchResult(
                        data=df,
                        provider=provider.name,
                        symbol=symbol,
                        interval=interval,
                        warnings=warnings,
                    )
                else:
                    warnings.append(f"{provider.name}: returned empty data")
            except Exception as e:
                warnings.append(f"{provider.name}: {e}")
                logger.warning(f"Provider {provider.name} failed for {symbol}: {e}")

        logger.error(f"All providers failed for {symbol}")
        return DataFetchResult(
            data=pd.DataFrame(),
            provider="none",
            symbol=symbol,
            interval=interval,
            warnings=warnings,
        )

    @property
    def provider_names(self) -> list[str]:
        """List registered provider names in priority order."""
        return [p.name for p in self._providers]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an OHLCV DataFrame to standard column names.
    Handles common variations (lowercase, uppercase, mixed case).
    """
    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if lower in ("open", "o"):
            col_map[col] = "Open"
        elif lower in ("high", "h"):
            col_map[col] = "High"
        elif lower in ("low", "l"):
            col_map[col] = "Low"
        elif lower in ("close", "c", "adj close", "adj_close"):
            col_map[col] = "Close"
        elif lower in ("volume", "vol", "v"):
            col_map[col] = "Volume"

    return df.rename(columns=col_map)
