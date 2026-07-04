"""
KG-Native Alpha Factor Library — CONCEPT:KG-2.6

Provides a comprehensive library of battle-tested alpha factors with
Information Coefficient (IC) / Information Ratio (IR) analysis.

Sources: Qlib Alpha158, Vibe-Trading Factor Research
"""

import logging
from typing import Any

from agent_utilities.numeric import xp as np

try:
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "Finance extra dependencies missing. Please install agent-utilities[finance]"
    ) from e

logger = logging.getLogger(__name__)


# ── Factor Computation Functions ────────────────────────────────────


def momentum_1d(close: pd.Series) -> pd.Series:
    """1-day log return."""
    return np.log(close / close.shift(1))


def momentum_5d(close: pd.Series) -> pd.Series:
    """5-day log return."""
    return np.log(close / close.shift(5))


def momentum_20d(close: pd.Series) -> pd.Series:
    """20-day log return."""
    return np.log(close / close.shift(20))


def momentum_60d(close: pd.Series) -> pd.Series:
    """60-day log return (quarterly momentum)."""
    return np.log(close / close.shift(60))


def volatility_5d(close: pd.Series) -> pd.Series:
    """5-day rolling realized volatility."""
    return close.pct_change().rolling(5).std()


def volatility_20d(close: pd.Series) -> pd.Series:
    """20-day rolling realized volatility."""
    return close.pct_change().rolling(20).std()


def volatility_ratio(close: pd.Series) -> pd.Series:
    """Short/long volatility ratio — measures volatility regime."""
    vol_5 = close.pct_change().rolling(5).std()
    vol_20 = close.pct_change().rolling(20).std()
    return vol_5 / vol_20


def momentum_normalized(close: pd.Series) -> pd.Series:
    """Return normalized by rolling volatility (risk-adjusted momentum)."""
    ret = close.pct_change()
    vol = ret.rolling(20).std()
    return ret / vol


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume z-score relative to rolling window."""
    mu = volume.rolling(window).mean()
    sigma = volume.rolling(window).std()
    return (volume - mu) / sigma


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — oscillator between 0 and 100."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd_signal(close: pd.Series) -> pd.Series:
    """MACD histogram — difference between MACD line and signal line."""
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line - signal_line


def bollinger_position(close: pd.Series, window: int = 20) -> pd.Series:
    """Position within Bollinger Bands — normalized to [-1, 1]."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (close - lower) / (upper - lower) * 2 - 1


def mean_reversion_20d(close: pd.Series) -> pd.Series:
    """Distance from 20-day SMA — mean reversion signal."""
    sma = close.rolling(20).mean()
    return (close - sma) / sma


def high_low_range_norm(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Daily range normalized by rolling average range."""
    raw_range = (high - low) / close
    avg_range = raw_range.rolling(20).mean()
    return raw_range / avg_range


def sma_ratio(close: pd.Series) -> pd.Series:
    """SMA 5/20 ratio — trend indicator."""
    sma_5 = close.rolling(5).mean()
    sma_20 = close.rolling(20).mean()
    return sma_5 / sma_20 - 1


def vwap_deviation(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Deviation from VWAP — measures price displacement from fair value."""
    vwap = (close * volume).rolling(window).sum() / volume.rolling(window).sum()
    return (close - vwap) / vwap


def turnover_rate(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume turnover rate — rolling volume change."""
    return volume / volume.rolling(window).mean()


def price_acceleration(close: pd.Series) -> pd.Series:
    """Second derivative of price — acceleration of momentum."""
    ret_1 = close.pct_change()
    ret_5 = close.pct_change(5)
    return ret_1 - ret_5 / 5


def overnight_return(open_price: pd.Series, close: pd.Series) -> pd.Series:
    """Gap between previous close and current open."""
    return open_price / close.shift(1) - 1


def intraday_return(open_price: pd.Series, close: pd.Series) -> pd.Series:
    """Return from open to close within the same day."""
    return close / open_price - 1


# ── Factor Registry ─────────────────────────────────────────────────

FACTOR_REGISTRY: dict[str, tuple[Any, list[str]]] = {
    "momentum_1d": (momentum_1d, ["Close"]),
    "momentum_5d": (momentum_5d, ["Close"]),
    "momentum_20d": (momentum_20d, ["Close"]),
    "momentum_60d": (momentum_60d, ["Close"]),
    "volatility_5d": (volatility_5d, ["Close"]),
    "volatility_20d": (volatility_20d, ["Close"]),
    "volatility_ratio": (volatility_ratio, ["Close"]),
    "momentum_normalized": (momentum_normalized, ["Close"]),
    "volume_zscore": (volume_zscore, ["Volume"]),
    "rsi": (rsi, ["Close"]),
    "macd_signal": (macd_signal, ["Close"]),
    "bollinger_position": (bollinger_position, ["Close"]),
    "mean_reversion_20d": (mean_reversion_20d, ["Close"]),
    "high_low_range_norm": (high_low_range_norm, ["High", "Low", "Close"]),
    "sma_ratio": (sma_ratio, ["Close"]),
    "vwap_deviation": (vwap_deviation, ["Close", "Volume"]),
    "turnover_rate": (turnover_rate, ["Volume"]),
    "price_acceleration": (price_acceleration, ["Close"]),
    "overnight_return": (overnight_return, ["Open", "Close"]),
    "intraday_return": (intraday_return, ["Open", "Close"]),
}


# ── IC/IR Analysis ──────────────────────────────────────────────────


def compute_factor_ic(factor_values: pd.Series, forward_returns: pd.Series) -> float:
    """
    Compute rank Information Coefficient (Spearman correlation) between
    a factor series and forward returns.
    """
    combined = pd.DataFrame(
        {"factor": factor_values, "returns": forward_returns}
    ).dropna()
    if len(combined) < 10:
        return 0.0
    corr, _ = np.spearmanr(combined["factor"], combined["returns"])
    return float(corr) if not np.isnan(corr) else 0.0


def compute_factor_ir(ic_series: pd.Series) -> float:
    """Compute Information Ratio = mean(IC) / std(IC)."""
    clean = ic_series.dropna()
    if len(clean) < 5 or clean.std() == 0:
        return 0.0
    return float(clean.mean() / clean.std())


def rank_factors(
    factor_dict: dict[str, pd.Series], forward_returns: pd.Series
) -> pd.DataFrame:
    """Rank factors by their IC and IR to identify the most predictive signals."""
    results = []
    for name, values in factor_dict.items():
        ic = compute_factor_ic(values, forward_returns)
        results.append({"factor": name, "ic": ic})
    df = pd.DataFrame(results)
    df["abs_ic"] = df["ic"].abs()
    df = df.sort_values("abs_ic", ascending=False).drop(columns=["abs_ic"])
    return df.reset_index(drop=True)


class AlphaFactorLibrary:
    """
    KG-native alpha factor library providing ~20 battle-tested factors
    with IC/IR analysis for factor selection.

    Usage:
        library = AlphaFactorLibrary()
        factors = library.compute_all(df)  # df has OHLCV columns
        ranking = library.rank_all(df, periods_ahead=1)
    """

    def __init__(self, factor_names: list[str] | None = None):
        if factor_names is not None:
            unknown = set(factor_names) - set(FACTOR_REGISTRY)
            if unknown:
                raise ValueError(f"Unknown factors: {unknown}")
            self.factor_names = factor_names
        else:
            self.factor_names = list(FACTOR_REGISTRY.keys())

    @property
    def available_factors(self) -> list[str]:
        """Return all available factor names."""
        return list(FACTOR_REGISTRY.keys())

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all configured factors from an OHLCV DataFrame."""
        col_map = {c: c.title() for c in df.columns}
        data = df.rename(columns=col_map)
        result = pd.DataFrame(index=data.index)

        for name in self.factor_names:
            func, required_cols = FACTOR_REGISTRY[name]
            missing = [c for c in required_cols if c not in data.columns]
            if missing:
                logger.warning(f"Skipping factor '{name}': missing columns {missing}")
                continue
            try:
                if len(required_cols) == 1:
                    result[name] = func(data[required_cols[0]])
                elif len(required_cols) == 2:
                    result[name] = func(data[required_cols[0]], data[required_cols[1]])
                elif len(required_cols) == 3:
                    result[name] = func(
                        data[required_cols[0]],
                        data[required_cols[1]],
                        data[required_cols[2]],
                    )
            except Exception:
                logger.exception(f"Error computing factor '{name}'")
                continue
        return result.dropna()

    def rank_all(self, df: pd.DataFrame, periods_ahead: int = 1) -> pd.DataFrame:
        """Compute all factors and rank them by IC against forward returns."""
        factors = self.compute_all(df)
        col_map = {c: c.title() for c in df.columns}
        data = df.rename(columns=col_map)
        forward_returns = data["Close"].pct_change(periods_ahead).shift(-periods_ahead)
        forward_returns = forward_returns.loc[factors.index]
        factor_dict = {col: factors[col] for col in factors.columns}
        return rank_factors(factor_dict, forward_returns)
