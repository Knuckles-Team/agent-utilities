import logging

try:
    import pandas as pd
    from statsmodels.tsa.stattools import adfuller
except ImportError as e:
    raise ImportError(
        "Finance extra dependencies missing. Please install agent-utilities[finance]"
    ) from e

logger = logging.getLogger(__name__)


def check_stationarity(series: pd.Series, name: str, threshold: float = 0.05) -> bool:
    """
    Perform Augmented Dickey-Fuller test on a time series.
    Returns True if the series is stationary (p-value < threshold).
    """
    # Drop any NaNs before testing
    clean_series = series.dropna()
    if len(clean_series) < 20:
        logger.warning(f"{name}: Not enough data for ADF test.")
        return False

    result = adfuller(clean_series)
    p_value = result[1]
    is_stationary = p_value < threshold
    logger.debug(f"{name}: ADF statistic={result[0]:.4f}, p-value={p_value:.4f}")
    return is_stationary


class StationaryFeatureEngineer:
    """
    Engineers stationary features from raw OHLCV market data.
    Designed to work across asset classes (equities, crypto, derivatives).
    """

    def __init__(self, check_all: bool = True, regularize: bool = False):
        self.check_all = check_all
        # CONCEPT:KG-2.252 — when True, gappy/irregular raw OHLCV is gap-filled onto a
        # regular daily grid IN-ENGINE (timeseries.gap_fill, LOCF) before features are
        # computed. The rolling/ewm feature math itself stays in pandas (already
        # vectorized + optimal); only the irregular-series alignment — the clear engine
        # win pandas lacks natively — is routed to the engine. Default OFF (most callers
        # already pass a regular series); a real disable/enable case, not a knob.
        self.regularize = regularize

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Takes OHLCV dataframe, returns (features_df, target_series).
        Target is next-period direction (binary classification).

        With ``regularize=True`` (CONCEPT:KG-2.252) the OHLCV columns are first
        gap-filled onto a regular daily grid via the engine tsdb (``gap_fill``), so a
        feed with missing bars yields aligned features without hand-rolled reindexing.
        """
        if self.regularize:
            df = self._regularize_ohlcv(df)
        features = pd.DataFrame(index=df.index)
        close = df["Close"]
        volume = df["Volume"]
        returns = close.pct_change()

        # Base log/pct returns
        features["return_1d"] = returns
        features["return_5d"] = close.pct_change(5)
        features["return_20d"] = close.pct_change(20)

        # Volatility ratios
        vol_5 = returns.rolling(5).std()
        vol_20 = returns.rolling(20).std()
        features["vol_ratio"] = vol_5 / vol_20
        features["momentum_norm"] = returns / vol_20

        # Volume
        features["volume_zscore"] = (
            volume - volume.rolling(20).mean()
        ) / volume.rolling(20).std()

        # Price range
        high_low_range = (df["High"] - df["Low"]) / close
        features["range_norm"] = high_low_range / high_low_range.rolling(20).mean()

        # SMA ratio
        sma_5 = close.rolling(5).mean()
        sma_20 = close.rolling(20).mean()
        features["sma_ratio"] = sma_5 / sma_20 - 1

        # Target variable: Next period direction
        target = (returns.shift(-1) > 0).astype(int)

        # Drop rows with NaN due to rolling windows
        features = features.dropna()
        target = target.loc[features.index]

        if self.check_all:
            non_stationary = []
            for col in features.columns:
                if not check_stationarity(features[col], col):
                    non_stationary.append(col)
            if non_stationary:
                logger.warning(f"Non-stationary features detected: {non_stationary}")

        return features, target

    def _regularize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gap-fill each OHLCV column onto a regular daily grid IN-ENGINE (KG-2.252).

        Routes the irregular→regular alignment to ``engine_series.gap_fill_series``
        (the engine's native LOCF gap-fill) — ONE engine connection reused across the
        columns — and returns a new aligned DataFrame. Degrades to a pandas reindex
        when no engine is reachable (gap_fill_series handles that fallback), so the
        public feature path always works.
        """
        from .engine_series import _client, gap_fill_series

        client = _client()
        try:
            cols = {}
            for name in df.columns:
                cols[name] = gap_fill_series(df[name], "1D", client=client)
            return pd.DataFrame(cols)
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    pass
