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

    def __init__(self, check_all: bool = True):
        self.check_all = check_all

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Takes OHLCV dataframe, returns (features_df, target_series).
        Target is next-period direction (binary classification).
        """
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
