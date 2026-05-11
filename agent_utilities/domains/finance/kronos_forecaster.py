"""
Kronos Foundation Model Forecaster — CONCEPT:KG-2.6

K-line candlestick tokenizer and autoregressive transformer predictor
for financial time series forecasting. Requires [finance-kronos] extra.

Source: Kronos — A Foundation Model Architecture for Financial Candlesticks
Architecture: Two-stage framework:
  Stage 1: Specialized K-line tokenization (OHLCV → discrete token sequences)
  Stage 2: Autoregressive transformer prediction over token sequences
"""

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

logger = logging.getLogger(__name__)


class CandleType(StrEnum):
    """Candlestick pattern classification."""

    BULLISH_MARUBOZU = "bullish_marubozu"
    BEARISH_MARUBOZU = "bearish_marubozu"
    DOJI = "doji"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    SPINNING_TOP = "spinning_top"
    NEUTRAL = "neutral"


@dataclass
class KLineToken:
    """A discretized representation of a single candlestick bar."""

    body_direction: int  # +1 bullish, -1 bearish, 0 doji
    body_size_bucket: int  # 0-4 (tiny/small/medium/large/extreme)
    upper_wick_bucket: int  # 0-3
    lower_wick_bucket: int  # 0-3
    volume_bucket: int  # 0-4
    candle_type: CandleType = CandleType.NEUTRAL
    token_id: int = 0

    def to_int(self) -> int:
        """Encode token as a single integer for transformer consumption."""
        # 5 dimensions: direction(3) × body(5) × upper(4) × lower(4) × volume(5)
        d = self.body_direction + 1  # 0,1,2
        return (
            d * 400
            + self.body_size_bucket * 80
            + self.upper_wick_bucket * 20
            + self.lower_wick_bucket * 5
            + self.volume_bucket
        )

    @classmethod
    def vocab_size(cls) -> int:
        """Total vocabulary size for the tokenizer."""
        return 3 * 5 * 4 * 4 * 5  # 1200 unique tokens


@dataclass
class ForecastResult:
    """Result of a Kronos foundation model forecast."""

    predicted_tokens: list[KLineToken] = field(default_factory=list)
    predicted_direction: int = 0  # +1 up, -1 down, 0 flat
    confidence: float = 0.0
    horizon: int = 0
    method: str = "kronos"
    raw_logits: list[list[float]] | None = None


class KLineTokenizer:
    """
    Specialized K-line tokenizer that converts OHLCV candlestick data
    into discrete token sequences for transformer consumption.

    Quantization schema:
    - Body direction: {bullish, bearish, doji}
    - Body size: 5 buckets by percentile (tiny/small/medium/large/extreme)
    - Upper/Lower wicks: 4 buckets each
    - Volume: 5 buckets relative to rolling average
    """

    def __init__(self, lookback: int = 20, doji_threshold: float = 0.001):
        self.lookback = lookback
        self.doji_threshold = doji_threshold
        self._body_percentiles: np.ndarray | None = None
        self._wick_percentiles: np.ndarray | None = None
        self._vol_percentiles: np.ndarray | None = None

    def fit(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> "KLineTokenizer":
        """
        Fit the tokenizer on historical data to learn quantization boundaries.
        """
        bodies = np.abs(closes - opens) / np.where(opens > 0, opens, 1.0)
        upper_wicks = (highs - np.maximum(opens, closes)) / np.where(
            opens > 0, opens, 1.0
        )
        (np.minimum(opens, closes) - lows) / np.where(opens > 0, opens, 1.0)

        # Body size percentiles (5 buckets: 0-20-40-60-80-100)
        self._body_percentiles = (
            np.percentile(bodies[bodies > 0], [20, 40, 60, 80])
            if np.any(bodies > 0)
            else np.array([0.001, 0.005, 0.01, 0.02])
        )
        # Wick percentiles (4 buckets)
        self._wick_percentiles = (
            np.percentile(upper_wicks[upper_wicks > 0], [25, 50, 75])
            if np.any(upper_wicks > 0)
            else np.array([0.002, 0.005, 0.01])
        )
        # Volume percentiles (5 buckets)
        self._vol_percentiles = (
            np.percentile(volumes[volumes > 0], [20, 40, 60, 80])
            if np.any(volumes > 0)
            else np.array([1e4, 5e4, 1e5, 5e5])
        )

        return self

    def _bucketize(self, value: float, boundaries: np.ndarray) -> int:
        """Assign a value to a bucket based on percentile boundaries."""
        for i, b in enumerate(boundaries):
            if value <= b:
                return i
        return len(boundaries)

    def _classify_candle(
        self, open_p: float, high: float, low: float, close: float
    ) -> CandleType:
        """Classify a single candlestick into a pattern type."""
        body = abs(close - open_p)
        total_range = high - low
        if total_range == 0:
            return CandleType.DOJI

        body_ratio = body / total_range
        upper_wick = high - max(open_p, close)
        lower_wick = min(open_p, close) - low

        if body_ratio < 0.1:
            return CandleType.DOJI
        if body_ratio > 0.8:
            return (
                CandleType.BULLISH_MARUBOZU
                if close > open_p
                else CandleType.BEARISH_MARUBOZU
            )
        if lower_wick > body * 2 and upper_wick < body * 0.5:
            return CandleType.HAMMER
        if upper_wick > body * 2 and lower_wick < body * 0.5:
            return CandleType.SHOOTING_STAR
        if body_ratio < 0.3:
            return CandleType.SPINNING_TOP

        return CandleType.NEUTRAL

    def tokenize_bar(
        self, open_p: float, high: float, low: float, close: float, volume: float
    ) -> KLineToken:
        """Tokenize a single OHLCV bar into a KLineToken."""
        if self._body_percentiles is None:
            raise RuntimeError("Tokenizer not fitted. Call fit() first.")

        body = (close - open_p) / open_p if open_p > 0 else 0
        abs_body = abs(body)
        upper_wick = (high - max(open_p, close)) / open_p if open_p > 0 else 0
        lower_wick = (min(open_p, close) - low) / open_p if open_p > 0 else 0

        direction = 0 if abs_body < self.doji_threshold else (1 if body > 0 else -1)

        token = KLineToken(
            body_direction=direction,
            body_size_bucket=self._bucketize(abs_body, self._body_percentiles),
            upper_wick_bucket=self._bucketize(upper_wick, self._wick_percentiles),
            lower_wick_bucket=self._bucketize(lower_wick, self._wick_percentiles),
            volume_bucket=self._bucketize(volume, self._vol_percentiles),
            candle_type=self._classify_candle(open_p, high, low, close),
        )
        token.token_id = token.to_int()
        return token

    def tokenize_series(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> list[KLineToken]:
        """Tokenize a full OHLCV series into a sequence of KLineTokens."""
        n = len(opens)
        tokens = []
        for i in range(n):
            token = self.tokenize_bar(
                float(opens[i]),
                float(highs[i]),
                float(lows[i]),
                float(closes[i]),
                float(volumes[i]),
            )
            tokens.append(token)
        return tokens


class KronosPredictor:
    """
    Autoregressive predictor using token transition probabilities.

    This is the CPU-friendly baseline predictor that uses empirical
    transition matrices instead of a full transformer. When the
    [finance-kronos] extra is installed, this can be replaced with
    the full GPU transformer model.
    """

    def __init__(self, context_length: int = 32):
        self.context_length = context_length
        self._transition_matrix: dict[int, dict[int, int]] = {}
        self._total_counts: dict[int, int] = {}
        self._fitted = False

    def fit(self, token_ids: list[int]) -> "KronosPredictor":
        """Learn transition probabilities from a token sequence."""
        for i in range(len(token_ids) - 1):
            current = token_ids[i]
            next_tok = token_ids[i + 1]

            if current not in self._transition_matrix:
                self._transition_matrix[current] = {}
                self._total_counts[current] = 0

            self._transition_matrix[current][next_tok] = (
                self._transition_matrix[current].get(next_tok, 0) + 1
            )
            self._total_counts[current] += 1

        self._fitted = True
        return self

    def predict_next(self, context: list[int]) -> tuple[int, float]:
        """
        Predict the next token given context.

        Returns:
            (predicted_token_id, confidence)
        """
        if not self._fitted or not context:
            return 0, 0.0

        last_token = context[-1]
        if last_token not in self._transition_matrix:
            return 0, 0.0

        transitions = self._transition_matrix[last_token]
        total = self._total_counts[last_token]

        # Most likely next token
        best_token = max(transitions, key=lambda x: transitions[x])
        confidence = transitions[best_token] / total

        return best_token, float(confidence)

    def predict_sequence(self, context: list[int], horizon: int = 5) -> ForecastResult:
        """
        Autoregressively predict a sequence of future tokens.
        """
        predicted_ids = []
        current_context = list(context[-self.context_length :])
        avg_confidence = 0.0

        for _ in range(horizon):
            next_token, conf = self.predict_next(current_context)
            predicted_ids.append(next_token)
            avg_confidence += conf
            current_context.append(next_token)
            current_context = current_context[-self.context_length :]

        avg_confidence /= horizon if horizon > 0 else 1

        # Determine overall direction from predicted body directions
        bullish = sum(1 for t in predicted_ids if (t // 400) == 2)
        bearish = sum(1 for t in predicted_ids if (t // 400) == 0)
        direction = 1 if bullish > bearish else (-1 if bearish > bullish else 0)

        return ForecastResult(
            predicted_direction=direction,
            confidence=avg_confidence,
            horizon=horizon,
            method="kronos_transition_matrix",
        )


class KronosForecaster:
    """
    High-level Kronos forecasting interface combining tokenizer and predictor.

    Usage:
        forecaster = KronosForecaster()
        forecaster.fit(opens, highs, lows, closes, volumes)
        result = forecaster.forecast(horizon=5)
    """

    def __init__(self, context_length: int = 32, doji_threshold: float = 0.001):
        self.tokenizer = KLineTokenizer(doji_threshold=doji_threshold)
        self.predictor = KronosPredictor(context_length=context_length)
        self._last_tokens: list[KLineToken] = []

    def fit(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> "KronosForecaster":
        """Fit tokenizer and predictor on historical OHLCV data."""
        self.tokenizer.fit(opens, highs, lows, closes, volumes)
        self._last_tokens = self.tokenizer.tokenize_series(
            opens, highs, lows, closes, volumes
        )
        token_ids = [t.token_id for t in self._last_tokens]
        self.predictor.fit(token_ids)
        return self

    def forecast(self, horizon: int = 5) -> ForecastResult:
        """Generate autoregressive forecast for the next N bars."""
        if not self._last_tokens:
            return ForecastResult()
        token_ids = [t.token_id for t in self._last_tokens]
        return self.predictor.predict_sequence(token_ids, horizon)

    @property
    def vocab_size(self) -> int:
        return KLineToken.vocab_size()
