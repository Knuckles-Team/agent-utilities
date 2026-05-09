"""
Visual Technical Analysis Agent — CONCEPT:KG-2.72

Provides programmatic chart pattern detection and generation for
visual technical analysis — identifying candlestick patterns,
support/resistance levels, and trend formations.

Source: QuantAgent visual chart analysis patterns
"""

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

logger = logging.getLogger(__name__)


class PatternType(StrEnum):
    """Recognized chart pattern types."""

    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_REJECTION = "resistance_rejection"


class TrendDirection(StrEnum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"


@dataclass
class SupportResistanceLevel:
    """A detected support or resistance price level."""

    price: float
    level_type: str  # "support" or "resistance"
    strength: float = 0.0  # Number of touches
    first_touch_idx: int = 0
    last_touch_idx: int = 0


@dataclass
class DetectedPattern:
    """A pattern detected in price data."""

    pattern_type: PatternType
    start_idx: int
    end_idx: int
    confidence: float
    implied_direction: int  # +1 bullish, -1 bearish
    target_price: float | None = None
    stop_loss: float | None = None
    description: str = ""


@dataclass
class TrendAnalysis:
    """Result of a trend analysis."""

    direction: TrendDirection
    strength: float  # 0.0 to 1.0
    slope: float
    r_squared: float
    duration: int  # bars
    support_levels: list[SupportResistanceLevel] = field(default_factory=list)
    resistance_levels: list[SupportResistanceLevel] = field(default_factory=list)
    patterns: list[DetectedPattern] = field(default_factory=list)


class SupportResistanceDetector:
    """
    Detects support and resistance levels from price data using
    local extrema clustering.
    """

    def __init__(self, window: int = 10, tolerance: float = 0.02):
        self.window = window
        self.tolerance = tolerance

    def detect(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> tuple[list[SupportResistanceLevel], list[SupportResistanceLevel]]:
        """Detect support and resistance levels."""
        supports = []
        resistances = []

        # Find local minima (supports)
        for i in range(self.window, len(lows) - self.window):
            if lows[i] == np.min(lows[i - self.window : i + self.window + 1]):
                # Count touches at this level
                level = float(lows[i])
                touches = sum(
                    1
                    for j in range(len(lows))
                    if abs(lows[j] - level) / level < self.tolerance
                )
                if touches >= 2:
                    supports.append(
                        SupportResistanceLevel(
                            price=level,
                            level_type="support",
                            strength=float(touches),
                            first_touch_idx=i,
                            last_touch_idx=max(
                                j
                                for j in range(len(lows))
                                if abs(lows[j] - level) / level < self.tolerance
                            ),
                        )
                    )

        # Find local maxima (resistances)
        for i in range(self.window, len(highs) - self.window):
            if highs[i] == np.max(highs[i - self.window : i + self.window + 1]):
                level = float(highs[i])
                touches = sum(
                    1
                    for j in range(len(highs))
                    if abs(highs[j] - level) / level < self.tolerance
                )
                if touches >= 2:
                    resistances.append(
                        SupportResistanceLevel(
                            price=level,
                            level_type="resistance",
                            strength=float(touches),
                            first_touch_idx=i,
                            last_touch_idx=max(
                                j
                                for j in range(len(highs))
                                if abs(highs[j] - level) / level < self.tolerance
                            ),
                        )
                    )

        # Deduplicate nearby levels
        supports = self._deduplicate(supports)
        resistances = self._deduplicate(resistances)

        return supports, resistances

    def _deduplicate(
        self, levels: list[SupportResistanceLevel]
    ) -> list[SupportResistanceLevel]:
        """Merge levels within tolerance of each other."""
        if not levels:
            return []
        sorted_levels = sorted(levels, key=lambda x: x.price)
        merged = [sorted_levels[0]]
        for level in sorted_levels[1:]:
            if abs(level.price - merged[-1].price) / merged[-1].price < self.tolerance:
                # Merge: keep the stronger one
                if level.strength > merged[-1].strength:
                    merged[-1] = level
            else:
                merged.append(level)
        return merged


class PatternDetector:
    """
    Detects classic chart patterns in price data.
    """

    def detect_double_top(
        self, highs: np.ndarray, closes: np.ndarray, tolerance: float = 0.02
    ) -> list[DetectedPattern]:
        """Detect double top formations."""
        patterns: list[DetectedPattern] = []
        n = len(highs)
        if n < 20:
            return patterns

        for i in range(10, n - 10):
            peak1 = highs[i]
            # Look for second peak within ±tolerance
            for j in range(i + 5, min(i + 30, n)):
                peak2 = highs[j]
                if abs(peak1 - peak2) / peak1 < tolerance:
                    # Check for valley between peaks
                    valley = np.min(closes[i:j])
                    if valley < peak1 * (1 - tolerance):
                        neckline = float(valley)
                        target = neckline - (peak1 - neckline)
                        patterns.append(
                            DetectedPattern(
                                pattern_type=PatternType.DOUBLE_TOP,
                                start_idx=i,
                                end_idx=j,
                                confidence=0.7,
                                implied_direction=-1,
                                target_price=float(target),
                                stop_loss=float(max(peak1, peak2) * 1.01),
                                description=f"Double top at {peak1:.2f}/{peak2:.2f}, neckline {neckline:.2f}",
                            )
                        )
                        break
        return patterns

    def detect_double_bottom(
        self, lows: np.ndarray, closes: np.ndarray, tolerance: float = 0.02
    ) -> list[DetectedPattern]:
        """Detect double bottom formations."""
        patterns: list[DetectedPattern] = []
        n = len(lows)
        if n < 20:
            return patterns

        for i in range(10, n - 10):
            trough1 = lows[i]
            for j in range(i + 5, min(i + 30, n)):
                trough2 = lows[j]
                if abs(trough1 - trough2) / trough1 < tolerance:
                    peak = np.max(closes[i:j])
                    if peak > trough1 * (1 + tolerance):
                        neckline = float(peak)
                        target = neckline + (neckline - trough1)
                        patterns.append(
                            DetectedPattern(
                                pattern_type=PatternType.DOUBLE_BOTTOM,
                                start_idx=i,
                                end_idx=j,
                                confidence=0.7,
                                implied_direction=1,
                                target_price=float(target),
                                stop_loss=float(min(trough1, trough2) * 0.99),
                                description=f"Double bottom at {trough1:.2f}/{trough2:.2f}",
                            )
                        )
                        break
        return patterns

    def detect_breakout(
        self, closes: np.ndarray, highs: np.ndarray, lookback: int = 20
    ) -> list[DetectedPattern]:
        """Detect breakout above recent highs."""
        patterns: list[DetectedPattern] = []
        n = len(closes)
        if n < lookback + 1:
            return patterns

        for i in range(lookback, n):
            recent_high = np.max(highs[i - lookback : i])
            if closes[i] > recent_high * 1.01:
                patterns.append(
                    DetectedPattern(
                        pattern_type=PatternType.BREAKOUT,
                        start_idx=i - lookback,
                        end_idx=i,
                        confidence=min(1.0, (closes[i] / recent_high - 1) * 20),
                        implied_direction=1,
                        description=f"Breakout above {recent_high:.2f}",
                    )
                )
        return patterns

    def detect_all(
        self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> list[DetectedPattern]:
        """Run all pattern detectors."""
        patterns = []
        patterns.extend(self.detect_double_top(highs, closes))
        patterns.extend(self.detect_double_bottom(lows, closes))
        patterns.extend(self.detect_breakout(closes, highs))
        return patterns


class VisualTAEngine:
    """
    Unified visual technical analysis engine combining trend analysis,
    support/resistance detection, and pattern recognition.

    Usage:
        engine = VisualTAEngine()
        analysis = engine.analyze(opens, highs, lows, closes, volumes)
    """

    def __init__(self, sr_window: int = 10, sr_tolerance: float = 0.02):
        self.sr_detector = SupportResistanceDetector(
            window=sr_window, tolerance=sr_tolerance
        )
        self.pattern_detector = PatternDetector()

    def compute_trend(
        self, closes: np.ndarray
    ) -> tuple[TrendDirection, float, float, float]:
        """Compute trend direction and strength using linear regression."""
        n = len(closes)
        if n < 5:
            return TrendDirection.SIDEWAYS, 0.0, 0.0, 0.0

        x = np.arange(n, dtype=float)
        x_mean = np.mean(x)
        y_mean = np.mean(closes)

        ss_xy = np.sum((x - x_mean) * (closes - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)
        ss_yy = np.sum((closes - y_mean) ** 2)

        slope = ss_xy / ss_xx if ss_xx > 0 else 0.0
        r_squared = (ss_xy**2) / (ss_xx * ss_yy) if (ss_xx * ss_yy) > 0 else 0.0

        # Normalize slope by average price for comparability
        norm_slope = slope / y_mean if y_mean > 0 else 0.0

        if norm_slope > 0.0005 and r_squared > 0.3:
            direction = TrendDirection.UPTREND
        elif norm_slope < -0.0005 and r_squared > 0.3:
            direction = TrendDirection.DOWNTREND
        else:
            direction = TrendDirection.SIDEWAYS

        strength = min(1.0, abs(norm_slope) * 500 * r_squared)

        return direction, float(strength), float(slope), float(r_squared)

    def analyze(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> TrendAnalysis:
        """Run full visual technical analysis."""
        direction, strength, slope, r_sq = self.compute_trend(closes)
        supports, resistances = self.sr_detector.detect(highs, lows, closes)
        patterns = self.pattern_detector.detect_all(opens, highs, lows, closes)

        return TrendAnalysis(
            direction=direction,
            strength=strength,
            slope=slope,
            r_squared=r_sq,
            duration=len(closes),
            support_levels=supports,
            resistance_levels=resistances,
            patterns=patterns,
        )
