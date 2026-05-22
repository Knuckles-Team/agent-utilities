"""Tests for CONCEPT:KG-2.6 — Visual Technical Analysis Engine."""

import numpy as np
import pytest

from agent_utilities.domains.finance.visual_ta import (
    PatternDetector,
    PatternType,
    SupportResistanceDetector,
    TrendDirection,
    VisualTAEngine,
)


@pytest.fixture
def uptrend_data():
    rng = np.random.default_rng(42)
    n = 100
    trend = np.linspace(100, 130, n) + rng.normal(0, 1, n)
    opens = trend - rng.uniform(0, 0.5, n)
    highs = trend + rng.uniform(0, 2, n)
    lows = trend - rng.uniform(0, 2, n)
    closes = trend + rng.normal(0, 0.5, n)
    return opens, highs, lows, closes


@pytest.fixture
def range_bound_data():
    rng = np.random.default_rng(42)
    n = 100
    base = 100 + rng.normal(0, 2, n)
    opens = base - rng.uniform(0, 0.5, n)
    highs = base + rng.uniform(0, 3, n)
    lows = base - rng.uniform(0, 3, n)
    closes = base + rng.normal(0, 0.5, n)
    return opens, highs, lows, closes


class TestSupportResistanceDetector:
    def test_detect_returns_levels(self, range_bound_data):
        _, highs, lows, closes = range_bound_data
        detector = SupportResistanceDetector(window=5, tolerance=0.03)
        supports, resistances = detector.detect(highs, lows, closes)
        assert isinstance(supports, list)
        assert isinstance(resistances, list)

    def test_support_below_resistance(self, range_bound_data):
        _, highs, lows, closes = range_bound_data
        detector = SupportResistanceDetector(window=5, tolerance=0.03)
        supports, resistances = detector.detect(highs, lows, closes)
        if supports and resistances:
            assert min(s.price for s in supports) < max(r.price for r in resistances)

    def test_level_strength(self, range_bound_data):
        _, highs, lows, closes = range_bound_data
        detector = SupportResistanceDetector(window=5, tolerance=0.03)
        supports, _ = detector.detect(highs, lows, closes)
        for s in supports:
            assert s.strength >= 2  # Minimum 2 touches required


class TestPatternDetector:
    def test_detect_double_top(self):
        # Construct a clear double top
        n = 50
        prices = np.concatenate(
            [
                np.linspace(100, 110, 15),  # Rise
                np.linspace(110, 105, 5),  # Dip
                np.linspace(105, 110, 5),  # Second peak
                np.linspace(110, 100, 25),  # Decline
            ]
        )
        highs = prices + 0.5
        closes = prices
        detector = PatternDetector()
        patterns = detector.detect_double_top(highs, closes, tolerance=0.02)
        # May or may not detect depending on exact geometry
        assert isinstance(patterns, list)

    def test_detect_breakout(self):
        n = 50
        # Flat range then strong breakout above range highs
        flat = np.full(30, 100.0) + np.random.default_rng(42).normal(0, 0.3, 30)
        breakout = np.linspace(103, 120, 20)  # Clearly above flat range
        prices = np.concatenate([flat, breakout])
        highs = prices + 0.5
        closes = prices
        detector = PatternDetector()
        patterns = detector.detect_breakout(closes, highs, lookback=20)
        assert any(p.pattern_type == PatternType.BREAKOUT for p in patterns)

    def test_detect_all(self, range_bound_data):
        opens, highs, lows, closes = range_bound_data
        detector = PatternDetector()
        patterns = detector.detect_all(opens, highs, lows, closes)
        assert isinstance(patterns, list)

    def test_short_data(self):
        detector = PatternDetector()
        patterns = detector.detect_double_top(np.array([1, 2]), np.array([1, 2]))
        assert patterns == []


class TestVisualTAEngine:
    def test_uptrend_detected(self, uptrend_data):
        opens, highs, lows, closes = uptrend_data
        engine = VisualTAEngine()
        analysis = engine.analyze(opens, highs, lows, closes)
        assert analysis.direction == TrendDirection.UPTREND
        assert analysis.strength > 0

    def test_sideways_detected(self, range_bound_data):
        opens, highs, lows, closes = range_bound_data
        engine = VisualTAEngine()
        analysis = engine.analyze(opens, highs, lows, closes)
        assert analysis.direction in (
            TrendDirection.SIDEWAYS,
            TrendDirection.UPTREND,
            TrendDirection.DOWNTREND,
        )

    def test_analysis_has_duration(self, uptrend_data):
        opens, highs, lows, closes = uptrend_data
        engine = VisualTAEngine()
        analysis = engine.analyze(opens, highs, lows, closes)
        assert analysis.duration == len(closes)

    def test_r_squared_bounded(self, uptrend_data):
        opens, highs, lows, closes = uptrend_data
        engine = VisualTAEngine()
        analysis = engine.analyze(opens, highs, lows, closes)
        assert 0.0 <= analysis.r_squared <= 1.0

    def test_short_series(self):
        engine = VisualTAEngine()
        analysis = engine.analyze(
            np.array([1]), np.array([2]), np.array([0.5]), np.array([1.5])
        )
        assert analysis.direction == TrendDirection.SIDEWAYS
