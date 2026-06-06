"""
Price-Action Pattern Classifier — CONCEPT:KG-2.6

Implements the four-pattern price-action framework and maps each to a
momentum-vs-mean-reversion edge label that a PATTERN_ANALYST swarm role
consumes:

  1. Large Bodies              → momentum (conviction/continuation)
  2. Wicks-into-levels         → mean-reversion (rejection at a level)
  3. Consecutive Candles       → momentum (sustained one-sided drive)
  4. Choppy                    → no edge (range / fade extremes lightly)

The candle-shape classification is local (cheap per-candle geometry — kept
in-process per the engine's batch-not-per-element rule), but the NUMERIC edge
confirmation is GROUNDED in the epistemic-graph engine signal kernels:
``client.finance.momentum``, ``mean_reversion``, ``rolling_zscore`` and
``detect_regimes``. Numbers are never hallucinated; when the engine is
unreachable the classifier still labels the shape but marks the edge as
``engine_confirmed=False``.

Source: price-action 4-pattern article; epistemic-graph signal kernels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

_ENGINE_PROBED = False
_ENGINE_CLIENT: Any = None


def _signal_engine() -> Any:
    """Return a connected SyncEpistemicGraphClient, or ``None`` if unavailable."""
    global _ENGINE_PROBED, _ENGINE_CLIENT
    if _ENGINE_PROBED:
        return _ENGINE_CLIENT
    _ENGINE_PROBED = True
    try:
        from epistemic_graph.client import SyncEpistemicGraphClient

        _ENGINE_CLIENT = SyncEpistemicGraphClient.connect()
        logger.info("epistemic-graph engine connected for pattern signal kernels")
    except Exception as exc:  # noqa: BLE001 — degrade gracefully
        logger.debug("epistemic-graph engine unavailable for pattern signals: %s", exc)
        _ENGINE_CLIENT = None
    return _ENGINE_CLIENT


def reset_engine_cache() -> None:
    """Reset the cached engine probe (used by tests to re-probe)."""
    global _ENGINE_PROBED, _ENGINE_CLIENT
    _ENGINE_PROBED = False
    _ENGINE_CLIENT = None


class PricePattern(StrEnum):
    """The four price-action patterns."""

    LARGE_BODY = "large_body"
    WICK_INTO_LEVEL = "wick_into_level"
    CONSECUTIVE = "consecutive"
    CHOPPY = "choppy"


class EdgeLabel(StrEnum):
    """The trading edge a pattern implies."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    NO_EDGE = "no_edge"


# Static mapping from the four patterns to their edge family.
_PATTERN_EDGE: dict[PricePattern, EdgeLabel] = {
    PricePattern.LARGE_BODY: EdgeLabel.MOMENTUM,
    PricePattern.WICK_INTO_LEVEL: EdgeLabel.MEAN_REVERSION,
    PricePattern.CONSECUTIVE: EdgeLabel.MOMENTUM,
    PricePattern.CHOPPY: EdgeLabel.NO_EDGE,
}


@dataclass
class Candle:
    """A single OHLC candle."""

    open: float
    high: float
    low: float
    close: float

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return max(self.high - self.low, 1e-12)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def direction(self) -> int:
        return 1 if self.close > self.open else (-1 if self.close < self.open else 0)


@dataclass
class PatternClassification:
    """Result of classifying a price-action window."""

    pattern: PricePattern
    edge: EdgeLabel
    direction: int  # +1 bullish, -1 bearish, 0 neutral
    confidence: float  # 0.0–1.0 (shape strength × engine confirmation)
    engine_confirmed: bool  # True when the edge was numerically confirmed
    rationale: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_signal_metadata(self) -> dict[str, Any]:
        """Shape suitable for a PATTERN_ANALYST ``AgentSignal.metadata``."""
        return {
            "pattern": str(self.pattern),
            "edge": str(self.edge),
            "engine_confirmed": self.engine_confirmed,
            "confidence": self.confidence,
            **self.metrics,
        }


def _to_candles(window: list[Any]) -> list[Candle]:
    candles: list[Candle] = []
    for c in window:
        if isinstance(c, Candle):
            candles.append(c)
        elif isinstance(c, dict):
            candles.append(
                Candle(
                    open=float(c["open"]),
                    high=float(c["high"]),
                    low=float(c["low"]),
                    close=float(c["close"]),
                )
            )
        elif isinstance(c, list | tuple) and len(c) >= 4:
            o, hi, lo, cl = c[0], c[1], c[2], c[3]
            candles.append(Candle(float(o), float(hi), float(lo), float(cl)))
        else:
            raise TypeError(f"Unsupported candle representation: {c!r}")
    return candles


class PatternClassifier:
    """Classify a window of candles into one of the four price-action patterns
    and attach an engine-confirmed momentum/mean-reversion edge.

    Usage::

        clf = PatternClassifier()
        result = clf.classify(candles, levels=[101.5])  # candles: OHLC dicts
        # feed result.to_signal_metadata() into a PATTERN_ANALYST AgentSignal
    """

    def __init__(
        self,
        engine_client: Any | None = None,
        large_body_ratio: float = 0.7,
        wick_ratio: float = 0.5,
        level_tolerance: float = 0.002,
        zscore_window: int = 20,
    ):
        self._explicit_client = engine_client
        self.large_body_ratio = large_body_ratio
        self.wick_ratio = wick_ratio
        self.level_tolerance = level_tolerance
        self.zscore_window = zscore_window

    def _client(self) -> Any:
        if self._explicit_client is not None:
            return self._explicit_client
        return _signal_engine()

    # ── shape classification (local geometry) ─────────────────────────────────
    def _classify_shape(
        self, candles: list[Candle], levels: list[float] | None
    ) -> tuple[PricePattern, int, float, str, dict[str, Any]]:
        last = candles[-1]
        body_frac = last.body / last.range
        upper_frac = last.upper_wick / last.range
        lower_frac = last.lower_wick / last.range

        metrics: dict[str, Any] = {
            "body_frac": round(body_frac, 4),
            "upper_wick_frac": round(upper_frac, 4),
            "lower_wick_frac": round(lower_frac, 4),
        }

        # Consecutive candles: run of same-direction closes incl. the last one.
        run = 1
        for i in range(len(candles) - 2, -1, -1):
            if candles[i].direction == last.direction and last.direction != 0:
                run += 1
            else:
                break
        metrics["consecutive_run"] = run

        near_level = False
        if levels:
            extreme = last.high if upper_frac >= lower_frac else last.low
            for lv in levels:
                if lv and abs(extreme - lv) / abs(lv) <= self.level_tolerance:
                    near_level = True
                    metrics["nearest_level"] = lv
                    break

        # Priority: rejection wick into a level → mean-reversion fade.
        if near_level and max(upper_frac, lower_frac) >= self.wick_ratio:
            # Rejection direction is opposite the dominant wick.
            direction = -1 if upper_frac >= lower_frac else 1
            conf = min(1.0, max(upper_frac, lower_frac))
            return (
                PricePattern.WICK_INTO_LEVEL,
                direction,
                conf,
                "Long wick rejecting a key level — mean-reversion fade.",
                metrics,
            )

        # Large body: strong directional conviction → momentum continuation.
        if body_frac >= self.large_body_ratio and last.direction != 0:
            return (
                PricePattern.LARGE_BODY,
                last.direction,
                min(1.0, body_frac),
                "Large-bodied candle with little wick — momentum continuation.",
                metrics,
            )

        # Consecutive same-direction candles → sustained momentum drive.
        if run >= 3 and last.direction != 0:
            return (
                PricePattern.CONSECUTIVE,
                last.direction,
                min(1.0, 0.4 + 0.15 * run),
                f"{run} consecutive {'up' if last.direction > 0 else 'down'} "
                "candles — sustained momentum.",
                metrics,
            )

        # Otherwise choppy / indecisive → no edge.
        return (
            PricePattern.CHOPPY,
            0,
            0.3,
            "Small bodies / mixed wicks — choppy range, no clean edge.",
            metrics,
        )

    # ── numeric edge confirmation (engine-grounded) ───────────────────────────
    def _confirm_edge(
        self,
        closes: list[float],
        pattern: PricePattern,
        direction: int,
    ) -> tuple[bool, dict[str, Any]]:
        """Confirm the shape's edge with engine signal kernels.

        Returns ``(engine_confirmed, metrics)``. ``engine_confirmed`` is True
        only when the engine returned numbers AND they agree with the pattern's
        implied edge family / direction.
        """
        client = self._client()
        if client is None or len(closes) < 3:
            return False, {}

        edge = _PATTERN_EDGE[pattern]
        metrics: dict[str, Any] = {}
        try:
            fin = client.finance
            if edge is EdgeLabel.MOMENTUM:
                lookback = min(self.zscore_window, len(closes) - 1)
                mom = fin.momentum(closes, lookback)
                last_mom = float(mom[-1]) if mom else 0.0
                metrics["momentum"] = round(last_mom, 6)
                confirmed = (last_mom > 0 and direction > 0) or (
                    last_mom < 0 and direction < 0
                )
                return confirmed, metrics
            if edge is EdgeLabel.MEAN_REVERSION:
                window = min(self.zscore_window, len(closes))
                mr = fin.mean_reversion(closes, window)
                z = fin.rolling_zscore(closes, window)
                last_mr = float(mr[-1]) if mr else 0.0
                last_z = float(z[-1]) if z else 0.0
                metrics["mean_reversion"] = round(last_mr, 6)
                metrics["rolling_zscore"] = round(last_z, 6)
                # Fade an extreme: a stretched z-score opposite the fade direction
                # confirms the reversion edge.
                confirmed = (last_z > 1.0 and direction < 0) or (
                    last_z < -1.0 and direction > 0
                )
                return confirmed, metrics
            # NO_EDGE (choppy): a regime classification still grounds the call.
            regimes = fin.detect_regimes(closes, n_states=2)
            if isinstance(regimes, dict):
                metrics["n_regimes"] = regimes.get("n_states", 2)
            return False, metrics
        except Exception as exc:  # noqa: BLE001 — degrade, never invent
            logger.debug("Engine edge confirmation failed for %s: %s", pattern, exc)
            return False, {}

    def classify(
        self,
        window: list[Any],
        levels: list[float] | None = None,
    ) -> PatternClassification:
        """Classify a window of OHLC candles into a pattern + engine-grounded edge.

        Args:
            window: A list of candles as ``Candle``, OHLC dicts
                (``{open,high,low,close}``), or ``[o,h,l,c]`` sequences.
            levels: Optional support/resistance levels for wick-into-level
                detection.

        Returns:
            A ``PatternClassification`` whose ``edge`` follows the static
            4-pattern map and whose ``engine_confirmed`` flag and numeric
            ``metrics`` come from the epistemic-graph signal kernels.
        """
        candles = _to_candles(window)
        if not candles:
            raise ValueError("classify() requires at least one candle")

        pattern, direction, shape_conf, rationale, metrics = self._classify_shape(
            candles, levels
        )
        edge = _PATTERN_EDGE[pattern]

        closes = [c.close for c in candles]
        engine_confirmed, engine_metrics = self._confirm_edge(
            closes, pattern, direction
        )
        metrics.update(engine_metrics)

        # Confidence: shape strength, boosted when the engine confirms the edge.
        confidence = shape_conf
        if edge is not EdgeLabel.NO_EDGE:
            confidence = min(1.0, shape_conf * (1.15 if engine_confirmed else 0.6))

        return PatternClassification(
            pattern=pattern,
            edge=edge,
            direction=direction,
            confidence=round(confidence, 4),
            engine_confirmed=engine_confirmed,
            rationale=rationale,
            metrics=metrics,
        )
