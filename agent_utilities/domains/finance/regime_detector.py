"""
Regime Detector — CONCEPT:AU-KG.research.research-pipeline-runner
Detects market regimes (Bull, Bear, Sideways, High Volatility) using heuristics.
Inspired by qlib's regime-aware model switching.
"""

import logging
import math
from collections.abc import Sequence
from typing import Any

from agent_utilities.numeric import xp

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects market regimes based on price and volatility data.
    """

    def __init__(self, engine: Any | None = None):
        self.engine = engine

    def detect_close_prices(
        self, close_prices: Sequence[float], ticker: str = ""
    ) -> str:
        """Detect a regime from a bounded close-price sequence.

        The numeric authority remains the epistemic-graph kernel: the rolling
        window enters each native statistic as one batch, never one PyO3 call
        per element. DataFrame libraries are optional adapters outside this
        domain operation.
        """
        prices = [float(value) for value in close_prices]
        if len(prices) < 50:
            return "unknown"

        returns = [
            (current / previous) - 1.0
            for previous, current in zip(prices[-21:-1], prices[-20:], strict=True)
            if previous != 0.0
        ]
        if len(returns) != 20:
            return "unknown"
        volatility = float(xp.std(returns)) * math.sqrt(252.0)
        sma_50 = float(xp.mean(prices[-50:]))
        current_price = prices[-1]

        # Heuristics
        if volatility > 0.40:
            regime = "high_volatility"
        elif current_price > sma_50 * 1.05:
            regime = "bull_market"
        elif current_price < sma_50 * 0.95:
            regime = "bear_market"
        else:
            regime = "sideways_market"

        logger.info("Detected %s for %s", regime, ticker)

        if self.engine and ticker:
            self._persist_to_kg(ticker, regime, volatility)

        return regime

    def detect_regime(self, table: Any, ticker: str = "") -> str:
        """Compatibility adapter for tabular callers without importing pandas.

        Arrow columns use ``to_pylist``. DataFrame-like edge callers may expose
        ``tolist``; conversion is immediate and the hot path remains the native
        sequence operation above.
        """
        if table is None or bool(getattr(table, "empty", False)):
            return "unknown"
        try:
            close_column = table["Close"]
        except (KeyError, TypeError):
            return "unknown"
        to_pylist = getattr(close_column, "to_pylist", None)
        if callable(to_pylist):
            values = to_pylist()
        else:
            tolist = getattr(close_column, "tolist", None)
            values = tolist() if callable(tolist) else list(close_column)
        return self.detect_close_prices(values, ticker=ticker)

    def _persist_to_kg(self, ticker: str, regime: str, volatility: float) -> None:
        """Persist current regime to KG for routing and strategy selection."""
        assert self.engine is not None
        node_id = f"Regime_{ticker}"
        self.engine.add_node(
            node_id=node_id,
            node_type="MarketRegime",
            properties={
                "ticker": ticker,
                "regime_type": regime,
                "volatility": volatility,
            },
        )
