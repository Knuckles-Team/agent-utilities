"""
Regime Detector — CONCEPT:KG-2.6
Detects market regimes (Bull, Bear, Sideways, High Volatility) using HMMs.
Inspired by qlib's regime-aware model switching.
"""

import logging

import numpy as np

try:
    import pandas as pd
except ImportError:
    pass

from typing import Any

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects market regimes based on price and volatility data.
    """

    def __init__(self, engine: Any | None = None):
        self.engine = engine

    def detect_regime(self, df: "pd.DataFrame", ticker: str = "") -> str:
        """
        Simple heuristic-based regime detection.
        In a full implementation, this would use hmmlearn.GaussianHMM.
        """
        if df.empty or len(df) < 50:
            return "unknown"

        # Calculate 50-day moving average and volatility
        returns = df["Close"].pct_change().dropna()
        volatility = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)

        sma_50 = df["Close"].rolling(window=50).mean().iloc[-1]
        current_price = df["Close"].iloc[-1]

        # Heuristics
        if volatility > 0.40:
            regime = "high_volatility"
        elif current_price > sma_50 * 1.05:
            regime = "bull_market"
        elif current_price < sma_50 * 0.95:
            regime = "bear_market"
        else:
            regime = "sideways_market"

        logger.info(f"Detected {regime} for {ticker}")

        if self.engine and ticker:
            self._persist_to_kg(ticker, regime, volatility)

        return regime

    def _persist_to_kg(self, ticker: str, regime: str, volatility: float) -> None:
        """Persist current regime to KG for routing and strategy selection."""
        assert self.engine is not None
        node_id = f"Regime_{ticker}"
        self.engine.add_node(
            node_id=node_id,
            node_type="MarketRegime",
            ticker=ticker,
            regime_type=regime,
            volatility=volatility,
        )
