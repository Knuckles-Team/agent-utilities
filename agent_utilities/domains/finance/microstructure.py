"""
Microstructure quantitative methodologies (CONCEPT:KG-2.6).

Contains models for High-Frequency Micro-Price, Order Book Imbalance (OBI),
and consensus logic (Convergence Filter, Brier Score Validator).
"""

import logging

from agent_utilities.numeric import xp as np

logger = logging.getLogger(__name__)


class OrderBookImbalance:
    """
    Computes Level 1 Order Book Imbalance (OBI).
    """

    @staticmethod
    def calculate(bid_volume: float, ask_volume: float) -> float:
        """
        Calculate OBI from best bid and ask volumes.

        Args:
            bid_volume: Volume available at the best bid price.
            ask_volume: Volume available at the best ask price.

        Returns:
            Imbalance scalar in [-1, 1].
        """
        total_vol = bid_volume + ask_volume
        if total_vol == 0:
            return 0.0
        return (bid_volume - ask_volume) / total_vol


class MicroPriceCalculator:
    """
    Computes instantaneous Micro-Price using OBI.
    """

    @staticmethod
    def calculate(
        bid_price: float, ask_price: float, bid_volume: float, ask_volume: float
    ) -> float:
        """
        Calculate the volume-weighted instantaneous micro-price.
        """
        total_vol = bid_volume + ask_volume
        if total_vol == 0:
            return (bid_price + ask_price) / 2.0

        return ((bid_volume * ask_price) + (ask_volume * bid_price)) / total_vol

    @staticmethod
    def from_imbalance(mid_price: float, spread: float, imbalance: float) -> float:
        """
        Calculate the micro-price linearly from pre-computed imbalance.
        P_micro(t) = P_mid(t) + I_t * (spread / 2)
        """
        return mid_price + imbalance * (spread / 2.0)


class ConvergenceFilter:
    """
    Requires strict consensus across N independent signals before returning True.
    """

    @staticmethod
    def check_agreement(signals: list[bool], threshold: int = 5) -> bool:
        """
        Returns True if at least `threshold` signals are True (e.g., 5/5 STRONG agreement).
        """
        if len(signals) < threshold:
            return False
        return sum(signals) >= threshold


class BrierScoreValidator:
    """
    Validates prediction model calibration using Brier score.
    """

    @staticmethod
    def calculate(predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
        """
        Calculate the Brier score for a set of predictions.

        Args:
            predicted_probs: Array of predicted probabilities [0, 1].
            actual_outcomes: Array of binary actual outcomes {0, 1}.

        Returns:
            Brier score (Mean Squared Error of predictions). Lower is better.
        """
        if len(predicted_probs) == 0 or len(predicted_probs) != len(actual_outcomes):
            raise ValueError("Arrays must be non-empty and of equal length.")

        return float(np.mean((predicted_probs - actual_outcomes) ** 2))

    @staticmethod
    def is_production_grade(score: float, threshold: float = 0.25) -> bool:
        """
        Checks if the calibration score passes the production threshold.
        """
        return score <= threshold
