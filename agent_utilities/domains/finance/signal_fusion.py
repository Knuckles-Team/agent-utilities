"""
Signal Fusion — CONCEPT:KG-2.6
Combines disparate signals (technical, fundamental, sentiment, on-chain)
into a unified directional conviction using Bayesian inference.
Inspired by AI-Trader minimal information paradigm.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SignalSource:
    name: str
    weight: float
    historical_accuracy: float = 0.5


class BayesianSignalFusion:
    """
    Fuses signals using Bayesian updates.
    Treats each signal as evidence to update a prior probability of a price move.
    """

    def __init__(self, prior: float = 0.5):
        self.prior = prior
        self.sources: dict[str, SignalSource] = {}

    def register_source(self, name: str, weight: float, accuracy: float = 0.6) -> None:
        """Register a new signal source with its historical accuracy."""
        self.sources[name] = SignalSource(name, weight, accuracy)

    def update(
        self, current_prior: float, signal_direction: int, source_name: str
    ) -> float:
        """
        Bayesian update step.
        signal_direction: 1 (Up), -1 (Down), 0 (Neutral)
        """
        if source_name not in self.sources or signal_direction == 0:
            return current_prior

        source = self.sources[source_name]

        # P(Signal | True Move) = accuracy if they match, (1-accuracy) if they differ
        if signal_direction == 1:
            likelihood_up = source.historical_accuracy
            likelihood_down = 1.0 - source.historical_accuracy
        else:
            likelihood_up = 1.0 - source.historical_accuracy
            likelihood_down = source.historical_accuracy

        # P(Up | Signal) = P(Signal | Up) * P(Up) / P(Signal)
        p_signal = (likelihood_up * current_prior) + (
            likelihood_down * (1.0 - current_prior)
        )

        if p_signal == 0:
            return current_prior

        posterior_up = (likelihood_up * current_prior) / p_signal

        # Apply source weight (partial update for less trusted sources)
        return current_prior + source.weight * (posterior_up - current_prior)

    def fuse(self, signals: dict[str, int]) -> float:
        """
        Fuse multiple signals into a final probability of an upward move.
        signals: Dict mapping source_name -> direction (1, -1, 0)
        """
        posterior = self.prior

        for source_name, direction in signals.items():
            posterior = self.update(posterior, direction, source_name)

        return posterior
