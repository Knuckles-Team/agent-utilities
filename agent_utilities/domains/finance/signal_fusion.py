"""
Signal Fusion — CONCEPT:KG-2.6
Combines disparate signals (technical, fundamental, sentiment, on-chain)
into a unified directional conviction using Bayesian inference.
Inspired by AI-Trader minimal information paradigm.
"""

import logging
from dataclasses import dataclass

import numpy as np

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


class AlphaCombinationEngine:
    """
    11-Step Alpha Combination Engine — CONCEPT:KG-2.6
    Combines N signals using an information-theoretic approach to remove shared variance,
    resulting in true independent edge weighting.
    """

    def __init__(self, lookback_d: int = 20):
        self.lookback_d = lookback_d

    def compute_weights(self, returns_matrix: np.ndarray) -> np.ndarray:
        """
        Computes optimal weights for N signals using serial and cross-sectional demeaning,
        and regression for independent edge extraction.

        Args:
            returns_matrix: Shape (N, M) where N is number of signals, M is number of periods.

        Returns:
            np.ndarray of shape (N,) containing optimal absolute-normalized weights.
        """
        import numpy as np
        from sklearn.linear_model import LinearRegression

        N, M = returns_matrix.shape
        if M < self.lookback_d + 2:
            raise ValueError(
                f"Not enough periods in returns_matrix. Need at least {self.lookback_d + 2}."
            )

        # Step 2: Serial demeaning
        X = returns_matrix - returns_matrix.mean(axis=1, keepdims=True)

        # Step 3: Sample variance
        sigma = np.sqrt((X**2).mean(axis=1))
        # Handle zero variance signals
        sigma[sigma == 0] = 1e-9

        # Step 4: Normalize
        Y = X / sigma[:, None]

        # Step 5: Drop most recent observation for lambda computation
        Y_prior = Y[:, :-1]

        # Step 6: Cross-sectional demeaning
        Lambda = Y_prior - Y_prior.mean(axis=0, keepdims=True)

        # Step 7: Drop one more period
        Lambda = Lambda[:, :-1]

        # Get recent window for forward expected returns
        recent = returns_matrix[:, -self.lookback_d :]

        # Step 8: Expected forward return
        E = recent.mean(axis=1) / sigma

        # Step 9: Regress E_normalized on Lambda to find independent contribution (residuals)
        reg = LinearRegression(fit_intercept=False)
        # Lambda is (N, M-2). E is (N,). Fit cross-sectionally across signals.
        reg.fit(Lambda, E)
        residuals = E - reg.predict(Lambda)

        # Step 10: Set weight
        w = residuals / sigma

        # Step 11: Normalize so sum of absolute weights equals 1
        sum_abs_w = np.abs(w).sum()
        if sum_abs_w > 0:
            w = w / sum_abs_w

        return w
