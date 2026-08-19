"""
Signal Fusion — CONCEPT:AU-KG.research.research-pipeline-runner
Combines disparate signals (technical, fundamental, sentiment, on-chain)
into a unified directional conviction using Bayesian inference.
Inspired by AI-Trader minimal information paradigm.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

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

    def seed_from_kg(
        self,
        signals: Iterable[Any] | None,
        *,
        min_sharpe: float = 0.0,
        max_pbo: float = 0.5,
    ) -> int:
        """Seed fusion sources from stored MicrostructureSignal priors.

        CONCEPT:AU-AHE.assimilation.microstructure-signal-fusion — closes the priors→weights loop. ``signals`` is an
        iterable of records (``MicrostructureSignalNode`` instances or plain
        mappings) carrying ``directional_accuracy``, ``standalone_sharpe``, and
        ``pbo``. Each surviving signal is registered with
        ``weight = directional_accuracy * standalone_sharpe`` (the ``fusionWeight``
        prior) and ``accuracy = directional_accuracy``. Signals that are overfit
        (``pbo > max_pbo``) or carry no edge (``standalone_sharpe <= min_sharpe``)
        are skipped, so the backtester's measured results drive the weights
        automatically on the next cycle. Returns the number registered.
        """

        def _get(rec: object, key: str, default: float) -> float:
            if isinstance(rec, dict):
                val = rec.get(key, default)
            else:
                val = getattr(rec, key, default)
            try:
                return float(val) if val is not None else default
            except (TypeError, ValueError):
                return default

        def _name(rec: object) -> str:
            if isinstance(rec, dict):
                return str(rec.get("name") or rec.get("id") or "")
            return str(getattr(rec, "name", "") or getattr(rec, "id", ""))

        registered = 0
        for rec in signals or []:
            sharpe = _get(rec, "standalone_sharpe", 0.0)
            pbo = _get(rec, "pbo", 0.0)
            acc = _get(rec, "directional_accuracy", 0.5)
            name = _name(rec)
            if not name or pbo > max_pbo or sharpe <= min_sharpe:
                continue
            self.register_source(name, weight=max(0.0, acc * sharpe), accuracy=acc)
            registered += 1
        logger.debug("seed_from_kg registered %d signal sources", registered)
        return registered

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
    11-Step Alpha Combination Engine — CONCEPT:AU-KG.research.research-pipeline-runner
    Combines N signals using an information-theoretic approach to remove shared variance.
    """

    def __init__(self, lookback_d: int = 20):
        self.lookback_d = lookback_d

    def compute_weights(self, returns_matrix: list[list[float]]) -> list[float]:
        """Compute optimal weights for N signals (Grinold–Kahn combination).

        Each row of ``returns_matrix`` is a signal's PnL/return series over T
        periods. We (1) remove the shared cross-sectional component so common
        market beta does not dominate, (2) estimate the de-correlated signal
        covariance (ridge-regularised), and (3) solve w ∝ Σ⁻¹ μ where μ is each
        signal's mean edge — the maximum-Sharpe linear combination. Weights are
        normalised to sum to 1. No equal-weight fallback: degenerate input raises
        rather than silently returning a meaningless uniform vector.
        """
        from agent_utilities.numeric import xp

        n = len(returns_matrix)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        r = [[float(value) for value in row] for row in returns_matrix]
        if any(len(row) != len(r[0]) for row in r) or len(r[0]) < 2:
            raise ValueError(
                "compute_weights requires each signal to have >=2 observations"
            )

        observations = len(r[0])
        mu = [float(value) for value in xp.mean(r, axis=1)]
        # Remove the shared cross-sectional (common-factor) component per period.
        column_means = [float(value) for value in xp.mean(r, axis=0)]
        r_decorr = [
            [value - column_means[column] for column, value in enumerate(row)]
            for row in r
        ]
        row_means = [float(value) for value in xp.mean(r_decorr, axis=1)]
        denominator = max(1, observations - 1)
        centered = [
            [value - row_means[row] for value in values]
            for row, values in enumerate(r_decorr)
        ]
        centered_transpose = [list(column) for column in zip(*centered, strict=True)]
        covariance = xp.matmul(centered, centered_transpose)
        cov = [
            [
                float(value) / denominator + (1e-6 if row == column else 0.0)
                for column, value in enumerate(values)
            ]
            for row, values in enumerate(covariance)
        ]

        try:
            raw = xp.linalg.solve(cov, mu)
        except xp.linalg.LinAlgError:
            inverse = xp.linalg.pinv(cov)
            raw_matrix = xp.matmul(inverse, [[value] for value in mu])
            raw = [float(row[0]) for row in raw_matrix]

        total = float(xp.sum(raw))
        if abs(total) < 1e-12:
            # Mean-zero net edge: L1-normalise magnitudes (still information-
            # driven, never uniform unless the inputs genuinely are).
            l1 = float(xp.sum([abs(value) for value in raw]))
            if l1 < 1e-12:
                raise ValueError(
                    "signals carry no separable edge (zero mean and zero variance)"
                )
            return [value / l1 for value in raw]
        return [value / total for value in raw]


class LaplaceEnsembleFusion:
    """
    Ensemble Probability Pipeline — CONCEPT:AU-KG.research.research-pipeline-runner
    Converts raw ensemble models (like 31 GFS runs or parallel orderbook snapshots)
    into a highly calibrated, bounded probability using Laplace smoothing.
    """

    @staticmethod
    def compute_probability(
        condition_met_count: int, total_members: int, smoothing_factor: int = 1
    ) -> float:
        """
        Computes the smoothed probability: (count + smoothing) / (total + 2 * smoothing)
        Prevents 0% or 100% confidence limits on small sample sizes.
        """
        if total_members <= 0:
            return 0.5

        return (condition_met_count + smoothing_factor) / (
            total_members + 2 * smoothing_factor
        )
