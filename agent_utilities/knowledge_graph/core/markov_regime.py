"""Markov Chain Market Regime Detection and Forecasting.

CONCEPT:KG-2.6 — Markov Regime Detection

Extends the core ``MarkovTransitionModel`` with financial market regime
detection, multi-step forecasting, trading signal generation, and
walk-forward backtesting capabilities.

Architecture::

    Raw Returns → MarketRegimeDetector → State Labels
                                              │
                                              ▼
                  MarkovRegimeModel ← MarkovTransitionModel (KG-2.6)
                        │                     │
                        ▼                     ▼
                 Regime Forecast        KG Persistence
                        │          (via FinanceEngineMixin)
                        ▼
                Trading Signal / Walk-Forward Backtest

This module does NOT depend on ``yfinance`` or external data providers.
All methods accept raw numpy arrays or pandas Series, keeping the core
dependency-free.  Convenience wrappers with ``yfinance`` live in the
``FinanceEngineMixin`` (engine_finance.py).
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from .formal_reasoning_core import MarkovTransitionModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Asset-Class-Specific Defaults
# ---------------------------------------------------------------------------


class AssetClass(StrEnum):
    """Supported asset classes with tuned default thresholds."""

    EQUITIES = "equities"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITIES = "commodities"
    FIXED_INCOME = "fixed_income"


@dataclass(frozen=True)
class RegimeThresholds:
    """Regime classification thresholds for a specific asset class.

    Attributes:
        bull_threshold: Minimum rolling return to classify as Bull.
        bear_threshold: Maximum rolling return to classify as Bear.
        window: Rolling window size (trading days).
    """

    bull_threshold: float
    bear_threshold: float
    window: int


# Sensible defaults calibrated per asset class.
# Equities use the article's ±2%/20-day as baseline.
ASSET_CLASS_DEFAULTS: dict[AssetClass, RegimeThresholds] = {
    AssetClass.EQUITIES: RegimeThresholds(
        bull_threshold=0.02, bear_threshold=-0.02, window=20
    ),
    AssetClass.CRYPTO: RegimeThresholds(
        bull_threshold=0.05, bear_threshold=-0.05, window=14
    ),
    AssetClass.FOREX: RegimeThresholds(
        bull_threshold=0.005, bear_threshold=-0.005, window=30
    ),
    AssetClass.COMMODITIES: RegimeThresholds(
        bull_threshold=0.015, bear_threshold=-0.015, window=25
    ),
    AssetClass.FIXED_INCOME: RegimeThresholds(
        bull_threshold=0.003, bear_threshold=-0.003, window=40
    ),
}


class RegimeState(StrEnum):
    """Discrete market regime states."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


# ---------------------------------------------------------------------------
# MarketRegimeDetector — State Labeling from Returns
# ---------------------------------------------------------------------------


class MarketRegimeDetector:
    """Detect market regimes from a returns time series.

    Supports two methods of computing rolling returns:
    - ``rolling_sum``: Simple sum of daily returns (article's approach,
      approximates log returns for small values).
    - ``compounding``: Proper compounding ``∏(1+r_i) - 1``, more accurate
      for volatile assets.

    Example::

        detector = MarketRegimeDetector(
            asset_class=AssetClass.EQUITIES
        )
        states = detector.detect(daily_returns)
    """

    def __init__(
        self,
        asset_class: AssetClass = AssetClass.EQUITIES,
        *,
        bull_threshold: float | None = None,
        bear_threshold: float | None = None,
        window: int | None = None,
        method: str = "rolling_sum",
    ) -> None:
        defaults = ASSET_CLASS_DEFAULTS[asset_class]
        self.asset_class = asset_class
        self.bull_threshold = bull_threshold or defaults.bull_threshold
        self.bear_threshold = bear_threshold or defaults.bear_threshold
        self.window = window or defaults.window
        self.method = method

    def detect(self, returns: np.ndarray) -> np.ndarray:
        """Label each period with a regime state.

        Args:
            returns: 1-D array of period returns (e.g., daily log returns).

        Returns:
            Array of ``RegimeState`` string values, same length as input.
            Periods before the window is filled are labeled ``SIDEWAYS``.
        """
        n = len(returns)
        states = np.full(n, RegimeState.SIDEWAYS, dtype=object)

        if n < self.window:
            return states

        if self.method == "compounding":
            rolling = self._compounding_return(returns)
        else:
            rolling = self._rolling_sum(returns)

        for i in range(self.window - 1, n):
            val = rolling[i]
            if val > self.bull_threshold:
                states[i] = RegimeState.BULL
            elif val < self.bear_threshold:
                states[i] = RegimeState.BEAR
            else:
                states[i] = RegimeState.SIDEWAYS

        return states

    def _rolling_sum(self, returns: np.ndarray) -> np.ndarray:
        """Simple rolling sum of returns (article's default)."""
        cum = np.cumsum(returns)
        rolling = np.zeros(len(returns))
        rolling[self.window - 1 :] = cum[self.window - 1 :] - np.concatenate(
            [[0], cum[: -self.window]]
        )
        return rolling

    def _compounding_return(self, returns: np.ndarray) -> np.ndarray:
        """Proper compounding rolling return: ∏(1+r_i) - 1."""
        n = len(returns)
        rolling = np.zeros(n)
        growth = 1.0 + returns
        for i in range(self.window - 1, n):
            rolling[i] = np.prod(growth[i - self.window + 1 : i + 1]) - 1.0
        return rolling


# ---------------------------------------------------------------------------
# MarkovRegimeModel — Full Regime-Switching Markov Chain
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Walk-forward backtest output.

    Attributes:
        signals: Array of trading signals per period.
        returns: Array of strategy returns per period.
        cumulative_return: Final cumulative return.
        n_regime_changes: Number of regime transitions detected.
    """

    signals: np.ndarray
    returns: np.ndarray
    cumulative_return: float
    n_regime_changes: int


class MarkovRegimeModel:
    """Complete Markov Chain model for market regime analysis.

    CONCEPT:KG-2.6 — Markov Regime Forecasting

    Wraps ``MarketRegimeDetector`` and ``MarkovTransitionModel`` to provide
    an end-to-end pipeline from raw returns to trading signals.

    Example::

        model = MarkovRegimeModel(asset_class=AssetClass.EQUITIES)
        model.fit(daily_returns)
        signal = model.generate_signal(RegimeState.BULL)
        forecast = model.forecast(RegimeState.BULL, n_steps=5)
    """

    def __init__(
        self,
        asset_class: AssetClass = AssetClass.EQUITIES,
        *,
        bull_threshold: float | None = None,
        bear_threshold: float | None = None,
        window: int | None = None,
        method: str = "rolling_sum",
    ) -> None:
        self.detector = MarketRegimeDetector(
            asset_class=asset_class,
            bull_threshold=bull_threshold,
            bear_threshold=bear_threshold,
            window=window,
            method=method,
        )
        self.markov = MarkovTransitionModel()
        self._states: np.ndarray | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def regime_states(self) -> np.ndarray | None:
        """The detected regime labels from the last fit."""
        return self._states

    def fit(self, returns: np.ndarray) -> MarkovRegimeModel:
        """Detect regimes and build the Markov transition matrix.

        Args:
            returns: 1-D array of period returns.

        Returns:
            self, for method chaining.
        """
        self._states = self.detector.detect(returns)
        # Feed state labels as a trace into the core Markov model.
        trace = self._states.tolist()
        self.markov.ingest_trace(trace)
        self._fitted = True
        return self

    def forecast(self, current_state: str, n_steps: int = 5) -> dict[str, float]:
        """Forecast regime probabilities n steps into the future.

        Args:
            current_state: Current regime state (e.g., ``RegimeState.BULL``).
            n_steps: Forecast horizon in periods.

        Returns:
            Dict mapping regime states to their n-step probabilities.
        """
        self._check_fitted()
        return self.markov.forecast_from_state(current_state, n_steps)

    def stationary_distribution(self) -> dict[str, float]:
        """Compute the long-run stationary distribution of regimes.

        Returns:
            Dict mapping regime states to their steady-state probabilities.
        """
        self._check_fitted()
        return self.markov.stationary_distribution()

    def generate_signal(self, current_state: str) -> dict[str, float]:
        """Generate a trading signal from current regime probabilities.

        The signal is computed as: ``bull_prob - bear_prob``.
        Range: [-1.0, 1.0]. Positive = bullish bias, negative = bearish.

        Also returns the individual regime probabilities for downstream
        Kelly sizing or risk management.

        Args:
            current_state: Current regime state.

        Returns:
            Dict with keys: signal, bull_prob, bear_prob, sideways_prob.
        """
        self._check_fitted()
        next_states = self.markov.predict_next_states(current_state, k=10)
        probs = dict(next_states)

        bull_prob = probs.get(RegimeState.BULL, 0.0)
        bear_prob = probs.get(RegimeState.BEAR, 0.0)
        sideways_prob = probs.get(RegimeState.SIDEWAYS, 0.0)

        return {
            "signal": bull_prob - bear_prob,
            "bull_prob": bull_prob,
            "bear_prob": bear_prob,
            "sideways_prob": sideways_prob,
        }

    def walk_forward_backtest(
        self,
        returns: np.ndarray,
        lookback: int = 252,
        *,
        signal_shift: int = 1,
    ) -> BacktestResult:
        """Walk-forward backtest with rolling re-estimation.

        CONCEPT:KG-2.6 — Walk-Forward Regime Backtesting

        At each step ``t >= lookback``:
        1. Fit a new ``MarkovRegimeModel`` on ``returns[t-lookback:t]``.
        2. Detect the regime at time ``t-1`` (no future information).
        3. Generate a signal and apply it to ``returns[t]`` (shifted).

        This prevents lookahead bias by never using returns at time ``t``
        in the model that generates the signal for time ``t``.

        Args:
            returns: Full returns array.
            lookback: Rolling window for re-estimation.
            signal_shift: Number of periods to shift signal (default 1 = next day).

        Returns:
            ``BacktestResult`` with signals, strategy returns, etc.
        """
        n = len(returns)
        signals = np.zeros(n)
        strategy_returns = np.zeros(n)
        n_regime_changes = 0
        prev_regime = None

        for t in range(lookback, n):
            window_returns = returns[t - lookback : t]

            # Fit on lookback window only
            temp_model = MarkovRegimeModel(
                asset_class=self.detector.asset_class,
                bull_threshold=self.detector.bull_threshold,
                bear_threshold=self.detector.bear_threshold,
                window=self.detector.window,
                method=self.detector.method,
            )
            temp_model.fit(window_returns)

            # Current regime is the last detected state in the lookback window
            current_regime = (
                temp_model.regime_states[-1]
                if temp_model.regime_states is not None
                else RegimeState.SIDEWAYS
            )

            # Track regime transitions
            if prev_regime is not None and current_regime != prev_regime:
                n_regime_changes += 1
            prev_regime = current_regime

            # Generate signal from current regime
            sig_dict = temp_model.generate_signal(current_regime)
            signals[t] = sig_dict["signal"]

            # Apply signal to NEXT period's return (shift by signal_shift)
            if t + signal_shift < n:
                strategy_returns[t + signal_shift] = (
                    signals[t] * returns[t + signal_shift]
                )

        cumulative = float(np.prod(1.0 + strategy_returns[lookback:]) - 1.0)

        return BacktestResult(
            signals=signals,
            returns=strategy_returns,
            cumulative_return=cumulative,
            n_regime_changes=n_regime_changes,
        )

    def get_transition_matrix_dict(self) -> dict[str, dict[str, float]]:
        """Return the transition matrix as a nested dict for serialization.

        Returns:
            Dict of {from_state: {to_state: probability}}.
        """
        self._check_fitted()
        if self.markov.transition_matrix is None:
            return {}

        result: dict[str, dict[str, float]] = {}
        for i, src in enumerate(self.markov.states):
            result[src] = {}
            for j, dst in enumerate(self.markov.states):
                result[src][dst] = float(self.markov.transition_matrix[i][j])
        return result

    def to_kg_properties(self) -> dict[str, Any]:
        """Serialize the model state for KG persistence.

        Returns:
            Dict suitable for ``RegistryNode.metadata``.
        """
        self._check_fitted()
        trans_dict = self.get_transition_matrix_dict()
        stat_dist = self.stationary_distribution()

        return {
            "asset_class": str(self.detector.asset_class),
            "bull_threshold": self.detector.bull_threshold,
            "bear_threshold": self.detector.bear_threshold,
            "window": self.detector.window,
            "method": self.detector.method,
            "transition_matrix": json.dumps(trans_dict),
            "stationary_distribution": json.dumps(stat_dist),
            "n_states": len(self.markov.states),
            "states": json.dumps(self.markov.states),
            "sample_count": sum(
                c for row in self.markov.transitions.values() for c in row.values()
            ),
        }

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "MarkovRegimeModel is not fitted. Call .fit(returns) first."
            )


# ---------------------------------------------------------------------------
# HiddenMarkovRegimeModel — Gaussian HMM (optional dependency)
# ---------------------------------------------------------------------------


class HiddenMarkovRegimeModel:
    """Gaussian Hidden Markov Model for latent regime detection.

    CONCEPT:KG-2.6 — Hidden Markov Regime Model

    Uses ``hmmlearn.GaussianHMM`` to discover latent regimes from
    returns data using Baum-Welch (EM) estimation with multiple restarts.

    Requires ``hmmlearn`` in the ``[finance]`` extras.

    Example::

        hmm = HiddenMarkovRegimeModel(n_states=3)
        hmm.fit(daily_returns)
        decoded_states = hmm.decode(daily_returns)
    """

    def __init__(
        self,
        n_states: int = 3,
        n_restarts: int = 10,
        n_iter: int = 100,
        covariance_type: str = "full",
    ) -> None:
        self.n_states = n_states
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self._model: Any = None
        self._state_ordering: list[int] | None = None
        self._fitted = False

    @staticmethod
    def _check_hmmlearn() -> Any:
        """Import hmmlearn or raise a clear error."""
        try:
            from hmmlearn.hmm import GaussianHMM

            return GaussianHMM
        except ImportError as e:
            raise ImportError(
                "hmmlearn is required for HiddenMarkovRegimeModel. "
                "Install with: pip install agent-utilities[finance]"
            ) from e

    def fit(self, returns: np.ndarray) -> HiddenMarkovRegimeModel:
        """Fit a Gaussian HMM with multiple random restarts.

        The best model (highest log-likelihood) across restarts is kept.
        States are re-ordered by mean return (ascending: Bear, Sideways, Bull).

        Args:
            returns: 1-D array of period returns.

        Returns:
            self, for method chaining.
        """
        GaussianHMM = self._check_hmmlearn()
        X = returns.reshape(-1, 1)

        best_score = -np.inf
        best_model = None

        for _ in range(self.n_restarts):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=np.random.randint(0, 2**31),
                )
                try:
                    model.fit(X)
                    score = model.score(X)
                    if score > best_score:
                        best_score = score
                        best_model = model
                except Exception:
                    continue  # nosec B112

        if best_model is None:
            raise RuntimeError(
                "HMM fitting failed across all restarts. "
                "Check returns data for NaN/Inf values."
            )

        self._model = best_model

        # Order states by mean return: lowest mean = Bear, highest = Bull
        means = self._model.means_.flatten()
        self._state_ordering = list(np.argsort(means))
        self._fitted = True
        return self

    def decode(self, returns: np.ndarray) -> np.ndarray:
        """Decode the most likely regime sequence using Viterbi.

        Args:
            returns: 1-D array of period returns.

        Returns:
            Array of ``RegimeState`` string values.
        """
        self._check_fitted_state()
        X = returns.reshape(-1, 1)
        _, raw_states = self._model.predict(X, algorithm="viterbi")

        # Map HMM state indices to semantic regime labels
        regime_map = self._build_regime_map()
        return np.array([regime_map[s] for s in raw_states], dtype=object)

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """Compute posterior regime probabilities for each time step.

        Args:
            returns: 1-D array of period returns.

        Returns:
            (n_samples, n_states) array of regime probabilities,
            columns ordered as [Bear, Sideways, Bull].
        """
        self._check_fitted_state()
        X = returns.reshape(-1, 1)
        raw_proba = self._model.predict_proba(X)
        # Reorder columns by mean-return ordering
        return raw_proba[:, self._state_ordering]

    def forecast_signal(self, returns: np.ndarray, lookback: int = 252) -> np.ndarray:
        """Walk-forward HMM signal generation.

        At each step, re-fits the HMM on the lookback window and generates
        a signal = P(Bull) - P(Bear) from the posterior at the last timestep.

        Args:
            returns: Full returns array.
            lookback: Rolling estimation window.

        Returns:
            Signal array, same length as returns.
        """
        n = len(returns)
        signals = np.zeros(n)

        for t in range(lookback, n):
            window = returns[t - lookback : t]
            try:
                temp = HiddenMarkovRegimeModel(
                    n_states=self.n_states,
                    n_restarts=max(3, self.n_restarts // 3),  # Fewer restarts for speed
                    n_iter=self.n_iter,
                    covariance_type=self.covariance_type,
                )
                temp.fit(window)
                proba = temp.predict_proba(window)
                # Signal from last timestep's posterior
                last_proba = proba[-1]
                # Columns are [Bear, Sideways, Bull] after reordering
                signals[t] = last_proba[2] - last_proba[0]  # Bull - Bear
            except Exception:
                signals[t] = 0.0  # Neutral on failure

        return signals

    def _build_regime_map(self) -> dict[int, str]:
        """Map raw HMM state indices to semantic regime labels."""
        if self._state_ordering is None or len(self._state_ordering) == 0:
            return {}

        labels = [RegimeState.BEAR, RegimeState.SIDEWAYS, RegimeState.BULL]
        regime_map: dict[int, str] = {}
        for rank, orig_idx in enumerate(self._state_ordering):
            if rank < len(labels):
                regime_map[orig_idx] = str(labels[rank].value)
            else:
                regime_map[orig_idx] = str(RegimeState.SIDEWAYS.value)
        return regime_map

    def _check_fitted_state(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "HiddenMarkovRegimeModel is not fitted. Call .fit(returns) first."
            )
