"""
Cross-Market Statistical Arbitrage (CONCEPT:KG-2.6).

Contains cointegration models, Ornstein-Uhlenbeck parameter estimation,
and optimal execution threshold derivation for predicting cross-platform spreads.
"""

import logging

from agent_utilities.numeric import xp as np

try:
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    adfuller = None

logger = logging.getLogger(__name__)


class CointegrationAnalyzer:
    """
    Analyzes whether the spread between two pricing series is stationary.
    """

    @staticmethod
    def is_cointegrated(
        platform_a_prices: np.ndarray,
        platform_b_prices: np.ndarray,
        beta: float = 1.0,
        p_value_threshold: float = 0.05,
    ) -> bool:
        """
        Check if the spread between two platforms is mean-reverting (I(0)).

        Args:
            platform_a_prices: Price series for Platform A.
            platform_b_prices: Price series for Platform B.
            beta: Cointegration coefficient (default 1.0 for identical contracts).
            p_value_threshold: Significance level for the Augmented Dickey-Fuller test.

        Returns:
            True if the spread is stationary.
        """
        if adfuller is None:
            logger.warning("statsmodels is required for CointegrationAnalyzer")
            return False

        if (
            len(platform_a_prices) != len(platform_b_prices)
            or len(platform_a_prices) < 10
        ):
            return False

        spread = platform_a_prices - (beta * platform_b_prices)

        try:
            adf_result = adfuller(spread)
            p_value = adf_result[1]
            return p_value < p_value_threshold
        except Exception as e:
            logger.error(f"ADF test failed: {e}")
            return False


class OrnsteinUhlenbeckModel:
    """
    Models continuous spread dynamics using the OU stochastic process.
    """

    @staticmethod
    def calibrate(spread: np.ndarray, dt: float) -> dict[str, float]:
        """
        Calibrate OU parameters (theta, mu, sigma) using linear regression (MLE).

        S_t = a + b * S_{t-1} + e

        Args:
            spread: Historical spread time series.
            dt: Sampling interval.

        Returns:
            Dictionary with 'theta' (mean reversion rate), 'mu' (long term mean),
            and 'sigma' (historical volatility).
        """
        if len(spread) < 2:
            raise ValueError("Spread array must contain at least 2 points.")

        S_prev = spread[:-1]
        S_curr = spread[1:]

        if np.std(S_prev) == 0:
            return {"theta": 0.0, "mu": float(S_prev[0]), "sigma": 0.0}

        # Linear regression: S_t = a + b * S_{t-1} + e
        # Equivalently: delta S = a + (b-1) * S_{t-1}
        # We can fit S_curr on S_prev
        A = np.vstack([S_prev, np.ones(len(S_prev))]).T
        b_hat, a_hat = np.linalg.lstsq(A, S_curr, rcond=None)[0]

        residuals = S_curr - (a_hat + b_hat * S_prev)
        std_resid = np.std(residuals)

        # Discretization formulas
        # b_hat = exp(-theta * dt)
        if b_hat <= 0 or b_hat >= 1:
            # Cannot extract mean-reverting theta if b_hat >= 1 (divergent) or <= 0 (oscillatory)
            return {"theta": 0.0, "mu": 0.0, "sigma": 0.0}

        theta = -np.log(b_hat) / dt
        mu = a_hat / (1 - b_hat)
        sigma = std_resid / np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta))

        return {"theta": float(theta), "mu": float(mu), "sigma": float(sigma)}

    @staticmethod
    def optimal_thresholds(theta: float, mu: float, c: float) -> tuple[float, float]:
        """
        Calculates simplistic optimal entry thresholds based on OU parameters.
        (A simplified continuous-time optimal stopping boundary).

        Args:
            theta: Rate of mean reversion.
            mu: Long-term mean of the spread.
            c: Linear transaction fees + slippage across both venues.

        Returns:
            Tuple containing (x_open_long, x_open_short).
        """
        if theta <= 0:
            return (0.0, 0.0)

        # Simplified threshold: Entry when expected reversion profit outpaces friction
        # x_open = mu +/- (c + variance_buffer)
        # Using a simplistic variance buffer proportional to 1/theta
        variance_buffer = 1.0 / theta if theta > 0 else 0

        x_open_long = mu - (c + variance_buffer)
        x_open_short = mu + (c + variance_buffer)

        return (x_open_long, x_open_short)


class CostAwareThresholdFilter:
    """
    Cost-Aware Edge Thresholding — CONCEPT:KG-2.6
    Ensures that an arbitrage signal strictly outpaces the cost of doing business
    (taker fees + slippage + model error) before entering a trade.
    """

    @staticmethod
    def passes_threshold(
        model_probability: float, market_price: float, execution_costs: float = 0.08
    ) -> bool:
        """
        Calculates if the absolute edge is strictly greater than or equal to the combined costs.
        Default 0.08 reflects a typical 8% threshold for prediction market taker fees and slippage.
        """
        edge = abs(model_probability - market_price)
        return edge >= execution_costs


class EventArbitrageEngine:
    """
    Dual-Platform Event Arbitrage Scanner — CONCEPT:KG-2.6
    Evaluates identical event probabilities across two distinct venues against
    a single deterministic model baseline (e.g. Polymarket vs Kalshi vs Open-Meteo).
    """

    @staticmethod
    def evaluate_dual_markets(
        model_probability: float,
        market_a_price: float,
        market_b_price: float,
        execution_costs: float = 0.08,
    ) -> dict[str, float]:
        """
        Returns a dictionary mapping the market to its actionable edge (if passing the threshold).
        """
        opportunities = {}

        edge_a = abs(model_probability - market_a_price)
        if edge_a >= execution_costs:
            opportunities["market_a"] = model_probability - market_a_price

        edge_b = abs(model_probability - market_b_price)
        if edge_b >= execution_costs:
            opportunities["market_b"] = model_probability - market_b_price

        return opportunities
