"""
Portfolio Optimization Suite — CONCEPT:KG-2.6

Provides Mean-Variance (Markowitz), Risk Parity, and Black-Litterman
portfolio optimization with KG-backed allocation tracking.

Sources: Vibe-Trading Quant Toolkit, FinceptTerminal Analytics
"""

import logging
from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.optimize import minimize
except ImportError as e:
    raise ImportError(
        "Finance extra dependencies missing. Please install agent-utilities[finance]"
    ) from e

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of a portfolio optimization."""

    weights: dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    method: str = ""


class MeanVarianceOptimizer:
    """
    Classic Markowitz Mean-Variance Optimization.
    Finds the portfolio on the efficient frontier that maximizes the Sharpe ratio.
    """

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        asset_names: list[str],
        risk_free_rate: float = 0.02,
        max_weight: float = 0.40,
        min_weight: float = 0.0,
    ) -> OptimizationResult:
        """
        Optimize portfolio weights to maximize the Sharpe ratio.

        Args:
            expected_returns: Array of expected returns for each asset.
            cov_matrix: Covariance matrix of asset returns.
            asset_names: List of asset identifiers.
            risk_free_rate: Risk-free rate for Sharpe calculation.
            max_weight: Maximum weight per asset.
            min_weight: Minimum weight per asset.
        """
        n = len(expected_returns)
        if n == 0:
            return OptimizationResult(method="mean_variance")

        # Objective: minimize negative Sharpe ratio
        def neg_sharpe(weights):
            port_return = np.dot(weights, expected_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if port_vol == 0:
                return 0.0
            return -(port_return - risk_free_rate) / port_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(min_weight, max_weight)] * n
        x0 = np.ones(n) / n

        result = minimize(
            neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        weights = result.x
        port_return = float(np.dot(weights, expected_returns))
        port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0

        weight_dict = {
            name: round(float(w), 6)
            for name, w in zip(asset_names, weights, strict=False)
        }

        return OptimizationResult(
            weights=weight_dict,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            method="mean_variance",
        )


class RiskParityOptimizer:
    """
    Risk Parity Optimization — each asset contributes equally to portfolio risk.
    """

    def optimize(
        self,
        cov_matrix: np.ndarray,
        asset_names: list[str],
        expected_returns: np.ndarray | None = None,
    ) -> OptimizationResult:
        """
        Compute risk parity weights such that each asset contributes
        equally to total portfolio volatility.
        """
        n = cov_matrix.shape[0]
        if n == 0:
            return OptimizationResult(method="risk_parity")

        target_risk = 1.0 / n

        def risk_budget_objective(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if port_vol == 0:
                return 0.0
            # Marginal risk contribution
            marginal = np.dot(cov_matrix, weights) / port_vol
            risk_contrib = weights * marginal
            # Minimize deviation from equal risk contribution
            return np.sum((risk_contrib - target_risk * port_vol) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01, 0.50)] * n
        x0 = np.ones(n) / n

        result = minimize(
            risk_budget_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        weights = result.x
        port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
        port_return = (
            float(np.dot(weights, expected_returns))
            if expected_returns is not None
            else 0.0
        )

        weight_dict = {
            name: round(float(w), 6)
            for name, w in zip(asset_names, weights, strict=False)
        }

        return OptimizationResult(
            weights=weight_dict,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=port_return / port_vol if port_vol > 0 else 0.0,
            method="risk_parity",
        )


class BlackLittermanOptimizer:
    """
    Black-Litterman model — blends market equilibrium with investor views
    to produce posterior expected returns for mean-variance optimization.
    """

    def optimize(
        self,
        market_caps: np.ndarray,
        cov_matrix: np.ndarray,
        asset_names: list[str],
        views: list[dict] | None = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.02,
    ) -> OptimizationResult:
        """
        Compute Black-Litterman posterior returns and optimize.

        Args:
            market_caps: Market capitalization weights for each asset.
            cov_matrix: Covariance matrix.
            asset_names: Asset identifiers.
            views: List of view dicts with keys: 'asset_idx', 'return', 'confidence'.
            risk_aversion: Market risk aversion parameter (delta).
            tau: Scaling factor for uncertainty in equilibrium returns.
            risk_free_rate: Risk-free rate.
        """
        n = len(market_caps)
        if n == 0:
            return OptimizationResult(method="black_litterman")

        # Step 1: Implied equilibrium returns (reverse optimization)
        market_weights = market_caps / np.sum(market_caps)
        pi = risk_aversion * np.dot(cov_matrix, market_weights)

        if views and len(views) > 0:
            # Step 2: Build view matrices
            k = len(views)
            P = np.zeros((k, n))
            Q = np.zeros(k)
            omega_diag = np.zeros(k)

            for i, view in enumerate(views):
                idx = view["asset_idx"]
                P[i, idx] = 1.0
                Q[i] = view["return"]
                confidence = view.get("confidence", 0.5)
                omega_diag[i] = (1.0 / confidence - 1.0) * tau * cov_matrix[idx, idx]

            Omega = np.diag(omega_diag)

            # Step 3: Posterior returns
            tau_sigma = tau * cov_matrix
            M1 = np.linalg.inv(
                np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(Omega) @ P
            )
            M2 = np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(Omega) @ Q
            posterior_returns = M1 @ M2
        else:
            posterior_returns = pi

        # Step 4: Optimize with posterior returns
        mvo = MeanVarianceOptimizer()
        return OptimizationResult(
            weights=mvo.optimize(
                posterior_returns, cov_matrix, asset_names, risk_free_rate
            ).weights,
            expected_return=float(
                np.dot(
                    list(
                        mvo.optimize(
                            posterior_returns, cov_matrix, asset_names, risk_free_rate
                        ).weights.values()
                    ),
                    posterior_returns,
                )
            ),
            expected_volatility=mvo.optimize(
                posterior_returns, cov_matrix, asset_names, risk_free_rate
            ).expected_volatility,
            sharpe_ratio=mvo.optimize(
                posterior_returns, cov_matrix, asset_names, risk_free_rate
            ).sharpe_ratio,
            method="black_litterman",
        )


class EmpiricalKellyOptimizer:
    """
    Empirical Kelly Position Sizing — CONCEPT:KG-2.6
    Adjusts standard Kelly criterion sizing based on the Coefficient of Variation (CV)
    of the expected edge, computed via Monte Carlo simulation of historical returns.
    """

    def compute_fraction(
        self,
        win_probability: float,
        win_loss_ratio: float,
        historical_returns: np.ndarray,
        n_simulations: int = 10000,
    ) -> float:
        """
        Compute uncertainty-adjusted Kelly fraction.

        Args:
            win_probability (p): Estimated probability of a winning trade.
            win_loss_ratio (b): Ratio of average win magnitude to average loss magnitude.
            historical_returns: Array of past strategy returns to measure edge stability.
            n_simulations: Number of Monte Carlo paths for edge CV estimation.

        Returns:
            f_empirical: The adjusted Kelly fraction to risk (0.0 to 1.0).
        """
        if len(historical_returns) == 0:
            return 0.0

        p = win_probability
        q = 1.0 - p
        b = win_loss_ratio

        if b <= 0:
            return 0.0

        # Standard Kelly fraction
        f_kelly = (p * b - q) / b

        if f_kelly <= 0:
            return 0.0

        # Monte Carlo to measure edge variation
        # Generate indices for resampling
        n_obs = len(historical_returns)
        sim_indices = np.random.randint(0, n_obs, size=(n_simulations, n_obs))
        edge_estimates = historical_returns[sim_indices].mean(axis=1)

        mean_edge = np.abs(np.mean(edge_estimates))
        if mean_edge == 0:
            return 0.0

        cv_edge = np.std(edge_estimates) / mean_edge

        # Empirical Kelly limits leverage based on variance of the edge
        f_empirical = f_kelly * (1.0 - cv_edge)

        return max(f_empirical, 0.0)


class FractionalKellyOptimizer:
    """
    Fractional Kelly Position Sizing — CONCEPT:KG-2.6
    Scales the Kelly criterion by a fixed fraction (e.g., 0.15x) to survive variance.
    """

    @staticmethod
    def compute_fraction(
        win_probability: float, win_loss_ratio: float, fraction: float = 0.15
    ) -> float:
        """
        Compute Fractionally scaled Kelly criterion.
        """
        p = win_probability
        q = 1.0 - p
        b = win_loss_ratio

        if b <= 0:
            return 0.0

        f_kelly = (p * b - q) / b

        if f_kelly <= 0:
            return 0.0

        return max(f_kelly * fraction, 0.0)


class CircuitBreaker:
    """
    Risk Management Circuit Breaker — CONCEPT:KG-2.6
    Halts trading if a daily drawdown threshold is breached.
    """

    @staticmethod
    def is_tripped(
        current_daily_loss: float, max_daily_loss_limit: float = 300.0
    ) -> bool:
        """
        Check if the daily loss circuit breaker is tripped.

        Args:
            current_daily_loss: The total loss accumulated today (as a positive float).
            max_daily_loss_limit: The maximum allowed loss before halting.

        Returns:
            True if trading should be halted.
        """
        return current_daily_loss >= max_daily_loss_limit
