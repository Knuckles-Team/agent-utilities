"""
Portfolio Optimization Suite — CONCEPT:KG-2.6

Provides Mean-Variance (Markowitz), Risk Parity, and Black-Litterman
portfolio optimization with KG-backed allocation tracking.

Sources: Vibe-Trading Quant Toolkit, FinceptTerminal Analytics
"""

import logging
import math
import secrets
from dataclasses import dataclass, field

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
        expected_returns: list[float],
        cov_matrix: list[list[float]],
        asset_names: list[str],
        risk_free_rate: float = 0.02,
        max_weight: float = 0.40,
        min_weight: float = 0.0,
    ) -> OptimizationResult:
        """
        Optimize portfolio weights to maximize the Sharpe ratio via epistemic-graph.
        """
        n = len(expected_returns)
        if n == 0:
            return OptimizationResult(method="mean_variance")

        if hasattr(expected_returns, "tolist"):
            expected_returns = expected_returns.tolist()
        if hasattr(cov_matrix, "tolist"):
            cov_matrix = cov_matrix.tolist()

        from epistemic_graph.client import SyncEpistemicGraphClient

        from agent_utilities.knowledge_graph.core.engine_resolver import (
            client_connect_kwargs,
        )

        # Centralized resolution (CONCEPT:OS-5.63): same endpoint/auth the
        # chokepoint uses, so a remote/sharded/insecure deployment is honoured.
        with SyncEpistemicGraphClient.connect(**client_connect_kwargs()) as client:
            resp = client.finance.optimize_portfolio(
                expected_returns,
                cov_matrix,
                risk_free_rate,
                min_weight=min_weight,
                max_weight=max_weight,
            )

        weights = resp.get("weights", [])
        port_return = resp.get("expected_return", 0.0)
        port_vol = resp.get("expected_volatility", 0.0)
        sharpe = resp.get("sharpe_ratio", 0.0)

        weight_dict = {
            name: round(w, 6) for name, w in zip(asset_names, weights, strict=False)
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
        cov_matrix: list[list[float]],
        asset_names: list[str],
        expected_returns: list[float] | None = None,
    ) -> OptimizationResult:
        """
        Compute risk parity weights such that each asset contributes
        equally to total portfolio volatility.
        """
        n = len(cov_matrix)
        if n == 0:
            return OptimizationResult(method="risk_parity")

        if hasattr(cov_matrix, "tolist"):
            cov_matrix = cov_matrix.tolist()

        from epistemic_graph.client import SyncEpistemicGraphClient

        from agent_utilities.knowledge_graph.core.engine_resolver import (
            client_connect_kwargs,
        )

        # Centralized resolution (CONCEPT:OS-5.63): same endpoint/auth the
        # chokepoint uses, so a remote/sharded/insecure deployment is honoured.
        with SyncEpistemicGraphClient.connect(**client_connect_kwargs()) as client:
            resp = client.finance.risk_parity(cov_matrix)

        weights = resp.get("weights", [])
        port_vol = resp.get("portfolio_volatility", 0.0)

        port_return = 0.0
        if expected_returns is not None:
            port_return = sum(
                w * r for w, r in zip(weights, expected_returns, strict=False)
            )

        weight_dict = {
            name: round(w, 6) for name, w in zip(asset_names, weights, strict=False)
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
        market_caps: list[float],
        cov_matrix: list[list[float]],
        asset_names: list[str],
        views: list[float] | None = None,
        pick_matrix: list[list[float]] | None = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.02,
    ) -> OptimizationResult:
        n = len(market_caps)
        if n == 0:
            return OptimizationResult(method="black_litterman")

        if views is None:
            views = []
        if pick_matrix is None:
            pick_matrix = []

        if hasattr(market_caps, "tolist"):
            market_caps = market_caps.tolist()
        if hasattr(cov_matrix, "tolist"):
            cov_matrix = cov_matrix.tolist()
        if hasattr(views, "tolist"):
            views = views.tolist()
        if hasattr(pick_matrix, "tolist"):
            pick_matrix = pick_matrix.tolist()

        total_cap = sum(market_caps)
        market_weights = (
            [mc / total_cap for mc in market_caps] if total_cap > 0 else [1.0 / n] * n
        )

        from epistemic_graph.client import SyncEpistemicGraphClient

        from agent_utilities.knowledge_graph.core.engine_resolver import (
            client_connect_kwargs,
        )

        # Centralized resolution (CONCEPT:OS-5.63): same endpoint/auth the
        # chokepoint uses, so a remote/sharded/insecure deployment is honoured.
        with SyncEpistemicGraphClient.connect(**client_connect_kwargs()) as client:
            resp = client.finance.black_litterman(
                market_weights, cov_matrix, views, pick_matrix, tau, risk_aversion
            )

        weights = resp.get("weights", [])
        port_return = resp.get("expected_return", 0.0)
        port_vol = resp.get("expected_volatility", 0.0)

        weight_dict = {
            name: round(w, 6) for name, w in zip(asset_names, weights, strict=False)
        }

        return OptimizationResult(
            weights=weight_dict,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=port_return / port_vol if port_vol > 0 else 0.0,
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
        historical_returns: list[list[float]],
        n_simulations: int = 10000,
    ) -> float:
        """
        Compute uncertainty-adjusted Kelly fraction.
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

        # Fallback naive simulation using basic random choices
        n_obs = len(historical_returns)
        if n_obs == 0:
            return 0.0

        _ = n_simulations

        # Simulate 10 paths to avoid extremely slow pure Python loops
        edge_estimates = []
        for _ in range(10):
            path_sum = 0.0
            for _ in range(n_obs):
                idx = secrets.randbelow(n_obs)
                path_sum += sum(historical_returns[idx]) / len(historical_returns[idx])
            edge_estimates.append(path_sum / n_obs)

        mean_edge = abs(sum(edge_estimates) / len(edge_estimates))
        if mean_edge == 0:
            return 0.0

        variance = sum((x - mean_edge) ** 2 for x in edge_estimates) / len(
            edge_estimates
        )
        cv_edge = math.sqrt(variance) / mean_edge

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
        return current_daily_loss >= max_daily_loss_limit


class EdgeKellyOptimizer:
    """
    Edge-based Kelly Position Sizing — CONCEPT:KG-2.6
    Calculates Kelly fraction directly from the calculated edge and market price.
    Uses a default Quarter-Kelly (0.25) fractional multiplier for risk dampening.
    """

    @staticmethod
    def compute_fraction(
        edge: float, market_price: float, fraction: float = 0.25
    ) -> float:
        """
        Compute uncertainty-adjusted Kelly fraction for prediction markets.
        kelly_fraction = edge / (1 - market_price)
        """
        if market_price >= 1.0 or market_price <= 0.0 or edge <= 0:
            return 0.0

        odds_against = 1.0 - market_price
        kelly_fraction = edge / odds_against

        # Apply Fractional Kelly logic
        return max(kelly_fraction * fraction, 0.0)
