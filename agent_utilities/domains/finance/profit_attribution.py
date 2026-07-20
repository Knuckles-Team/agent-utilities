"""
Profit Attribution Engine — CONCEPT:AU-KG.research.research-pipeline-runner

Decomposes P&L into alpha, beta, and residual components with
comprehensive performance analytics.

Source: Qlib Profit Attribution Module
"""

import logging
from dataclasses import dataclass

from agent_utilities.numeric import NDArray
from agent_utilities.numeric import xp as np

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Result of P&L attribution decomposition."""

    total_return: float = 0.0
    alpha_return: float = 0.0
    beta_return: float = 0.0
    residual_return: float = 0.0
    beta_coefficient: float = 0.0
    r_squared: float = 0.0


@dataclass
class PerformanceReport:
    """Comprehensive performance analytics report."""

    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    n_trades: int = 0
    best_day: float = 0.0
    worst_day: float = 0.0


@dataclass
class BenchmarkComparison:
    """Comparison of strategy returns against a benchmark."""

    strategy_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    correlation: float = 0.0


class ProfitAttributor:
    """
    Decomposes P&L into alpha, beta, and residual components
    using regression against a benchmark.
    """

    def attribute(
        self,
        strategy_returns: NDArray,
        benchmark_returns: NDArray,
        risk_free_rate: float = 0.0,
    ) -> AttributionResult:
        """
        Decompose strategy returns into alpha + beta * benchmark + residual.

        Uses OLS regression: R_strategy = alpha + beta * R_benchmark + epsilon
        """
        if len(strategy_returns) < 5 or len(benchmark_returns) < 5:
            return AttributionResult()

        n = min(len(strategy_returns), len(benchmark_returns))
        start = strategy_returns[:n]
        bench = benchmark_returns[:n]

        # OLS regression
        x_mean = np.mean(bench)
        y_mean = np.mean(start)
        ss_xy = np.sum((bench - x_mean) * (start - y_mean))
        ss_xx = np.sum((bench - x_mean) ** 2)

        if ss_xx == 0:
            return AttributionResult(total_return=float(np.sum(start)))

        beta = ss_xy / ss_xx
        alpha = y_mean - beta * x_mean

        # R-squared
        y_pred = alpha + beta * bench
        ss_res = np.sum((start - y_pred) ** 2)
        ss_tot = np.sum((start - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Decomposition
        total = float(np.sum(start))
        beta_component = float(beta * np.sum(bench))
        alpha_component = float(alpha * n)
        residual = total - alpha_component - beta_component

        return AttributionResult(
            total_return=total,
            alpha_return=alpha_component,
            beta_return=beta_component,
            residual_return=residual,
            beta_coefficient=float(beta),
            r_squared=float(max(0.0, r_squared)),
        )


def compute_performance_report(
    returns: NDArray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> PerformanceReport:
    """
    Compute comprehensive performance metrics from a return series.

    Args:
        returns: Array of period returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Trading periods per year (252 for daily, 12 for monthly).
    """
    if len(returns) < 2:
        return PerformanceReport()

    # Basic returns
    total_return = float(np.prod(1 + returns) - 1)
    n_periods = len(returns)
    annualized_return = float((1 + total_return) ** (periods_per_year / n_periods) - 1)
    volatility = float(np.std(returns) * np.sqrt(periods_per_year))

    # Sharpe
    excess = returns - risk_free_rate / periods_per_year
    sharpe = (
        float(np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year))
        if np.std(excess) > 0
        else 0.0
    )

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    downside_std = (
        float(np.std(downside) * np.sqrt(periods_per_year))
        if len(downside) > 0
        else 0.001
    )
    sortino = (
        float((annualized_return - risk_free_rate) / downside_std)
        if downside_std > 0
        else 0.0
    )

    # Max drawdown
    cumulative = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(np.min(drawdowns))

    # Calmar
    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # Win/loss stats
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = float(len(wins) / len(returns)) if len(returns) > 0 else 0.0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    profit_factor = (
        float(np.sum(wins) / abs(np.sum(losses)))
        if np.sum(losses) != 0
        else float("inf")
    )

    return PerformanceReport(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_drawdown,
        volatility=volatility,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        n_trades=len(returns),
        best_day=float(np.max(returns)),
        worst_day=float(np.min(returns)),
    )


def compare_to_benchmark(
    strategy_returns: NDArray,
    benchmark_returns: NDArray,
    periods_per_year: int = 252,
) -> BenchmarkComparison:
    """
    Compare strategy performance against a benchmark.
    """
    if len(strategy_returns) < 5 or len(benchmark_returns) < 5:
        return BenchmarkComparison()

    n = min(len(strategy_returns), len(benchmark_returns))
    start = strategy_returns[:n]
    bench = benchmark_returns[:n]

    start_total = float(np.prod(1 + start) - 1)
    bench_total = float(np.prod(1 + bench) - 1)
    excess = start - bench

    tracking_error = float(np.std(excess) * np.sqrt(periods_per_year))
    info_ratio = (
        float(np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year))
        if np.std(excess) > 0
        else 0.0
    )

    # Beta and alpha via regression
    attributor = ProfitAttributor()
    attr = attributor.attribute(start, bench)

    # Correlation
    correlation = float(np.corrcoef(start, bench)[0, 1])

    return BenchmarkComparison(
        strategy_return=start_total,
        benchmark_return=bench_total,
        excess_return=start_total - bench_total,
        tracking_error=tracking_error,
        information_ratio=info_ratio,
        beta=attr.beta_coefficient,
        alpha=attr.alpha_return,
        correlation=correlation,
    )
