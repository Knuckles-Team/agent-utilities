"""
Profit Attribution Engine — CONCEPT:AU-KG.research.research-pipeline-runner

Decomposes P&L into alpha, beta, and residual components with
comprehensive performance analytics.

Source: Qlib Profit Attribution Module
"""

import logging
import math
from dataclasses import dataclass

from agent_utilities.numeric import NDArray, xp

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
        start = [float(value) for value in strategy_returns[:n]]
        bench = [float(value) for value in benchmark_returns[:n]]

        # OLS regression
        x_mean = float(xp.mean(bench))
        y_mean = float(xp.mean(start))
        ss_xy = sum(
            (x - x_mean) * (y - y_mean) for x, y in zip(bench, start, strict=True)
        )
        ss_xx = sum((x - x_mean) ** 2 for x in bench)

        if ss_xx == 0:
            return AttributionResult(total_return=float(xp.sum(start)))

        beta = ss_xy / ss_xx
        alpha = y_mean - beta * x_mean

        # R-squared
        y_pred = [alpha + beta * value for value in bench]
        ss_res = sum(
            (actual - predicted) ** 2
            for actual, predicted in zip(start, y_pred, strict=True)
        )
        ss_tot = sum((value - y_mean) ** 2 for value in start)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Decomposition
        total = float(xp.sum(start))
        beta_component = float(beta * xp.sum(bench))
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
    values = [float(value) for value in returns]
    total_return = float(math.prod(1.0 + value for value in values) - 1.0)
    n_periods = len(returns)
    annualized_return = float((1 + total_return) ** (periods_per_year / n_periods) - 1)
    volatility = float(xp.std(values) * math.sqrt(periods_per_year))

    # Sharpe
    excess = [value - risk_free_rate / periods_per_year for value in values]
    sharpe = (
        float(xp.mean(excess) / xp.std(excess) * math.sqrt(periods_per_year))
        if xp.std(excess) > 0
        else 0.0
    )

    # Sortino (downside deviation)
    downside = [value for value in values if value < 0]
    downside_std = (
        float(xp.std(downside) * math.sqrt(periods_per_year))
        if len(downside) > 0
        else 0.001
    )
    sortino = (
        float((annualized_return - risk_free_rate) / downside_std)
        if downside_std > 0
        else 0.0
    )

    # Max drawdown
    cumulative: list[float] = []
    running = 1.0
    for value in values:
        running *= 1.0 + value
        cumulative.append(running)
    rolling_max: list[float] = []
    current_max = float("-inf")
    for value in cumulative:
        current_max = max(current_max, value)
        rolling_max.append(current_max)
    drawdowns = [
        (value - high) / high
        for value, high in zip(cumulative, rolling_max, strict=True)
    ]
    max_drawdown = float(min(drawdowns))

    # Calmar
    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # Win/loss stats
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    win_rate = float(len(wins) / len(returns)) if len(returns) > 0 else 0.0
    avg_win = float(xp.mean(wins)) if wins else 0.0
    avg_loss = float(xp.mean(losses)) if losses else 0.0
    profit_factor = (
        float(xp.sum(wins) / abs(xp.sum(losses)))
        if losses and xp.sum(losses) != 0
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
        n_trades=len(values),
        best_day=float(max(values)),
        worst_day=float(min(values)),
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
    start = [float(value) for value in strategy_returns[:n]]
    bench = [float(value) for value in benchmark_returns[:n]]

    start_total = float(math.prod(1.0 + value for value in start) - 1.0)
    bench_total = float(math.prod(1.0 + value for value in bench) - 1.0)
    excess = [
        strategy - benchmark for strategy, benchmark in zip(start, bench, strict=True)
    ]

    tracking_error = float(xp.std(excess) * math.sqrt(periods_per_year))
    info_ratio = (
        float(xp.mean(excess) / xp.std(excess) * math.sqrt(periods_per_year))
        if xp.std(excess) > 0
        else 0.0
    )

    # Beta and alpha via regression
    attributor = ProfitAttributor()
    attr = attributor.attribute(start, bench)

    # Correlation
    start_mean = float(xp.mean(start))
    bench_mean = float(xp.mean(bench))
    covariance = sum(
        (strategy - start_mean) * (benchmark - bench_mean)
        for strategy, benchmark in zip(start, bench, strict=True)
    )
    start_sd = math.sqrt(sum((value - start_mean) ** 2 for value in start))
    bench_sd = math.sqrt(sum((value - bench_mean) ** 2 for value in bench))
    correlation = (
        float(covariance / (start_sd * bench_sd)) if start_sd and bench_sd else 0.0
    )

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
