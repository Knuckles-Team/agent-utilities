#!/usr/bin/env python3
from __future__ import annotations

"""Optimal Execution Engine.

CONCEPT:KG-2.6 — Optimal Execution Engine

Mathematical optimal execution strategies derived from Oxford-Man Institute
HFT lecture notes (Drissi, 2024).  Provides production-grade implementations of:

- **Almgren-Chriss Discrete** (Ch 3): Minimize implementation shortfall.
- **Almgren-Chriss Continuous** (Ch 4): HJB-based smooth execution trajectories.
- **Cartea-Jaimungal** (Ch 5): Running inventory penalty with adverse selection.
- **Optimal Market Making** (Ch 10): Avellaneda-Stoikov reservation price model.
- **Cointegration Pairs Trading** (Ch 12): OU mean-reversion with optimal thresholds.
- **Signal-Adaptive Execution** (Ch 7): Predictive signal incorporation.

All strategies produce ``ExecutionPlan`` objects that integrate with KG-2.6
(Financial Pipeline), KG-2.7 (Risk Scoring), and AHE-3.8 (Backtest Harness).

Glossary of financial terms:
- **Implementation Shortfall**: Difference between decision price and actual
  execution price.  The cost of trading.
- **Market Impact**: Price change caused by trading.  Temporary impact reverses;
  permanent impact persists.
- **Limit Order Book (LOB)**: Electronic book of resting buy/sell orders at
  different price levels.
- **Inventory Risk**: Risk from holding a position whose value may change.
- **Mean Reversion**: Tendency of a price series to revert to its long-term mean.
- **Ornstein-Uhlenbeck (OU) Process**: Continuous-time mean-reverting stochastic
  process: dX_t = θ(μ - X_t)dt + σdW_t.
- **Hamilton-Jacobi-Bellman (HJB)**: PDE for optimal control in continuous time.
- **Riccati ODE**: Nonlinear ODE of the form y' = q₀ + q₁y + q₂y², arises in
  optimal control problems.
"""


import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """An optimal execution plan for a trading strategy.

    Attributes:
        strategy_name: Name of the execution strategy used.
        total_shares: Total shares/units to execute.
        time_horizon: Execution horizon (in time units).
        schedule: List of (time, quantity) tuples — the execution schedule.
        expected_cost: Expected implementation shortfall.
        risk: Variance of execution cost.
        parameters: Strategy-specific parameters used.
        metadata: Additional strategy metadata.
    """

    strategy_name: str = ""
    total_shares: float = 0.0
    time_horizon: float = 1.0
    schedule: list[tuple[float, float]] = field(default_factory=list)
    expected_cost: float = 0.0
    risk: float = 0.0
    parameters: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketMakingQuote:
    """Optimal market-making bid/ask quotes.

    Attributes:
        reservation_price: Fair value adjusted for inventory risk.
        optimal_spread: Optimal bid-ask spread.
        bid_price: Optimal bid price.
        ask_price: Optimal ask price.
        inventory: Current inventory position.
        time_remaining: Time until session end.
    """

    reservation_price: float = 0.0
    optimal_spread: float = 0.0
    bid_price: float = 0.0
    ask_price: float = 0.0
    inventory: float = 0.0
    time_remaining: float = 1.0


@dataclass
class PairsTradeSignal:
    """Signal from cointegration pairs trading analysis.

    Attributes:
        spread_value: Current spread between the pair.
        mean_level: Long-term mean of the spread.
        z_score: Standardized distance from mean.
        signal: 'long', 'short', 'exit', or 'hold'.
        entry_threshold: Z-score threshold for entry.
        exit_threshold: Z-score threshold for exit.
        half_life: Mean-reversion half-life in time units.
    """

    spread_value: float = 0.0
    mean_level: float = 0.0
    z_score: float = 0.0
    signal: str = "hold"
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    half_life: float = 0.0


class AlmgrenChrissDiscrete:
    """Almgren-Chriss optimal execution in discrete time.

    CONCEPT:KG-2.6 — Almgren-Chriss Discrete (Oxford HFT Ch 3)

    Minimizes expected implementation shortfall (execution cost + risk penalty):
        min E[C] + λ·Var[C]
    subject to completing the trade in N time steps.

    The closed-form solution uses hyperbolic functions of a risk parameter κ:
        n_j = X · sinh(κ(T-t_j)) / sinh(κT)

    where X is total shares, T is horizon, and κ depends on risk aversion.
    """

    def compute_schedule(
        self,
        total_shares: float,
        n_steps: int,
        volatility: float,
        temporary_impact: float,
        permanent_impact: float,
        risk_aversion: float = 1e-6,
    ) -> ExecutionPlan:
        """Compute the optimal discrete execution schedule.

        Args:
            total_shares: Total quantity to execute (positive = buy).
            n_steps: Number of discrete time steps.
            volatility: Price volatility σ (annualized std dev).
            temporary_impact: Temporary market impact coefficient η.
            permanent_impact: Permanent market impact coefficient γ.
            risk_aversion: Risk aversion parameter λ (higher = more risk-averse).

        Returns:
            ExecutionPlan with the optimal trading schedule.
        """
        if n_steps <= 0 or total_shares == 0:
            return ExecutionPlan(strategy_name="almgren_chriss_discrete")

        tau = 1.0 / n_steps  # Time per step
        sigma = volatility

        # Risk parameter κ (Oxford HFT Ch 3, Eq. 3.6)
        # κ = arccosh(1 + (τ²σ²λ)/(2η))
        inner = 1.0 + (tau**2 * sigma**2 * risk_aversion) / (2.0 * temporary_impact)
        inner = max(1.0, inner)  # Ensure valid for arccosh
        kappa = math.acosh(inner)

        # Optimal schedule: n_j = X · sinh(κ(N-j)) / sinh(κN)
        schedule: list[tuple[float, float]] = []
        remaining = total_shares

        for j in range(n_steps):
            time_frac = j * tau
            if kappa * n_steps > 500:  # Prevent overflow
                trade_size = total_shares / n_steps
            else:
                denom = math.sinh(kappa * n_steps) if kappa * n_steps > 0 else n_steps
                if denom == 0:
                    trade_size = total_shares / n_steps
                else:
                    trade_size = total_shares * math.sinh(kappa * (n_steps - j)) / denom

            # Ensure we don't overshoot
            trade_size = min(abs(trade_size), abs(remaining))
            trade_size = math.copysign(trade_size, total_shares)
            remaining -= trade_size

            schedule.append((time_frac, trade_size))

        # Handle rounding residual
        if abs(remaining) > 1e-10:
            schedule[-1] = (schedule[-1][0], schedule[-1][1] + remaining)

        # Expected cost = γX² + ηΣnⱼ²
        expected_cost = permanent_impact * total_shares**2 + temporary_impact * sum(
            q**2 for _, q in schedule
        )

        # Risk = σ²τΣx_j² where x_j is remaining inventory at step j
        inventory = total_shares
        risk = 0.0
        for _, qty in schedule:
            risk += sigma**2 * tau * inventory**2
            inventory -= qty

        return ExecutionPlan(
            strategy_name="almgren_chriss_discrete",
            total_shares=total_shares,
            time_horizon=1.0,
            schedule=schedule,
            expected_cost=expected_cost,
            risk=risk,
            parameters={
                "volatility": volatility,
                "temporary_impact": temporary_impact,
                "permanent_impact": permanent_impact,
                "risk_aversion": risk_aversion,
                "kappa": kappa,
                "n_steps": float(n_steps),
            },
        )


class AlmgrenChrissContinuous:
    """Almgren-Chriss optimal execution in continuous time.

    CONCEPT:KG-2.6 — Almgren-Chriss Continuous (Oxford HFT Ch 4)

    Solves the Hamilton-Jacobi-Bellman (HJB) equation for continuous-time
    optimal execution.  The optimal trajectory is:

        x(t) = X · sinh(κ(T-t)) / sinh(κT)

    where κ = √(λσ²/η) and x(t) is remaining inventory at time t.
    """

    def compute_trajectory(
        self,
        total_shares: float,
        time_horizon: float,
        volatility: float,
        temporary_impact: float,
        risk_aversion: float = 1e-6,
        n_points: int = 100,
    ) -> ExecutionPlan:
        """Compute the continuous-time optimal execution trajectory.

        Args:
            total_shares: Total quantity to execute.
            time_horizon: Trading horizon T (e.g., 1.0 for one day).
            volatility: Price volatility σ.
            temporary_impact: Temporary impact coefficient η.
            risk_aversion: Risk aversion λ.
            n_points: Number of discretization points for the trajectory.

        Returns:
            ExecutionPlan with smooth execution trajectory.
        """
        if time_horizon <= 0 or total_shares == 0:
            return ExecutionPlan(strategy_name="almgren_chriss_continuous")

        # κ = √(λσ²/η)
        if temporary_impact > 0:
            kappa = math.sqrt(risk_aversion * volatility**2 / temporary_impact)
        else:
            kappa = 0.0

        dt = time_horizon / n_points
        schedule: list[tuple[float, float]] = []
        prev_inventory = total_shares

        for i in range(n_points):
            t = i * dt
            (i + 1) * dt

            # Remaining inventory at time t
            if kappa * time_horizon > 500:
                inv_t = total_shares * (1.0 - t / time_horizon)
            else:
                sinh_T = (
                    math.sinh(kappa * time_horizon)
                    if kappa * time_horizon > 0
                    else time_horizon
                )
                if sinh_T == 0:
                    inv_t = total_shares * (1.0 - t / time_horizon)
                else:
                    inv_t = (
                        total_shares * math.sinh(kappa * (time_horizon - t)) / sinh_T
                    )

            trade = prev_inventory - inv_t
            schedule.append((t, trade))
            prev_inventory = inv_t

        # Final trade
        if abs(prev_inventory) > 1e-10:
            schedule.append((time_horizon, prev_inventory))

        # Expected cost (analytical formula)
        if kappa > 0 and kappa * time_horizon < 500:
            expected_cost = (
                0.5
                * temporary_impact
                * kappa
                * total_shares**2
                * (math.cosh(kappa * time_horizon) / math.sinh(kappa * time_horizon))
                if math.sinh(kappa * time_horizon) > 0
                else 0.0
            )
        else:
            expected_cost = temporary_impact * total_shares**2 / time_horizon

        return ExecutionPlan(
            strategy_name="almgren_chriss_continuous",
            total_shares=total_shares,
            time_horizon=time_horizon,
            schedule=schedule,
            expected_cost=expected_cost,
            risk=0.5 * risk_aversion * volatility**2 * total_shares**2 * time_horizon,
            parameters={
                "volatility": volatility,
                "temporary_impact": temporary_impact,
                "risk_aversion": risk_aversion,
                "kappa": kappa,
            },
        )


class CarteaJaimungalExecutor:
    """Cartea-Jaimungal optimal execution framework.

    CONCEPT:KG-2.6 — Cartea-Jaimungal (Oxford HFT Ch 5)

    Extends Almgren-Chriss with running inventory penalty and adverse
    selection modeling.  Solves the associated Riccati ODE for the
    optimal trading rate v*(t).

    The HJB equation includes a running penalty φq² on inventory q,
    producing more aggressive early liquidation than Almgren-Chriss.
    """

    def compute_schedule(
        self,
        total_shares: float,
        time_horizon: float,
        volatility: float,
        temporary_impact: float,
        inventory_penalty: float = 0.01,
        terminal_penalty: float = 0.0,
        n_steps: int = 50,
    ) -> ExecutionPlan:
        """Compute the Cartea-Jaimungal optimal execution schedule.

        Args:
            total_shares: Total quantity to execute.
            time_horizon: Trading horizon T.
            volatility: Price volatility σ.
            temporary_impact: Temporary impact η.
            inventory_penalty: Running inventory penalty φ (per time × inventory²).
            terminal_penalty: Terminal penalty α for remaining inventory.
            n_steps: Number of time steps.

        Returns:
            ExecutionPlan with the optimal schedule.
        """
        if time_horizon <= 0 or total_shares == 0:
            return ExecutionPlan(strategy_name="cartea_jaimungal")

        dt = time_horizon / n_steps

        # Solve Riccati ODE backward: h'(t) = -(φ + σ²h²(t)/(2η))
        # with h(T) = terminal_penalty
        # Using Euler method (backward in time)
        h = np.zeros(n_steps + 1)
        h[-1] = terminal_penalty

        for i in range(n_steps - 1, -1, -1):
            dhdt = -(
                inventory_penalty
                + volatility**2 * h[i + 1] ** 2 / (2 * temporary_impact)
            )
            h[i] = h[i + 1] - dhdt * dt

        # Optimal trading rate: v*(t) = h(t)·q(t) / η
        schedule: list[tuple[float, float]] = []
        inventory = total_shares

        for i in range(n_steps):
            t = i * dt
            rate = (
                h[i] * inventory / temporary_impact
                if temporary_impact > 0
                else inventory / n_steps
            )
            trade = rate * dt
            trade = min(abs(trade), abs(inventory))
            trade = math.copysign(trade, total_shares)
            inventory -= trade
            schedule.append((t, trade))

        if abs(inventory) > 1e-10:
            schedule[-1] = (schedule[-1][0], schedule[-1][1] + inventory)

        expected_cost = temporary_impact * sum(q**2 for _, q in schedule) / dt

        return ExecutionPlan(
            strategy_name="cartea_jaimungal",
            total_shares=total_shares,
            time_horizon=time_horizon,
            schedule=schedule,
            expected_cost=expected_cost,
            parameters={
                "volatility": volatility,
                "temporary_impact": temporary_impact,
                "inventory_penalty": inventory_penalty,
                "terminal_penalty": terminal_penalty,
            },
        )


class AvellanedaStoikovMarketMaker:
    """Avellaneda-Stoikov optimal market making.

    CONCEPT:KG-2.6 — Optimal Market Making (Oxford HFT Ch 10)

    Computes optimal bid-ask quotes as functions of inventory and time.
    The reservation price adjusts the fair value for inventory risk:

        r(q, t) = s - q·γ·σ²·(T-t)

    where s is the mid-price, q is inventory, γ is risk aversion,
    σ is volatility, and T-t is time remaining.

    The optimal spread is:
        δ*(t) = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/k)

    where k is the order arrival intensity parameter.
    """

    def compute_quotes(
        self,
        mid_price: float,
        inventory: float,
        volatility: float,
        risk_aversion: float,
        time_remaining: float,
        arrival_intensity: float = 1.0,
    ) -> MarketMakingQuote:
        """Compute optimal bid and ask quotes.

        Args:
            mid_price: Current mid-price of the asset.
            inventory: Current inventory position (positive = long).
            volatility: Price volatility σ.
            risk_aversion: Risk aversion parameter γ.
            time_remaining: Time until end of trading session (T-t).
            arrival_intensity: Order arrival rate parameter k.

        Returns:
            MarketMakingQuote with optimal quotes.
        """
        # Reservation price: r = s - q·γ·σ²·(T-t)
        reservation = (
            mid_price - inventory * risk_aversion * volatility**2 * time_remaining
        )

        # Optimal spread: δ = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/k)
        if risk_aversion > 0 and arrival_intensity > 0:
            spread = risk_aversion * volatility**2 * time_remaining + (
                2.0 / risk_aversion
            ) * math.log(1.0 + risk_aversion / arrival_intensity)
        else:
            spread = (
                2.0 * volatility * math.sqrt(time_remaining)
                if time_remaining > 0
                else 0.01
            )

        spread = max(spread, 1e-6)
        bid = reservation - spread / 2.0
        ask = reservation + spread / 2.0

        return MarketMakingQuote(
            reservation_price=reservation,
            optimal_spread=spread,
            bid_price=bid,
            ask_price=ask,
            inventory=inventory,
            time_remaining=time_remaining,
        )


class CointegrationPairsTrader:
    """Cointegration-based pairs trading with OU mean-reversion.

    CONCEPT:KG-2.6 — Cointegration Pairs Trading (Oxford HFT Ch 12)

    Models the spread between two cointegrated assets as an
    Ornstein-Uhlenbeck (OU) process:
        dX_t = θ(μ - X_t)dt + σdW_t

    where θ is the mean-reversion speed, μ is the long-term mean,
    and σ is the volatility of the spread.

    Trading signals are generated based on z-score thresholds.
    """

    def fit_ou_parameters(
        self,
        spread_series: list[float] | np.ndarray,
        dt: float = 1.0,
    ) -> dict[str, float]:
        """Estimate Ornstein-Uhlenbeck parameters from spread data.

        Uses the discrete AR(1) representation:
            X_{t+1} = (1 - θΔt)X_t + θμΔt + σ√(Δt)ε

        Args:
            spread_series: Time series of spread values.
            dt: Time step between observations.

        Returns:
            Dict with estimated θ (mean_reversion_speed), μ (long_term_mean),
            σ (volatility), and half_life.
        """
        series = np.array(spread_series, dtype=np.float64)
        if len(series) < 10:
            return {"theta": 0.0, "mu": 0.0, "sigma": 0.0, "half_life": float("inf")}

        # AR(1) regression: X_{t+1} = a + b·X_t + ε
        y = series[1:]
        x = series[:-1]
        len(x)

        # OLS regression
        x_mean = x.mean()
        y_mean = y.mean()
        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))

        if ss_xx == 0:
            return {
                "theta": 0.0,
                "mu": float(y_mean),
                "sigma": 0.0,
                "half_life": float("inf"),
            }

        b = ss_xy / ss_xx
        a = y_mean - b * x_mean

        # Residual volatility
        residuals = y - (a + b * x)
        sigma_res = float(np.std(residuals))

        # Convert AR(1) to OU parameters
        # b = e^(-θΔt) ≈ 1 - θΔt for small θΔt
        if b <= 0 or b >= 1:
            # No mean reversion
            return {
                "theta": 0.0,
                "mu": float(np.mean(series)),
                "sigma": sigma_res,
                "half_life": float("inf"),
            }

        theta = -math.log(b) / dt
        mu = a / (1 - b)
        sigma = (
            sigma_res * math.sqrt(2 * theta / (1 - b**2))
            if (1 - b**2) > 0
            else sigma_res
        )

        # Half-life: time to revert halfway to mean
        half_life = math.log(2) / theta if theta > 0 else float("inf")

        return {
            "theta": theta,
            "mu": mu,
            "sigma": sigma,
            "half_life": half_life,
        }

    def generate_signal(
        self,
        current_spread: float,
        ou_params: dict[str, float],
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
    ) -> PairsTradeSignal:
        """Generate a trading signal based on spread z-score.

        Args:
            current_spread: Current value of the spread.
            ou_params: OU parameters from ``fit_ou_parameters()``.
            entry_threshold: Z-score threshold for trade entry.
            exit_threshold: Z-score threshold for trade exit.

        Returns:
            PairsTradeSignal with signal and diagnostics.
        """
        mu = ou_params.get("mu", 0.0)
        sigma = ou_params.get("sigma", 1.0)
        half_life = ou_params.get("half_life", float("inf"))

        # Z-score
        z_score = (current_spread - mu) / sigma if sigma > 0 else 0.0

        # Signal generation
        if abs(z_score) >= entry_threshold:
            signal = "short" if z_score > 0 else "long"  # Mean-revert
        elif abs(z_score) <= exit_threshold:
            signal = "exit"
        else:
            signal = "hold"

        return PairsTradeSignal(
            spread_value=current_spread,
            mean_level=mu,
            z_score=z_score,
            signal=signal,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            half_life=half_life,
        )


class SignalAdaptiveExecutor:
    """Signal-adaptive optimal execution.

    CONCEPT:KG-2.6 — Signal-Adaptive Execution (Oxford HFT Ch 7)

    Incorporates predictive signals (e.g., order flow imbalance, MACD)
    into the execution schedule.  When the signal predicts favorable
    price movement, the executor delays trading; when unfavorable,
    it accelerates.

    The adjustment modifies the Almgren-Chriss schedule by a
    signal-dependent urgency factor.
    """

    def compute_adaptive_schedule(
        self,
        total_shares: float,
        n_steps: int,
        volatility: float,
        temporary_impact: float,
        risk_aversion: float,
        signal_values: list[float],
        signal_weight: float = 0.3,
    ) -> ExecutionPlan:
        """Compute signal-adaptive execution schedule.

        Args:
            total_shares: Total quantity to execute.
            n_steps: Number of time steps.
            volatility: Price volatility σ.
            temporary_impact: Temporary impact η.
            risk_aversion: Risk aversion λ.
            signal_values: Predictive signal per time step.
                Positive = favorable (delay), Negative = unfavorable (accelerate).
            signal_weight: How much the signal adjusts the schedule (0–1).

        Returns:
            ExecutionPlan with signal-adjusted schedule.
        """
        # Get base Almgren-Chriss schedule
        ac = AlmgrenChrissDiscrete()
        base_plan = ac.compute_schedule(
            total_shares,
            n_steps,
            volatility,
            temporary_impact,
            0.0,
            risk_aversion,
        )

        if not base_plan.schedule:
            return base_plan

        # Adjust schedule by signal
        adjusted_schedule: list[tuple[float, float]] = []
        signal_len = len(signal_values)

        for i, (t, qty) in enumerate(base_plan.schedule):
            signal = signal_values[i] if i < signal_len else 0.0

            # Urgency adjustment: accelerate when signal is negative (unfavorable)
            # Decelerate when signal is positive (favorable)
            adjustment = 1.0 - signal_weight * np.tanh(signal)
            adjusted_qty = qty * adjustment
            adjusted_schedule.append((t, adjusted_qty))

        # Normalize to ensure total shares are preserved
        total_adjusted = sum(q for _, q in adjusted_schedule)
        if abs(total_adjusted) > 1e-10:
            scale = total_shares / total_adjusted
            adjusted_schedule = [(t, q * scale) for t, q in adjusted_schedule]

        return ExecutionPlan(
            strategy_name="signal_adaptive",
            total_shares=total_shares,
            time_horizon=1.0,
            schedule=adjusted_schedule,
            expected_cost=base_plan.expected_cost,
            risk=base_plan.risk,
            parameters={
                **base_plan.parameters,
                "signal_weight": signal_weight,
                "signal_mean": float(np.mean(signal_values)) if signal_values else 0.0,
            },
        )
