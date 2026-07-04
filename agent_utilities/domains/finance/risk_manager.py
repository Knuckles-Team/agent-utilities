"""
Risk Management Engine — CONCEPT:AU-KG.research.research-pipeline-runner

Provides a risk-first guard pipeline with VaR calculation, stress testing,
and pre-trade validation integrated with KG provenance.

Sources: AutoHedge Risk Architecture, OpenAlice Guard Pipeline, HFT Lecture Notes
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.numeric import xp as np

logger = logging.getLogger(__name__)

# Lazy, cached epistemic-graph client for Rust-backed VaR/CVaR. Probed once;
# falls back to the local numpy path when the engine is unreachable so that
# offline/unit-test environments behave exactly as before.
_ENGINE_PROBED = False
_ENGINE_CLIENT: Any = None


def _risk_engine() -> Any:
    global _ENGINE_PROBED, _ENGINE_CLIENT
    if _ENGINE_PROBED:
        return _ENGINE_CLIENT
    _ENGINE_PROBED = True
    try:
        from epistemic_graph.client import SyncEpistemicGraphClient

        from agent_utilities.knowledge_graph.core.engine_resolver import (
            client_connect_kwargs,
        )

        # Centralized resolution (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision): honour a remote/sharded/insecure
        # deployment instead of the engine's bare env defaults. No autostart — this
        # path degrades to the local numeric-kernel shim when the engine is unreachable.
        _ENGINE_CLIENT = SyncEpistemicGraphClient.connect(**client_connect_kwargs())
        logger.info("epistemic-graph engine connected for VaR/CVaR")
    except Exception as exc:  # noqa: BLE001 — degrade to local kernel shim
        logger.debug(
            "epistemic-graph engine unavailable for VaR, using local kernel shim: %s",
            exc,
        )
        _ENGINE_CLIENT = None
    return _ENGINE_CLIENT


def _to_list(returns: Any) -> list[float]:
    return (
        returns.tolist() if hasattr(returns, "tolist") else [float(r) for r in returns]
    )


@dataclass
class RiskLimits:
    """Configurable risk limits for a portfolio or strategy."""

    max_position_pct: float = 0.10
    max_drawdown_pct: float = 0.20
    max_sector_concentration: float = 0.30
    max_single_loss: float = 0.02
    max_daily_var_pct: float = 0.05
    max_leverage: float = 1.0


@dataclass
class PreTradeResult:
    """Result of a pre-trade guard check."""

    approved: bool
    violations: list[str] = field(default_factory=list)
    risk_score: float = 0.0


@dataclass
class VaRResult:
    """Value at Risk calculation result."""

    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    method: str = "historical"
    n_observations: int = 0


@dataclass
class StressTestResult:
    """Result of a stress test scenario."""

    scenario_name: str = ""
    pnl_impact: float = 0.0
    portfolio_value_after: float = 0.0
    max_loss_position: str = ""
    positions_breaching_limits: list[str] = field(default_factory=list)


class PreTradeGuard:
    """
    Pre-execution validation guard inspired by OpenAlice's guard pipeline.
    Validates orders against risk limits before they reach the execution layer.
    """

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()

    def validate(
        self,
        _order_side: str,
        order_quantity: float,
        order_price: float,
        portfolio_value: float,
        _current_position_pct: float = 0.0,
        current_drawdown: float = 0.0,
        sector_exposure: float = 0.0,
    ) -> PreTradeResult:
        """Validate a proposed trade against all risk limits."""
        violations = []
        order_value = order_quantity * order_price

        # Position size check
        position_pct = order_value / portfolio_value if portfolio_value > 0 else 1.0
        if position_pct > self.limits.max_position_pct:
            violations.append(
                f"Position size {position_pct:.1%} exceeds limit {self.limits.max_position_pct:.1%}"
            )

        # Drawdown check
        if current_drawdown > self.limits.max_drawdown_pct:
            violations.append(
                f"Current drawdown {current_drawdown:.1%} exceeds limit {self.limits.max_drawdown_pct:.1%}"
            )

        # Sector concentration
        new_sector = sector_exposure + position_pct
        if new_sector > self.limits.max_sector_concentration:
            violations.append(
                f"Sector concentration {new_sector:.1%} would exceed limit {self.limits.max_sector_concentration:.1%}"
            )

        # Max single loss
        potential_loss = position_pct * 0.10  # Assume 10% adverse move
        if potential_loss > self.limits.max_single_loss:
            violations.append(
                f"Potential single loss {potential_loss:.1%} exceeds limit {self.limits.max_single_loss:.1%}"
            )

        risk_score = min(1.0, len(violations) * 0.25 + position_pct)
        return PreTradeResult(
            approved=len(violations) == 0,
            violations=violations,
            risk_score=risk_score,
        )


class VaRCalculator:
    """
    Value at Risk calculator supporting Historical, Parametric, and Monte Carlo methods.
    """

    def historical(
        self,
        returns: np.ndarray,
        confidence_95: float = 0.05,
        confidence_99: float = 0.01,
    ) -> VaRResult:
        """Compute VaR using historical simulation.

        Routes to the Rust epistemic-graph engine when reachable (one batched
        `risk_metrics` round-trip yields var_95/var_99/cvar_95); otherwise uses
        the local numeric-kernel shim path below.
        """
        if len(returns) < 10:
            return VaRResult(method="historical", n_observations=len(returns))

        # Engine path only for the default 95/99 confidences it reports natively.
        if abs(confidence_95 - 0.05) < 1e-9 and abs(confidence_99 - 0.01) < 1e-9:
            client = _risk_engine()
            if client is not None:
                try:
                    m = client.finance.risk_metrics(_to_list(returns), 0.0)
                    return VaRResult(
                        var_95=float(m["var_95"]),
                        var_99=float(m["var_99"]),
                        cvar_95=float(m["cvar_95"]),
                        method="historical",
                        n_observations=len(returns),
                    )
                except Exception as exc:  # noqa: BLE001 — degrade to local kernel shim
                    logger.debug("engine VaR failed, using local kernel shim: %s", exc)

        sorted_returns = np.sort(returns)
        var_95 = -np.percentile(sorted_returns, confidence_95 * 100)
        var_99 = -np.percentile(sorted_returns, confidence_99 * 100)

        # CVaR (Expected Shortfall) — average loss beyond VaR
        tail_95 = sorted_returns[sorted_returns <= -var_95]
        cvar_95 = -np.mean(tail_95) if len(tail_95) > 0 else var_95

        return VaRResult(
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            method="historical",
            n_observations=len(returns),
        )

    def parametric(self, returns: np.ndarray) -> VaRResult:
        """Compute VaR using parametric (normal distribution) assumption."""
        if len(returns) < 10:
            return VaRResult(method="parametric", n_observations=len(returns))

        mu = np.mean(returns)
        sigma = np.std(returns)

        var_95 = -(mu + np.norm_ppf(0.05) * sigma)
        var_99 = -(mu + np.norm_ppf(0.01) * sigma)

        # Analytical CVaR for normal distribution
        cvar_95 = -(mu - sigma * np.norm_pdf(np.norm_ppf(0.05)) / 0.05)

        return VaRResult(
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            method="parametric",
            n_observations=len(returns),
        )

    def monte_carlo(
        self, returns: np.ndarray, n_simulations: int = 10000, seed: int = 42
    ) -> VaRResult:
        """Compute VaR using Monte Carlo simulation."""
        if len(returns) < 10:
            return VaRResult(method="monte_carlo", n_observations=len(returns))

        rng = np.random.default_rng(seed)
        mu = np.mean(returns)
        sigma = np.std(returns)

        simulated = rng.normal(mu, sigma, n_simulations)
        sorted_sim = np.sort(simulated)

        var_95 = -np.percentile(sorted_sim, 5)
        var_99 = -np.percentile(sorted_sim, 1)

        tail = sorted_sim[sorted_sim <= -var_95]
        cvar_95 = -np.mean(tail) if len(tail) > 0 else var_95

        return VaRResult(
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            method="monte_carlo",
            n_observations=len(returns),
        )


class StressTestEngine:
    """
    Scenario-based stress testing engine for portfolio P&L impact analysis.
    """

    PREDEFINED_SCENARIOS: dict[str, dict[str, float]] = {
        "market_crash_2008": {
            "equity": -0.40,
            "bond": 0.10,
            "commodity": -0.25,
            "crypto": -0.60,
        },
        "covid_march_2020": {
            "equity": -0.34,
            "bond": 0.05,
            "commodity": -0.30,
            "crypto": -0.50,
        },
        "rate_hike_shock": {
            "equity": -0.15,
            "bond": -0.10,
            "commodity": 0.05,
            "crypto": -0.20,
        },
        "black_swan_tail": {
            "equity": -0.50,
            "bond": -0.05,
            "commodity": -0.40,
            "crypto": -0.80,
        },
        "bull_market_rally": {
            "equity": 0.20,
            "bond": -0.05,
            "commodity": 0.10,
            "crypto": 0.50,
        },
    }

    def run_scenario(
        self,
        scenario_name: str,
        positions: dict[str, dict[str, Any]],
        portfolio_value: float,
        custom_shocks: dict[str, float] | None = None,
    ) -> StressTestResult:
        """
        Run a stress test scenario against a portfolio.

        Args:
            scenario_name: Name of scenario (predefined or custom).
            positions: Dict of {instrument_id: {"value": float, "asset_class": str}}.
            portfolio_value: Total portfolio value.
            custom_shocks: Optional custom shocks {asset_class: shock_pct}.
        """
        shocks = custom_shocks or self.PREDEFINED_SCENARIOS.get(scenario_name, {})
        if not shocks:
            return StressTestResult(scenario_name=scenario_name)

        total_impact = 0.0
        worst_loss = 0.0
        worst_position = ""
        breaching = []

        for inst_id, pos_info in positions.items():
            value = pos_info.get("value", 0.0)
            asset_class = str(pos_info.get("asset_class", "equity"))
            shock = shocks.get(asset_class, 0.0)
            impact = value * shock
            total_impact += impact

            if impact < worst_loss:
                worst_loss = impact
                worst_position = inst_id

            # Check if loss exceeds 20% of position
            if value > 0 and abs(impact / value) > 0.20:
                breaching.append(inst_id)

        return StressTestResult(
            scenario_name=scenario_name,
            pnl_impact=float(total_impact),
            portfolio_value_after=portfolio_value + total_impact,
            max_loss_position=worst_position,
            positions_breaching_limits=breaching,
        )


class RiskManager:
    """
    Unified risk management facade integrating pre-trade guards, VaR,
    and stress testing with KG provenance tracking.
    """

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self.guard = PreTradeGuard(self.limits)
        self.var_calculator = VaRCalculator()
        self.stress_engine = StressTestEngine()

    def check_order(self, **kwargs) -> PreTradeResult:
        """Validate a proposed order against risk limits."""
        return self.guard.validate(**kwargs)

    def compute_var(self, returns: np.ndarray, method: str = "historical") -> VaRResult:
        """Compute VaR using the specified method."""
        if method == "parametric":
            return self.var_calculator.parametric(returns)
        elif method == "monte_carlo":
            return self.var_calculator.monte_carlo(returns)
        return self.var_calculator.historical(returns)

    def run_stress_test(
        self,
        scenario: str,
        positions: dict[str, dict[str, Any]],
        portfolio_value: float,
    ) -> StressTestResult:
        """Run a stress test scenario."""
        return self.stress_engine.run_scenario(scenario, positions, portfolio_value)
