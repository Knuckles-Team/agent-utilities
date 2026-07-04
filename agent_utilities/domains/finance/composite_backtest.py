"""
CONCEPT:AU-KG.domains.multi-market-composite-backtester — Multi-Market Composite Backtester

Extends the single-market ``SimpleBacktester`` (``research_autopilot.py``, which
runs one signal stream against one return series) to backtest a portfolio across
**multiple markets / asset classes simultaneously** out of a **shared capital
pool**, with per-market strategy weights and a global capital constraint.

Design (path-dependent, no look-ahead):

* Each market supplies aligned, *same-length* arrays of period returns and a
  per-period target exposure signal in ``[0, 1]`` (or [-1, 1] for shorts). The
  simulation walks bars **left to right**: the exposure applied to bar *t*'s
  return is the signal decided at the **close of bar t-1** (signals are shifted
  by one bar internally), so no future information leaks into the current bar.
* Capital is **one shared pool**. At each bar the requested per-market dollar
  allocations (``weight_t * signal_{t-1} * equity_t``) are scaled down
  proportionally if their sum exceeds the global ``max_gross_exposure`` of
  current equity — markets compete for the same capital exactly as in a real
  combined book.
* Outputs: a combined equity curve, per-market P&L attribution, and aggregate
  metrics (annualized return, annualized Sharpe, max drawdown) computed from the
  realized **combined** return path — never from placeholder figures.

The epistemic-graph engine is used for the heavy aggregate stats
(``client.finance.risk_metrics`` for Sharpe/vol and ``deflated_sharpe`` for an
overfit-aware Sharpe) when reachable; a numerically-identical local NumPy path
runs offline so unit tests and air-gapped runs behave the same.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.numeric import xp as np

logger = logging.getLogger(__name__)

_ENGINE_PROBED = False
_ENGINE_CLIENT: Any = None


def _composite_engine() -> Any:
    """Return a connected SyncEpistemicGraphClient, or ``None`` if unavailable."""
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
        # path degrades to the local numpy kernel when the engine is unreachable.
        _ENGINE_CLIENT = SyncEpistemicGraphClient.connect(**client_connect_kwargs())
        logger.info("epistemic-graph engine connected for composite backtest")
    except Exception as exc:  # noqa: BLE001 — degrade to numpy
        logger.debug(
            "epistemic-graph engine unavailable for composite backtest: %s", exc
        )
        _ENGINE_CLIENT = None
    return _ENGINE_CLIENT


def reset_engine_cache() -> None:
    """Reset the cached engine probe (used by tests to re-probe)."""
    global _ENGINE_PROBED, _ENGINE_CLIENT
    _ENGINE_PROBED = False
    _ENGINE_CLIENT = None


@dataclass
class MarketSpec:
    """One market / asset-class leg of a composite backtest.

    Attributes:
        name: Market identifier (e.g. "us_equity", "btc", "ust_10y").
        returns: Per-period simple returns (decimal), length == n_periods.
        signals: Per-period target exposure in [-1, 1] decided at that bar's
            *close*. Internally shifted by one bar so bar t uses signal t-1
            (no look-ahead). Defaults to fully-invested (1.0) when omitted.
        weight: Static capital weight (relative budget share) for this market.
        asset_class: Free-form class label for attribution grouping.
    """

    name: str
    returns: np.ndarray
    signals: np.ndarray | None = None
    weight: float = 1.0
    asset_class: str = "equity"


@dataclass
class MarketAttribution:
    """Per-market contribution to the combined book."""

    name: str
    asset_class: str
    pnl: float = 0.0  # absolute realized P&L (currency units)
    contribution_return: float = 0.0  # pnl / initial_capital
    avg_exposure: float = 0.0  # mean applied gross exposure fraction
    realized_return: float = 0.0  # cumulative return of the leg's own path


@dataclass
class CompositeBacktestResult:
    """Outputs of a multi-market composite backtest."""

    equity_curve: list[float] = field(default_factory=list)
    combined_returns: list[float] = field(default_factory=list)
    attribution: list[MarketAttribution] = field(default_factory=list)
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_sharpe: float = 0.0
    max_drawdown: float = 0.0
    deflated_sharpe: float | None = None  # overfit-aware (engine), when available
    n_periods: int = 0
    n_markets: int = 0
    metrics_source: str = "local"  # "engine" | "local"

    def summary(self) -> str:
        lines = [
            f"Composite backtest: {self.n_markets} markets x {self.n_periods} periods",
            f"  Total return:      {self.total_return:.2%}",
            f"  Annualized return: {self.annualized_return:.2%}",
            f"  Annualized Sharpe: {self.annualized_sharpe:.2f}",
            f"  Max drawdown:      {self.max_drawdown:.2%}",
        ]
        if self.deflated_sharpe is not None:
            lines.append(f"  Deflated Sharpe:   {self.deflated_sharpe:.2f}")
        lines.append(f"  Metrics source:    {self.metrics_source}")
        lines.append("  Attribution:")
        for a in sorted(self.attribution, key=lambda x: x.pnl, reverse=True):
            lines.append(
                f"    {a.name} ({a.asset_class}): "
                f"contrib={a.contribution_return:+.2%}, "
                f"avg_exposure={a.avg_exposure:.2f}, "
                f"leg_return={a.realized_return:+.2%}"
            )
        return "\n".join(lines)


def _shift_signals(signals: np.ndarray, n: int) -> np.ndarray:
    """Shift a signal array forward one bar (decision at close of t-1 applies to
    bar t); bar 0 starts flat. This is the no-look-ahead guarantee."""
    shifted = np.zeros(n, dtype=float)
    m = min(len(signals), n)
    if m > 1:
        shifted[1:m] = np.asarray(signals[: m - 1], dtype=float)
    return shifted


class CompositeBacktester:
    """Path-dependent backtester over many markets sharing one capital pool.

    Usage::

        bt = CompositeBacktester(initial_capital=1_000_000, max_gross_exposure=1.0)
        result = bt.run([
            MarketSpec("us_equity", eq_rets, eq_sig, weight=0.6, asset_class="equity"),
            MarketSpec("crypto",    btc_rets, btc_sig, weight=0.4, asset_class="crypto"),
        ], periods_per_year=252)
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        max_gross_exposure: float = 1.0,
        engine_client: Any | None = None,
    ):
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be positive")
        self.initial_capital = float(initial_capital)
        self.max_gross_exposure = float(max_gross_exposure)
        self._explicit_client = engine_client

    def _client(self) -> Any:
        if self._explicit_client is not None:
            return self._explicit_client
        return _composite_engine()

    def run(
        self,
        markets: list[MarketSpec],
        *,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ) -> CompositeBacktestResult:
        """Run the composite backtest.

        Args:
            markets: One ``MarketSpec`` per market/leg. Return arrays should be
                aligned; the run truncates to the shortest series.
            periods_per_year: Annualization factor (252 daily, 12 monthly, ...).
            risk_free_rate: Per-period risk-free rate for Sharpe.

        Returns:
            A ``CompositeBacktestResult`` with equity curve, per-market
            attribution, and aggregate metrics.
        """
        markets = [m for m in markets if m.returns is not None and len(m.returns) > 0]
        if not markets:
            return CompositeBacktestResult()

        n = min(len(m.returns) for m in markets)
        if n < 2:
            return CompositeBacktestResult(n_periods=n, n_markets=len(markets))

        # Normalize static weights to sum to 1 (relative budget shares).
        raw_weights = np.array([max(0.0, m.weight) for m in markets], dtype=float)
        wsum = raw_weights.sum()
        weights = (
            raw_weights / wsum
            if wsum > 0
            else np.full(len(markets), 1.0 / len(markets))
        )

        rets = np.array([np.asarray(m.returns[:n], dtype=float) for m in markets])
        # Shift each leg's signal forward one bar -> no look-ahead.
        sigs = np.array(
            [
                _shift_signals(m.signals, n)
                if m.signals is not None
                else np.ones(n, dtype=float)
                for m in markets
            ]
        )

        equity = self.initial_capital
        equity_curve = [equity]
        combined_returns: list[float] = []
        pnl_by_market = np.zeros(len(markets))
        exposure_sum = np.zeros(len(markets))
        leg_cum = np.ones(len(markets))  # compounding per-leg gross return path

        for t in range(n):
            # Requested gross allocation fraction per market for THIS bar, using
            # the signal decided at t-1 (already shifted) and the static weight.
            requested = weights * sigs[:, t]  # fraction of equity, signed
            gross = float(np.abs(requested).sum())

            # Shared-pool constraint: scale down if combined gross exceeds cap.
            scale = 1.0
            if gross > self.max_gross_exposure and gross > 0:
                scale = self.max_gross_exposure / gross
            applied = requested * scale  # fraction of current equity per market

            # Dollar allocation -> P&L from this bar's realized return.
            alloc_dollars = applied * equity
            bar_pnl_by_market = alloc_dollars * rets[:, t]
            bar_pnl = float(bar_pnl_by_market.sum())

            pnl_by_market += bar_pnl_by_market
            exposure_sum += np.abs(applied)
            # Each leg's own compounded path: the bar's strategy return for the
            # leg is its applied exposure times the market return.
            leg_cum *= 1.0 + applied * rets[:, t]

            combined_ret = bar_pnl / equity if equity != 0 else 0.0
            combined_returns.append(combined_ret)
            equity += bar_pnl
            equity_curve.append(equity)

        combined = np.array(combined_returns)
        total_return = equity / self.initial_capital - 1.0

        ann_return, ann_sharpe, max_dd, dsr, source = self._aggregate_metrics(
            combined, periods_per_year, risk_free_rate
        )

        attribution = [
            MarketAttribution(
                name=m.name,
                asset_class=m.asset_class,
                pnl=float(pnl_by_market[i]),
                contribution_return=float(pnl_by_market[i] / self.initial_capital),
                avg_exposure=float(exposure_sum[i] / n),
                realized_return=float(leg_cum[i] - 1.0),
            )
            for i, m in enumerate(markets)
        ]

        return CompositeBacktestResult(
            equity_curve=equity_curve,
            combined_returns=combined_returns,
            attribution=attribution,
            total_return=float(total_return),
            annualized_return=float(ann_return),
            annualized_sharpe=float(ann_sharpe),
            max_drawdown=float(max_dd),
            deflated_sharpe=dsr,
            n_periods=n,
            n_markets=len(markets),
            metrics_source=source,
        )

    def _aggregate_metrics(
        self,
        combined: np.ndarray,
        periods_per_year: int,
        risk_free_rate: float,
    ) -> tuple[float, float, float, float | None, str]:
        """Annualized return, annualized Sharpe, max drawdown, deflated Sharpe.

        Routes Sharpe/vol through the engine's ``risk_metrics`` (one round-trip)
        when reachable; otherwise computes locally with identical formulas.
        """
        n = len(combined)
        if n == 0:
            return 0.0, 0.0, 0.0, None, "local"

        # Annualized geometric return from the realized combined path.
        growth = float(np.prod(1.0 + combined))
        ann_return = growth ** (periods_per_year / n) - 1.0 if growth > 0 else -1.0

        # Max drawdown on the compounded equity path.
        equity_path = np.cumprod(1.0 + combined)
        running_max = np.maximum.accumulate(equity_path)
        drawdowns = equity_path / running_max - 1.0
        max_dd = float(np.min(drawdowns)) if len(drawdowns) else 0.0

        source = "local"
        ann_sharpe = 0.0
        client = self._client()
        if client is not None:
            try:
                m = client.finance.risk_metrics(combined.tolist(), risk_free_rate)
                # Engine reports a (periodic) Sharpe; annualize consistently.
                periodic_sharpe = float(m.get("sharpe", m.get("sharpe_ratio", 0.0)))
                ann_sharpe = periodic_sharpe * np.sqrt(periods_per_year)
                source = "engine"
            except Exception as exc:  # noqa: BLE001 — degrade to numpy
                logger.debug("engine risk_metrics failed, using numpy: %s", exc)

        if source == "local":
            excess = combined - risk_free_rate
            sd = float(np.std(excess))
            ann_sharpe = (
                float(np.mean(excess) / sd * np.sqrt(periods_per_year))
                if sd > 0
                else 0.0
            )

        # Overfit-aware deflated Sharpe (engine only; left None offline).
        dsr: float | None = None
        if client is not None:
            try:
                dsr = float(
                    client.finance.deflated_sharpe(ann_sharpe, 1, combined.tolist())
                )
            except Exception as exc:  # noqa: BLE001 — optional enrichment
                logger.debug("engine deflated_sharpe unavailable: %s", exc)

        return ann_return, ann_sharpe, max_dd, dsr, source


def run_composite_backtest(
    markets: list[MarketSpec],
    *,
    initial_capital: float = 1_000_000.0,
    max_gross_exposure: float = 1.0,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    engine_client: Any | None = None,
) -> CompositeBacktestResult:
    """Convenience one-shot wrapper around ``CompositeBacktester.run``."""
    return CompositeBacktester(
        initial_capital=initial_capital,
        max_gross_exposure=max_gross_exposure,
        engine_client=engine_client,
    ).run(markets, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate)
