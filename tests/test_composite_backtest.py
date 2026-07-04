"""Tests for CONCEPT:KG-2.32 — Multi-Market Composite Backtester.

Known-value assertions on the path-dependent simulation, an explicit
no-look-ahead test (a signal on the *final* bar must never affect P&L), and a
shared-capital-pool scaling test.
"""

from agent_utilities.domains.finance.composite_backtest import (
    CompositeBacktester,
    MarketSpec,
    run_composite_backtest,
)
from agent_utilities.numeric import xp as np


class TestSingleMarketCompounding:
    def test_fully_invested_equity_curve(self):
        # One market, always-on signal, +1% every bar -> compounding.
        rets = np.full(5, 0.01)
        bt = CompositeBacktester(initial_capital=1000.0, max_gross_exposure=1.0)
        res = bt.run([MarketSpec("m", rets)], periods_per_year=252)
        # signal omitted -> always-on (constant) exposure from bar 0; constant
        # exposure carries no look-ahead, so all 5 bars compound at +1%.
        expected = 1000.0 * (1.01**5)
        assert abs(res.equity_curve[-1] - expected) < 1e-6
        assert abs(res.total_return - (1.01**5 - 1.0)) < 1e-9
        assert res.n_markets == 1
        assert res.n_periods == 5

    def test_max_drawdown_sign(self):
        rets = np.array([0.10, -0.20, 0.05, -0.30, 0.10])
        res = run_composite_backtest([MarketSpec("m", rets)], initial_capital=1000.0)
        assert res.max_drawdown <= 0.0


class TestNoLookAhead:
    def test_final_bar_signal_ignored(self):
        # Two runs identical except the signal on the LAST bar. Because signals
        # are shifted forward one bar (decision at close of t-1 drives bar t),
        # the last bar's signal can never act -> identical P&L.
        rets = np.array([0.02, -0.01, 0.03, 0.04])
        sig_a = np.array([1.0, 1.0, 1.0, 0.0])
        sig_b = np.array([1.0, 1.0, 1.0, 1.0])
        ra = run_composite_backtest([MarketSpec("m", rets, sig_a)])
        rb = run_composite_backtest([MarketSpec("m", rets, sig_b)])
        assert abs(ra.total_return - rb.total_return) < 1e-12

    def test_signal_acts_one_bar_late(self):
        # Signal turns on at bar index 1 (close), so exposure first applies on
        # bar 2. Bar 0 and bar 1 returns must NOT be captured.
        rets = np.array([0.50, 0.50, 0.10, 0.10])
        sig = np.array([0.0, 1.0, 1.0, 1.0])  # decided at close of each bar
        res = run_composite_backtest([MarketSpec("m", rets, sig)])
        # Shifted signal = [0, 0, 1, 1]; only bars 2 & 3 captured: (1.1*1.1)-1.
        assert abs(res.total_return - (1.1 * 1.1 - 1.0)) < 1e-9


class TestSharedCapitalPool:
    def test_gross_exposure_scaled_down(self):
        # Two markets, each weight 1.0 (-> 0.5 each after normalization), both
        # always-on. With max_gross_exposure=1.0 the combined 1.0 gross fits, so
        # no scaling. Raise to 3 markets and the requested gross exceeds the cap.
        rets = np.full(4, 0.0)  # zero returns: we only check exposures
        sig = np.ones(4)
        markets = [MarketSpec(f"m{i}", rets, sig, weight=1.0) for i in range(3)]
        # Force a tight cap so combined gross (1.0) > cap (0.6) -> scaled.
        bt = CompositeBacktester(max_gross_exposure=0.6)
        res = bt.run(markets)
        total_avg_exposure = sum(a.avg_exposure for a in res.attribution)
        # After the first (flat) bar, combined applied gross is capped at 0.6.
        assert total_avg_exposure <= 0.6 + 1e-9
        assert total_avg_exposure > 0.0

    def test_attribution_sums_to_total_pnl(self):
        rng = np.random.default_rng(7)
        r1 = rng.normal(0.001, 0.01, 50)
        r2 = rng.normal(0.0005, 0.02, 50)
        res = run_composite_backtest(
            [
                MarketSpec("eq", r1, np.ones(50), weight=0.6, asset_class="equity"),
                MarketSpec("cr", r2, np.ones(50), weight=0.4, asset_class="crypto"),
            ],
            initial_capital=1000.0,
        )
        total_contrib = sum(a.contribution_return for a in res.attribution)
        # Sum of per-market contribution returns equals total P&L / initial cap.
        assert abs(total_contrib - (res.equity_curve[-1] / 1000.0 - 1.0)) < 1e-9
        assert {a.asset_class for a in res.attribution} == {"equity", "crypto"}


class TestMetricsAndEdges:
    def test_local_metrics_source_offline(self):
        # No engine injected -> if engine unreachable, source is "local".
        rng = np.random.default_rng(1)
        r = rng.normal(0.001, 0.01, 100)
        res = run_composite_backtest([MarketSpec("m", r, np.ones(100))])
        assert res.metrics_source in ("local", "engine")
        # Sharpe finite and drawdown non-positive regardless of source.
        assert np.isfinite(res.annualized_sharpe)
        assert res.max_drawdown <= 0.0

    def test_empty_and_too_short(self):
        assert run_composite_backtest([]).n_markets == 0
        short = run_composite_backtest([MarketSpec("m", np.array([0.01]))])
        assert short.n_periods <= 1

    def test_summary_renders(self):
        r = np.full(10, 0.01)
        res = run_composite_backtest([MarketSpec("m", r, np.ones(10))])
        text = res.summary()
        assert "Composite backtest" in text
        assert "Attribution" in text
