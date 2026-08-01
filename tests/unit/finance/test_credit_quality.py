"""Tests for CONCEPT:AU-KG.domains.dividend-sustainability-credit-fixed — Dividend Sustainability & Credit Quality.

Known-value assertions on the formulas plus a live-path wiring test that the
combined verdict folds into a ``DebateContext.fundamentals_report`` (the field
the Bear/Risk personas actually read in ``DebateEngine``).
"""

import math

from agent_utilities.domains.finance.credit_quality import (
    CreditDividendReport,
    assess_credit_quality,
    assess_dividend_quality,
    attach_to_debate_context,
    emit_credit_dividend_report,
    merton_distance_to_default,
    normal_cdf,
)


class TestNormalCdf:
    def test_known_values(self):
        # Phi(0) = 0.5, Phi(1.96) ~ 0.975, symmetry Phi(-x) = 1 - Phi(x).
        assert abs(normal_cdf(0.0) - 0.5) < 1e-12
        assert abs(normal_cdf(1.96) - 0.9750) < 1e-3
        assert abs(normal_cdf(-1.0) - (1.0 - normal_cdf(1.0))) < 1e-12


class TestDividendQuality:
    def test_safe_dividend(self):
        # Payout 40% (40/100), FCF coverage 150/40 = 3.75x, growing DPS.
        q = assess_dividend_quality(
            "SAFE",
            dividends_paid=40.0,
            net_income=100.0,
            free_cash_flow=150.0,
            dps=1.1,
            prior_dps=1.0,
            dividend_yield=0.03,
        )
        assert q.available is True
        assert abs(q.payout_ratio - 0.40) < 1e-9
        assert abs(q.coverage - 3.75) < 1e-9
        assert q.coverage_basis == "fcf"
        assert abs(q.dividend_growth - 0.10) < 1e-9
        assert q.verdict == "SAFE"
        assert q.yield_trap is False
        assert q.is_red_flag is False

    def test_yield_trap(self):
        # 9% yield, payout 120% (120/100), FCF coverage 0.5x, DPS cut.
        q = assess_dividend_quality(
            "TRAP",
            dividends_paid=120.0,
            net_income=100.0,
            free_cash_flow=60.0,
            dps=0.9,
            prior_dps=1.0,
            dividend_yield=0.09,
        )
        assert q.available is True
        assert abs(q.payout_ratio - 1.20) < 1e-9
        assert abs(q.coverage - 0.5) < 1e-9
        assert q.yield_trap is True
        assert q.verdict == "UNSUSTAINABLE"
        assert q.is_red_flag is True
        assert any("yield-trap" in f for f in q.flags)

    def test_earnings_coverage_fallback(self):
        # No FCF -> EPS/DPS coverage: 2.0 / 1.0 = 2.0x.
        q = assess_dividend_quality(
            "EPS",
            dividends_paid=50.0,
            net_income=120.0,
            eps=2.0,
            dps=1.0,
        )
        assert q.coverage_basis == "earnings"
        assert abs(q.coverage - 2.0) < 1e-9

    def test_unavailable_without_inputs(self):
        q = assess_dividend_quality("NONE")
        assert q.available is False
        assert q.verdict == "UNAVAILABLE"
        assert "UNAVAILABLE" in q.citation()

    def test_dividend_from_losses_flagged(self):
        q = assess_dividend_quality(
            "LOSS", dividends_paid=10.0, net_income=-5.0, free_cash_flow=8.0
        )
        assert q.payout_ratio == float("inf")
        assert any("non-positive net income" in f for f in q.flags)


class TestMertonModel:
    def test_known_distance_to_default(self):
        # V=120, D=100, sigma=0.20, mu=0, T=1.
        # DD = (ln(1.2) - 0.5*0.04) / 0.20 = (0.182322 - 0.02)/0.2 = 0.811608
        dd, pd = merton_distance_to_default(120.0, 100.0, 0.20)
        assert abs(dd - 0.811608) < 1e-4
        assert abs(pd - normal_cdf(-dd)) < 1e-12
        # Sanity: ~21% default prob.
        assert 0.20 < pd < 0.22

    def test_higher_vol_raises_default_prob(self):
        _, pd_low = merton_distance_to_default(120.0, 100.0, 0.20)
        _, pd_high = merton_distance_to_default(120.0, 100.0, 0.60)
        assert pd_high > pd_low

    def test_invalid_inputs_raise(self):
        for bad in [
            (0.0, 100.0, 0.2),
            (120.0, 0.0, 0.2),
            (120.0, 100.0, 0.0),
        ]:
            try:
                merton_distance_to_default(*bad)
                raise AssertionError("expected ValueError")
            except ValueError:
                pass


class TestCreditQuality:
    def test_investment_grade(self):
        # Big equity cushion, strong coverage, low leverage.
        c = assess_credit_quality(
            "IG",
            equity_value=1000.0,
            equity_vol=0.20,
            total_debt=100.0,
            total_assets=1200.0,
            total_equity=1000.0,
            ebit=300.0,
            interest_expense=10.0,
        )
        assert c.available is True
        assert abs(c.interest_coverage - 30.0) < 1e-9
        assert abs(c.debt_to_equity - 0.10) < 1e-9
        assert math.isclose(c.debt_to_assets, 100.0 / 1200.0)
        assert c.verdict == "INVESTMENT_GRADE"
        assert c.is_red_flag is False

    def test_distressed(self):
        # Thin equity, debt > equity, interest barely covered, high default prob.
        c = assess_credit_quality(
            "DISTRESS",
            equity_value=20.0,
            equity_vol=0.80,
            total_debt=100.0,
            total_assets=130.0,
            total_equity=20.0,
            ebit=8.0,
            interest_expense=10.0,
        )
        assert c.available is True
        assert abs(c.interest_coverage - 0.8) < 1e-9  # below 1.0x
        assert c.verdict == "DISTRESSED"
        assert c.is_red_flag is True

    def test_leverage_only_when_no_equity_inputs(self):
        c = assess_credit_quality(
            "LEV",
            total_debt=300.0,
            total_assets=400.0,
            total_equity=100.0,
            ebit=50.0,
            interest_expense=20.0,
        )
        assert c.available is True
        assert c.distance_to_default is None  # no equity value/vol -> skipped
        assert abs(c.debt_to_equity - 3.0) < 1e-9

    def test_unavailable_without_inputs(self):
        c = assess_credit_quality("NONE")
        assert c.available is False
        assert "UNAVAILABLE" in c.citation()


class TestEmitterAndWiring:
    def test_combined_report(self):
        report = emit_credit_dividend_report(
            "ACME",
            {
                "dividends_paid": 120.0,
                "net_income": 100.0,
                "free_cash_flow": 60.0,
                "dps": 0.9,
                "prior_dps": 1.0,
                "dividend_yield": 0.09,
                "equity_value": 20.0,
                "equity_vol": 0.80,
                "total_debt": 100.0,
                "total_assets": 130.0,
                "total_equity": 20.0,
                "ebit": 8.0,
                "interest_expense": 10.0,
            },
        )
        assert isinstance(report, CreditDividendReport)
        assert report.is_red_flag is True  # both legs flag
        cite = report.citation()
        assert "Credit" in cite and "Dividend" in cite

    def test_attach_to_debate_context_live_path(self):
        # LIVE-PATH: the Bear/Risk personas read DebateContext.fundamentals_report;
        # assert our citation is actually folded into that field.
        from agent_utilities.domains.finance.debate_engine import DebateContext

        ctx = DebateContext(
            ticker="ACME",
            asset_class="equity",
            fundamentals_report="P/E 12.",
        )
        report = attach_to_debate_context(
            ctx,
            {
                "dividends_paid": 120.0,
                "net_income": 100.0,
                "free_cash_flow": 60.0,
                "dividend_yield": 0.09,
                "dps": 0.9,
                "prior_dps": 1.0,
                "equity_value": 20.0,
                "equity_vol": 0.80,
                "total_debt": 100.0,
                "total_equity": 20.0,
                "ebit": 8.0,
                "interest_expense": 10.0,
            },
        )
        assert "P/E 12." in ctx.fundamentals_report  # original preserved
        assert "Credit & Dividend Quality" in ctx.fundamentals_report
        assert "Merton DD" in ctx.fundamentals_report
        assert report.ticker == "ACME"
