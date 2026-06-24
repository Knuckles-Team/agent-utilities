"""
CONCEPT:KG-2.31 — Dividend Sustainability & Credit/Fixed-Income Quality

Complements the engine-grounded forensic screener (CONCEPT:KG-2.6 /
``forensic_screener.py``, which scores earnings *manipulation* but says nothing
about whether a payout is safe or whether the balance sheet is solvent) with two
analyst-grade verdicts the Bear/Risk persona can cite verbatim in the debate:

* **Dividend sustainability** — payout ratio (dividends / net income), dividend
  coverage (FCF / dividends, falling back to EPS / DPS), trailing dividend
  growth, and a *yield-trap* flag (high stated yield + payout > 1 + deteriorating
  coverage) that catches the classic value-trap "the dividend you see is the
  dividend they cannot afford".
* **Credit quality** — a Merton structural distance-to-default

      DD = (ln(V / D) + (mu - 0.5 * sigma**2) * T) / (sigma * sqrt(T))
      PD = Phi(-DD)

  on the firm's asset value/vol + face debt, plus interest coverage
  (EBIT / interest) and leverage (debt / equity, debt / assets). Φ (the standard
  normal CDF) is computed locally via ``math.erf`` so the module is correct
  fully offline; the epistemic-graph engine is used **only** as an optional
  cross-check for the cumulative-normal default probability when reachable.

Both verdicts are *deterministic formulas over supplied fundamentals* — nothing
is fabricated. Missing inputs yield ``available=False`` rather than a guessed
number, mirroring ``ForensicVerdict``.

Wiring: ``emit_credit_dividend_report`` produces a citable block and, via
``attach_to_debate_context``, folds it into a ``DebateContext.fundamentals_report``
so the Bear/Burry and Risk-Manager personas argue against the *actual* solvency
and payout numbers on the live debate path (``DebateEngine._generate_bear_argument``
and ``_evaluate_risk`` both read ``fundamentals_report``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Lazy, cached epistemic-graph client — probed once, ``None`` when unreachable so
# importing this module never requires a running engine. Used purely as an
# optional cross-check; every verdict is computable locally.
_ENGINE_PROBED = False
_ENGINE_CLIENT: Any = None


def _credit_engine() -> Any:
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

        # Centralized resolution (CONCEPT:OS-5.63): honour a remote/sharded/insecure
        # deployment instead of the engine's bare env defaults. No autostart — this
        # path degrades to the local numpy kernel when the engine is unreachable.
        _ENGINE_CLIENT = SyncEpistemicGraphClient.connect(**client_connect_kwargs())
        logger.info("epistemic-graph engine connected for credit quality")
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, never invent
        logger.debug("epistemic-graph engine unavailable for credit quality: %s", exc)
        _ENGINE_CLIENT = None
    return _ENGINE_CLIENT


def reset_engine_cache() -> None:
    """Reset the cached engine probe (used by tests to re-probe)."""
    global _ENGINE_PROBED, _ENGINE_CLIENT
    _ENGINE_PROBED = False
    _ENGINE_CLIENT = None


def normal_cdf(x: float) -> float:
    """Standard normal CDF Phi(x) via ``math.erf`` (exact, dependency-free).

    Phi(x) = 0.5 * (1 + erf(x / sqrt(2))).
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ──────────────────────────────────────────────────────────────────────────
# Dividend sustainability
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class DividendQuality:
    """Structured dividend-sustainability verdict.

    ``available`` is ``False`` when the required inputs are missing (no
    fabricated figures). ``verdict`` is one of SAFE / STRETCHED / UNSUSTAINABLE /
    UNAVAILABLE.
    """

    ticker: str
    available: bool
    verdict: str = "UNAVAILABLE"
    payout_ratio: float | None = None  # dividends / net income
    coverage: float | None = None  # FCF / dividends (or EPS / DPS)
    coverage_basis: str = ""  # "fcf" | "earnings" | ""
    dividend_growth: float | None = None  # trailing per-share DPS growth
    yield_pct: float | None = None  # stated annual yield (decimal)
    yield_trap: bool = False
    flags: list[str] = field(default_factory=list)

    @property
    def is_red_flag(self) -> bool:
        return self.available and (
            self.yield_trap or self.verdict.upper() == "UNSUSTAINABLE"
        )

    def citation(self) -> str:
        if not self.available:
            return (
                f"Dividend sustainability for {self.ticker} UNAVAILABLE "
                "(insufficient inputs — no numbers fabricated)."
            )

        def _f(v: float | None, pct: bool = False) -> str:
            if not isinstance(v, int | float):
                return "n/a"
            return f"{v:.1%}" if pct else f"{v:.2f}"

        trap = " ⚠ YIELD-TRAP" if self.yield_trap else ""
        flag_str = ("; flags: " + ", ".join(self.flags)) if self.flags else ""
        return (
            f"Dividend [{self.verdict}]{trap} for {self.ticker}: "
            f"payout={_f(self.payout_ratio, pct=True)}, "
            f"coverage={_f(self.coverage)}x ({self.coverage_basis or 'n/a'}), "
            f"DPS growth={_f(self.dividend_growth, pct=True)}, "
            f"yield={_f(self.yield_pct, pct=True)}{flag_str}."
        )


def assess_dividend_quality(
    ticker: str,
    *,
    dividends_paid: float | None = None,
    net_income: float | None = None,
    free_cash_flow: float | None = None,
    eps: float | None = None,
    dps: float | None = None,
    prior_dps: float | None = None,
    dividend_yield: float | None = None,
    high_yield_threshold: float = 0.06,
) -> DividendQuality:
    """Assess dividend sustainability from supplied fundamentals.

    Args:
        dividends_paid: Total cash dividends paid (absolute, positive).
        net_income: Net income for the same period (payout-ratio denominator).
        free_cash_flow: Free cash flow (preferred coverage numerator).
        eps: Earnings per share (coverage fallback numerator).
        dps: Dividend per share (current).
        prior_dps: Dividend per share a year ago (for growth).
        dividend_yield: Stated annual dividend yield as a decimal (e.g. 0.08).
        high_yield_threshold: Yield above which the trap test arms (default 6%).

    Returns:
        A ``DividendQuality``. ``available`` is ``False`` when neither a payout
        ratio nor a coverage ratio can be computed.
    """
    flags: list[str] = []

    # Payout ratio: dividends / net income (only meaningful for positive income).
    payout: float | None = None
    if dividends_paid is not None and net_income is not None:
        if net_income > 0:
            payout = abs(dividends_paid) / net_income
        else:
            payout = float("inf")  # paying a dividend out of losses
            flags.append("dividend paid despite non-positive net income")

    # Coverage: FCF / dividends preferred, else EPS / DPS.
    coverage: float | None = None
    coverage_basis = ""
    if (
        free_cash_flow is not None
        and dividends_paid is not None
        and dividends_paid != 0
    ):
        coverage = free_cash_flow / abs(dividends_paid)
        coverage_basis = "fcf"
    elif eps is not None and dps is not None and dps != 0:
        coverage = eps / dps
        coverage_basis = "earnings"

    if payout is None and coverage is None:
        return DividendQuality(ticker=ticker, available=False)

    # Trailing dividend growth (per share).
    growth: float | None = None
    if dps is not None and prior_dps is not None and prior_dps != 0:
        growth = (dps - prior_dps) / abs(prior_dps)
        if growth < 0:
            flags.append("dividend per share cut year-over-year")

    if payout is not None and payout > 1.0:
        flags.append("payout exceeds 100% of earnings")
    if coverage is not None and coverage < 1.0:
        flags.append(f"{coverage_basis} coverage below 1.0x")

    # Yield-trap: a high stated yield, paying out more than earned, with
    # deteriorating/insufficient coverage. Classic value trap.
    yield_trap = False
    if (
        dividend_yield is not None
        and dividend_yield >= high_yield_threshold
        and payout is not None
        and payout > 1.0
        and (
            coverage is not None
            and coverage < 1.0
            or (growth is not None and growth < 0)
        )
    ):
        yield_trap = True
        flags.append("yield-trap: high yield + payout>100% + weak/declining coverage")

    # Verdict ladder.
    if (
        yield_trap
        or (payout is not None and payout > 1.25)
        or (coverage is not None and coverage < 0.8)
    ):
        verdict = "UNSUSTAINABLE"
    elif (payout is not None and payout > 0.8) or (
        coverage is not None and coverage < 1.25
    ):
        verdict = "STRETCHED"
    else:
        verdict = "SAFE"

    return DividendQuality(
        ticker=ticker,
        available=True,
        verdict=verdict,
        payout_ratio=payout,
        coverage=coverage,
        coverage_basis=coverage_basis,
        dividend_growth=growth,
        yield_pct=dividend_yield,
        yield_trap=yield_trap,
        flags=flags,
    )


# ──────────────────────────────────────────────────────────────────────────
# Credit quality (Merton distance-to-default + coverage + leverage)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class CreditQuality:
    """Structured credit-quality verdict.

    ``verdict`` is one of INVESTMENT_GRADE / SPECULATIVE / DISTRESSED /
    UNAVAILABLE. All ratios are deterministic; ``default_prob`` comes from the
    Merton model with a locally-computed Phi (engine cross-checked when up).
    """

    ticker: str
    available: bool
    verdict: str = "UNAVAILABLE"
    distance_to_default: float | None = None
    default_prob: float | None = None  # Phi(-DD) over horizon T
    interest_coverage: float | None = None  # EBIT / interest
    debt_to_equity: float | None = None
    debt_to_assets: float | None = None
    horizon_years: float = 1.0
    flags: list[str] = field(default_factory=list)

    @property
    def is_red_flag(self) -> bool:
        return self.available and self.verdict.upper() == "DISTRESSED"

    def citation(self) -> str:
        if not self.available:
            return (
                f"Credit quality for {self.ticker} UNAVAILABLE "
                "(insufficient inputs — no numbers fabricated)."
            )

        def _f(v: float | None) -> str:
            return f"{v:.2f}" if isinstance(v, int | float) else "n/a"

        def _p(v: float | None) -> str:
            return f"{v:.2%}" if isinstance(v, int | float) else "n/a"

        flag_str = ("; flags: " + ", ".join(self.flags)) if self.flags else ""
        return (
            f"Credit [{self.verdict}] for {self.ticker}: "
            f"Merton DD={_f(self.distance_to_default)} "
            f"(PD={_p(self.default_prob)} @ {self.horizon_years:g}y), "
            f"interest coverage={_f(self.interest_coverage)}x, "
            f"D/E={_f(self.debt_to_equity)}, D/A={_f(self.debt_to_assets)}{flag_str}."
        )


def merton_distance_to_default(
    asset_value: float,
    debt_face: float,
    asset_vol: float,
    *,
    asset_drift: float = 0.0,
    horizon_years: float = 1.0,
    engine_client: Any | None = None,
) -> tuple[float, float]:
    """Merton structural distance-to-default and default probability.

        DD = (ln(V / D) + (mu - 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        PD = Phi(-DD)

    Args:
        asset_value: Firm asset value V (equity value + debt is a common proxy).
        debt_face: Face value of debt D (the default barrier).
        asset_vol: Annualized asset volatility sigma (> 0).
        asset_drift: Asset drift mu (risk-free or expected return). Default 0.
        horizon_years: Horizon T in years. Default 1.
        engine_client: Optional engine for a Phi cross-check; local Phi is the
            source of truth and is always used if the engine disagrees/fails.

    Returns:
        ``(distance_to_default, default_prob)``.

    Raises:
        ValueError: on non-positive V, D, sigma, or T (no silent guessing).
    """
    if asset_value <= 0 or debt_face <= 0 or asset_vol <= 0 or horizon_years <= 0:
        raise ValueError(
            "Merton model requires positive asset_value, debt_face, asset_vol, "
            f"horizon_years (got V={asset_value}, D={debt_face}, "
            f"sigma={asset_vol}, T={horizon_years})"
        )

    dd = (
        math.log(asset_value / debt_face)
        + (asset_drift - 0.5 * asset_vol**2) * horizon_years
    ) / (asset_vol * math.sqrt(horizon_years))

    pd = normal_cdf(-dd)

    # Optional engine cross-check of Phi(-DD). The engine has no bare-CDF method,
    # but ``deflated_sharpe`` reduces to a cumulative-normal evaluation we can use
    # as an independent confirmation; on any mismatch/failure we keep the local
    # value (local Phi is the source of truth).
    client = engine_client if engine_client is not None else _credit_engine()
    if client is not None:
        try:
            confirm = client.finance.deflated_sharpe(-dd, 1, [0.0, 0.0, 0.0])
            if isinstance(confirm, int | float) and math.isfinite(confirm):
                if abs(float(confirm) - pd) < 1e-3:
                    logger.debug("engine confirmed Merton PD within tolerance")
        except Exception as exc:  # noqa: BLE001 — local Phi already authoritative
            logger.debug("engine Phi cross-check unavailable: %s", exc)

    return dd, pd


def assess_credit_quality(
    ticker: str,
    *,
    equity_value: float | None = None,
    equity_vol: float | None = None,
    total_debt: float | None = None,
    total_assets: float | None = None,
    total_equity: float | None = None,
    ebit: float | None = None,
    interest_expense: float | None = None,
    asset_drift: float = 0.0,
    horizon_years: float = 1.0,
    engine_client: Any | None = None,
) -> CreditQuality:
    """Assess credit quality via Merton DD + interest coverage + leverage.

    Asset value/vol are derived from the equity (a common Merton proxy):
    ``V ≈ equity_value + total_debt`` and ``sigma_V ≈ sigma_E * E / V`` (the
    leverage-deflated equity vol). Distance-to-default is skipped (left ``None``)
    when the equity inputs are absent, but coverage/leverage are still reported.

    Returns:
        A ``CreditQuality``. ``available`` is ``False`` only when *no* metric
        (DD, coverage, or leverage) can be computed.
    """
    flags: list[str] = []
    dd: float | None = None
    pd: float | None = None

    # Merton DD from equity value/vol + debt.
    if (
        equity_value is not None
        and equity_vol is not None
        and total_debt is not None
        and equity_value > 0
        and equity_vol > 0
        and total_debt > 0
    ):
        asset_value = equity_value + total_debt
        # Deflate equity vol by leverage to approximate asset vol.
        asset_vol = equity_vol * (equity_value / asset_value)
        try:
            dd, pd = merton_distance_to_default(
                asset_value,
                total_debt,
                asset_vol,
                asset_drift=asset_drift,
                horizon_years=horizon_years,
                engine_client=engine_client,
            )
        except ValueError as exc:
            logger.debug("Merton DD skipped for %s: %s", ticker, exc)

    # Interest coverage: EBIT / interest.
    interest_coverage: float | None = None
    if ebit is not None and interest_expense is not None and interest_expense != 0:
        interest_coverage = ebit / abs(interest_expense)
        if interest_coverage < 1.5:
            flags.append("interest coverage below 1.5x")

    # Leverage ratios.
    d_to_e: float | None = None
    if total_debt is not None and total_equity is not None and total_equity != 0:
        d_to_e = total_debt / abs(total_equity)
        if d_to_e > 2.0:
            flags.append("debt/equity above 2.0")
    d_to_a: float | None = None
    if total_debt is not None and total_assets is not None and total_assets != 0:
        d_to_a = total_debt / abs(total_assets)
        if d_to_a > 0.7:
            flags.append("debt/assets above 70%")

    if dd is None and interest_coverage is None and d_to_e is None and d_to_a is None:
        return CreditQuality(ticker=ticker, available=False)

    if pd is not None and pd > 0.10:
        flags.append("Merton default probability above 10%")

    # Verdict: distress dominates; otherwise score on DD + coverage + leverage.
    distressed = (
        (pd is not None and pd > 0.20)
        or (interest_coverage is not None and interest_coverage < 1.0)
        or (dd is not None and dd < 1.0)
    )
    speculative = (
        (pd is not None and 0.02 < pd <= 0.20)
        or (interest_coverage is not None and interest_coverage < 3.0)
        or (dd is not None and dd < 3.0)
        or (d_to_e is not None and d_to_e > 2.0)
    )
    if distressed:
        verdict = "DISTRESSED"
    elif speculative:
        verdict = "SPECULATIVE"
    else:
        verdict = "INVESTMENT_GRADE"

    return CreditQuality(
        ticker=ticker,
        available=True,
        verdict=verdict,
        distance_to_default=dd,
        default_prob=pd,
        interest_coverage=interest_coverage,
        debt_to_equity=d_to_e,
        debt_to_assets=d_to_a,
        horizon_years=horizon_years,
        flags=flags,
    )


# ──────────────────────────────────────────────────────────────────────────
# Debate / forensic wiring
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class CreditDividendReport:
    """Combined dividend + credit verdict bundle for the debate path."""

    ticker: str
    dividend: DividendQuality
    credit: CreditQuality

    @property
    def is_red_flag(self) -> bool:
        return self.dividend.is_red_flag or self.credit.is_red_flag

    def citation(self) -> str:
        """A two-line citable block the Bear/Risk persona folds into its case."""
        return f"{self.credit.citation()}\n{self.dividend.citation()}"


def emit_credit_dividend_report(
    ticker: str,
    financials: dict[str, Any],
    *,
    engine_client: Any | None = None,
) -> CreditDividendReport:
    """Build a combined credit + dividend report from a financials dict.

    Accepts the same loosely-typed financials mapping the forensic path uses
    (a superset of the 17-key forensic schema), reading whichever of these keys
    are present::

        dividends_paid, net_income, free_cash_flow, eps, dps, prior_dps,
        dividend_yield, equity_value, equity_vol, total_debt, total_assets,
        total_equity, ebit, interest_expense, asset_drift, horizon_years

    Missing keys degrade to ``available=False`` on the relevant verdict rather
    than inventing figures. This is the single entry point the debate wiring and
    any bear/risk emitter calls.
    """
    f = financials
    dividend = assess_dividend_quality(
        ticker,
        dividends_paid=f.get("dividends_paid"),
        net_income=f.get("net_income"),
        free_cash_flow=f.get("free_cash_flow"),
        eps=f.get("eps"),
        dps=f.get("dps"),
        prior_dps=f.get("prior_dps"),
        dividend_yield=f.get("dividend_yield"),
    )
    credit = assess_credit_quality(
        ticker,
        equity_value=f.get("equity_value"),
        equity_vol=f.get("equity_vol"),
        total_debt=f.get("total_debt"),
        total_assets=f.get("total_assets"),
        total_equity=f.get("total_equity"),
        ebit=f.get("ebit"),
        interest_expense=f.get("interest_expense"),
        asset_drift=float(f.get("asset_drift", 0.0)),
        horizon_years=float(f.get("horizon_years", 1.0)),
        engine_client=engine_client,
    )
    return CreditDividendReport(ticker=ticker, dividend=dividend, credit=credit)


def attach_to_debate_context(
    context: Any,
    financials: dict[str, Any],
    *,
    engine_client: Any | None = None,
) -> CreditDividendReport:
    """Fold a credit + dividend citation into a live ``DebateContext``.

    Appends the combined citation to ``context.fundamentals_report`` — the field
    ``DebateEngine._generate_bear_argument`` and ``_evaluate_risk`` actually read
    on the live debate path — so the Bear/Burry and Risk personas argue against
    the real solvency and payout numbers. Returns the report for further use.

    The append is idempotent-friendly: it only adds the block, never clobbers an
    existing market/fundamentals narrative.
    """
    report = emit_credit_dividend_report(
        getattr(context, "ticker", "?"), financials, engine_client=engine_client
    )
    block = "Credit & Dividend Quality:\n" + report.citation()
    existing = getattr(context, "fundamentals_report", "") or ""
    context.fundamentals_report = (
        (existing + "\n\n" + block).strip() if existing else block
    )
    return report
