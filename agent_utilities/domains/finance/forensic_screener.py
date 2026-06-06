"""
Forensic Screener — CONCEPT:KG-2.6

Engine-grounded forensic-accounting screen for an equity. Takes a ticker's two
fiscal years of standardized financial line items, calls the epistemic-graph
engine's ``client.finance.forensic_report`` (Beneish M-score / Altman Z-score /
Piotroski F-score / Sloan accruals), and returns a structured verdict the
Bear/Burry persona can cite verbatim in the debate.

The numbers are NEVER hallucinated — every score comes from the Rust engine.
When the engine is unreachable (offline / unit tests), the screen degrades
gracefully to an ``UNAVAILABLE`` verdict instead of inventing figures.

Source: epistemic-graph CONCEPT:KG-2.20g forensic accounting kernels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Lazy, cached epistemic-graph client. Probed once; ``None`` when the engine is
# unreachable so importing this module never requires a running engine.
_ENGINE_PROBED = False
_ENGINE_CLIENT: Any = None


def _forensic_engine() -> Any:
    """Return a connected SyncEpistemicGraphClient, or ``None`` if unavailable."""
    global _ENGINE_PROBED, _ENGINE_CLIENT
    if _ENGINE_PROBED:
        return _ENGINE_CLIENT
    _ENGINE_PROBED = True
    try:
        from epistemic_graph.client import SyncEpistemicGraphClient

        _ENGINE_CLIENT = SyncEpistemicGraphClient.connect()
        logger.info("epistemic-graph engine connected for forensic screening")
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, never invent
        logger.debug(
            "epistemic-graph engine unavailable for forensic screening: %s", exc
        )
        _ENGINE_CLIENT = None
    return _ENGINE_CLIENT


def reset_engine_cache() -> None:
    """Reset the cached engine probe (used by tests to re-probe)."""
    global _ENGINE_PROBED, _ENGINE_CLIENT
    _ENGINE_PROBED = False
    _ENGINE_CLIENT = None


@dataclass
class ForensicVerdict:
    """Structured, engine-grounded forensic screen result.

    All scores originate from the engine's ``forensic_report``; ``available``
    is ``False`` when the engine could not be reached (no fabricated numbers).
    """

    ticker: str
    available: bool
    verdict: str = "UNAVAILABLE"  # INVESTIGATE | CLEAN | UNAVAILABLE
    m_score: float | None = None
    z_score: float | None = None
    f_score: float | None = None
    accruals_ratio: float | None = None
    flags: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_red_flag(self) -> bool:
        """True when the engine flagged the filing for investigation."""
        return self.available and self.verdict.upper() == "INVESTIGATE"

    def citation(self) -> str:
        """A one-line, citable summary for a debate argument."""
        if not self.available:
            return (
                f"Forensic screen for {self.ticker} UNAVAILABLE "
                "(engine offline — no numbers fabricated)."
            )

        def _fmt(v: float | None) -> str:
            return f"{v:.2f}" if isinstance(v, int | float) else "n/a"

        flag_str = ("; flags: " + ", ".join(self.flags)) if self.flags else ""
        return (
            f"Forensic screen [{self.verdict}] for {self.ticker}: "
            f"Beneish M={_fmt(self.m_score)}, Altman Z={_fmt(self.z_score)}, "
            f"Piotroski F={_fmt(self.f_score)}, "
            f"Sloan accruals={_fmt(self.accruals_ratio)}{flag_str}. "
            "(source: epistemic-graph forensic_report)"
        )


class ForensicScreener:
    """Run the engine forensic screen for an equity over two fiscal years.

    Usage::

        screener = ForensicScreener()
        verdict = screener.screen("ACME", this_year={...}, prior_year={...})
        if verdict.is_red_flag:
            ...  # Bear/Burry persona cites verdict.citation()
    """

    def __init__(self, engine_client: Any | None = None):
        # An explicit client (e.g. an injected mock) overrides lazy probing.
        self._explicit_client = engine_client

    def _client(self) -> Any:
        if self._explicit_client is not None:
            return self._explicit_client
        return _forensic_engine()

    def screen(
        self,
        ticker: str,
        this_year: dict[str, Any],
        prior_year: dict[str, Any],
    ) -> ForensicVerdict:
        """Call the engine's forensic_report and return a structured verdict.

        Args:
            ticker: The equity symbol (for labelling/citation only).
            this_year: Standardized line items for the most recent fiscal year
                (sales, cogs, net_income, cfo, total_assets, ...).
            prior_year: The same line items for the prior fiscal year.

        Returns:
            A ``ForensicVerdict``. ``available`` is ``False`` (and the verdict
            is ``UNAVAILABLE``) when the engine is unreachable — never a
            fabricated score.
        """
        client = self._client()
        if client is None:
            return ForensicVerdict(ticker=ticker, available=False)

        try:
            report: dict[str, Any] = client.finance.forensic_report(
                this_year, prior_year
            )
        except Exception as exc:  # noqa: BLE001 — engine error degrades, never invents
            logger.warning("forensic_report failed for %s: %s", ticker, exc)
            return ForensicVerdict(ticker=ticker, available=False)

        if not isinstance(report, dict):
            logger.warning(
                "Unexpected forensic_report payload for %s: %r", ticker, report
            )
            return ForensicVerdict(ticker=ticker, available=False)

        flags_raw = report.get("flags") or []
        flags = [str(f) for f in flags_raw] if isinstance(flags_raw, list) else []

        return ForensicVerdict(
            ticker=ticker,
            available=True,
            verdict=str(report.get("verdict", "UNAVAILABLE")),
            m_score=report.get("m_score"),
            z_score=report.get("z_score"),
            f_score=report.get("f_score"),
            accruals_ratio=report.get("accruals_ratio"),
            flags=flags,
            raw=report,
        )
