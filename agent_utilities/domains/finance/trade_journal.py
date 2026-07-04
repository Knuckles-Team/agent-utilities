"""CONCEPT:AU-KG.domains.trade-journal-bias-auditor — Trade-Journal Bias Auditor and Shadow Account

Borrowed from Vibe-Trading's *shadow account / trade-journal* skill, but made a
first-class, **queryable learning signal** rather than a one-shot report.

Given a list of executed roundtrips (entry/exit/size/pnl/timestamps) this module:

* computes a **trader profile** (win rate, avg holding period, PnL ratio, max
  drawdown, trade frequency), and
* runs **four behavioural-bias diagnostics** — disposition effect (holding
  losers longer than winners), overtrading (busy-day PnL drag), momentum-chasing
  (buying into one's own recent runs), and anchoring (re-trading the same narrow
  price band) — each with a numeric metric + severity, and
* optionally **persists the profile + biases as KG nodes** through the single
  ``write_batch`` → ``GraphBackend`` path every other enrichment source uses.

The KG/OWL uniqueness this leverages (vs. Vibe-Trading's flat report): once a
trader's biases are nodes (``:BehavioralBias`` ``EXHIBITED_BY`` ``:TraderProfile``),
a future Bull/Bear debate or the risk officer can *cite* them — e.g. "this account
has a HIGH disposition effect, so weight the bear's stop-loss argument up". The
report becomes a fact reasoned over, not a PDF.

All maths is real (FIFO already-matched roundtrips in, no placeholders). The
module imports and runs **fully offline**: ``pandas`` and a KG backend are both
optional; absent either, it degrades to pure-Python metrics with no persistence.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Severity thresholds shared across diagnostics: (medium_cut, high_cut).
_DISPOSITION_THRESHOLDS = (1.2, 1.5)
_CHASE_THRESHOLDS = (0.35, 0.6)
_ANCHOR_THRESHOLDS = (0.3, 0.5)
_OVERTRADE_THRESHOLDS = (0.3, 1.0)


def _severity(score: float, thresholds: tuple[float, float]) -> str:
    """Map a magnitude onto low/medium/high using (medium, high) cut-offs."""
    medium, high = thresholds
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    return "low"


def _to_dt(value: Any) -> datetime:
    """Coerce an ISO string / datetime / epoch into a ``datetime``."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value))
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


@dataclass
class Roundtrip:
    """One closed round-trip (already FIFO-matched upstream).

    ``side`` is the direction of the *opening* leg (``"buy"`` for a long,
    ``"sell"`` for a short). ``pnl`` is realized P&L in account currency.
    """

    symbol: str
    entry_time: Any
    exit_time: Any
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    side: str = "buy"

    @property
    def hold_days(self) -> float:
        """Holding period in days (>= 0)."""
        delta = _to_dt(self.exit_time) - _to_dt(self.entry_time)
        return max(0.0, delta.total_seconds() / 86_400.0)

    @property
    def pnl_pct(self) -> float:
        """Return on the entry notional (signed for shorts)."""
        notional = abs(self.entry_price * self.size)
        if notional <= 0:
            return 0.0
        return self.pnl / notional


@dataclass
class BiasDiagnostic:
    """A single behavioural-bias finding with numeric evidence."""

    name: str
    severity: str  # low | medium | high
    metric: float
    evidence: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraderProfile:
    """Aggregate portrait of a trader from their closed roundtrips."""

    trader_id: str
    total_roundtrips: int
    win_rate: float
    avg_holding_days: float
    profit_loss_ratio: float
    total_pnl: float
    max_drawdown: float
    avg_win_pnl: float
    avg_loss_pnl: float
    biases: list[BiasDiagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["biases"] = [asdict(b) for b in self.biases]
        return d

    def bias(self, name: str) -> BiasDiagnostic | None:
        """Return the named diagnostic, or ``None`` if absent."""
        return next((b for b in self.biases if b.name == name), None)

    def citation(self) -> str:
        """One-line, citable summary a debate persona can quote verbatim."""
        flagged = [b for b in self.biases if b.severity in ("medium", "high")]
        bias_str = (
            "; ".join(f"{b.name}={b.severity}" for b in flagged)
            if flagged
            else "no material behavioural bias"
        )
        return (
            f"Trader {self.trader_id}: win-rate {self.win_rate:.0%}, "
            f"PnL-ratio {self.profit_loss_ratio:.2f}, "
            f"max-drawdown {self.max_drawdown:.0f}, "
            f"avg-hold {self.avg_holding_days:.1f}d. Biases: {bias_str}. "
            "(source: trade_journal auditor KG-2.26)"
        )


def _max_drawdown(pnls: list[float]) -> float:
    """Most negative trough of the cumulative-PnL equity curve (<= 0)."""
    cum = 0.0
    peak = 0.0
    worst = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    return worst


def _disposition_effect(rts: list[Roundtrip]) -> BiasDiagnostic:
    """Holding losers longer than winners — ratio of avg loser/winner hold."""
    wins = [r for r in rts if r.pnl > 0]
    losses = [r for r in rts if r.pnl < 0]
    if not wins or not losses:
        return BiasDiagnostic(
            "disposition_effect",
            "low",
            0.0,
            "not enough winners and losers to compare holding times",
        )
    win_hold = statistics.fmean(r.hold_days for r in wins)
    loss_hold = statistics.fmean(r.hold_days for r in losses)
    ratio = loss_hold / win_hold if win_hold > 0 else float("inf")
    sev = _severity(ratio, _DISPOSITION_THRESHOLDS)
    return BiasDiagnostic(
        "disposition_effect",
        sev,
        round(ratio, 3),
        f"Losers held {loss_hold:.1f}d vs winners {win_hold:.1f}d (ratio {ratio:.2f}).",
        {
            "avg_winner_hold_days": round(win_hold, 2),
            "avg_loser_hold_days": round(loss_hold, 2),
        },
    )


def _overtrading(rts: list[Roundtrip]) -> BiasDiagnostic:
    """Busy days produce worse PnL than quiet days (activity-PnL drag)."""
    if len(rts) < 4:
        return BiasDiagnostic("overtrading", "low", 0.0, "fewer than 4 roundtrips")
    by_day: dict[Any, list[Roundtrip]] = {}
    for r in rts:
        by_day.setdefault(_to_dt(r.exit_time).date(), []).append(r)
    if len(by_day) < 4:
        return BiasDiagnostic("overtrading", "low", 0.0, "fewer than 4 trading days")
    counts = sorted(len(v) for v in by_day.values())
    n = len(counts)
    busy_cut = counts[int(0.75 * (n - 1))]
    quiet_cut = counts[int(0.25 * (n - 1))]
    busy_pnls = [r.pnl for d, v in by_day.items() if len(v) >= busy_cut for r in v]
    quiet_pnls = [r.pnl for d, v in by_day.items() if len(v) <= quiet_cut for r in v]
    if not busy_pnls or not quiet_pnls:
        return BiasDiagnostic(
            "overtrading", "low", 0.0, "roundtrips not spread across busy/quiet days"
        )
    busy_avg = statistics.fmean(busy_pnls)
    quiet_avg = statistics.fmean(quiet_pnls)
    base = abs(quiet_avg) if quiet_avg != 0 else 1.0
    gap = (quiet_avg - busy_avg) / base
    sev = _severity(gap, _OVERTRADE_THRESHOLDS) if busy_avg < quiet_avg else "low"
    return BiasDiagnostic(
        "overtrading",
        sev,
        round(gap, 3) if busy_avg < quiet_avg else 0.0,
        f"Busy-day avg PnL {busy_avg:+.0f} (>= {busy_cut} trades) vs "
        f"quiet-day {quiet_avg:+.0f} (<= {quiet_cut}).",
        {
            "busy_day_avg_pnl": round(busy_avg, 2),
            "quiet_day_avg_pnl": round(quiet_avg, 2),
        },
    )


def _momentum_chasing(rts: list[Roundtrip]) -> BiasDiagnostic:
    """Buying into one's own recent run — fraction of buys at a higher entry
    than the previous buy of the same symbol (a chase)."""
    buys = sorted(
        (r for r in rts if r.side == "buy"),
        key=lambda r: (r.symbol, _to_dt(r.entry_time)),
    )
    if len(buys) < 3:
        return BiasDiagnostic("momentum_chasing", "low", 0.0, "fewer than 3 buys")
    chased = 0
    matured = 0
    last_price: dict[str, float] = {}
    for r in buys:
        prev = last_price.get(r.symbol)
        if prev is not None:
            matured += 1
            if r.entry_price > prev * 1.03:  # entered >3% above own prior buy
                chased += 1
        last_price[r.symbol] = r.entry_price
    if matured == 0:
        return BiasDiagnostic(
            "momentum_chasing", "low", 0.0, "not enough repeat buys per symbol"
        )
    ratio = chased / matured
    sev = _severity(ratio, _CHASE_THRESHOLDS)
    return BiasDiagnostic(
        "momentum_chasing",
        sev,
        round(ratio, 3),
        f"{chased}/{matured} buys ({ratio:.0%}) entered >3% above the prior buy "
        "of the same symbol.",
        {"chased": chased, "matured": matured},
    )


def _anchoring(rts: list[Roundtrip]) -> BiasDiagnostic:
    """Re-trading the same narrow price band — fraction of frequently-traded
    symbols whose entry prices have a coefficient of variation < 5%."""
    by_symbol: dict[str, list[float]] = {}
    for r in rts:
        by_symbol.setdefault(r.symbol, []).append(r.entry_price)
    frequent = {s: ps for s, ps in by_symbol.items() if len(ps) >= 5}
    if not frequent:
        return BiasDiagnostic(
            "anchoring", "low", 0.0, "no symbol traded >=5 times to evaluate anchoring"
        )
    anchored = 0
    for prices in frequent.values():
        mean = statistics.fmean(prices)
        if mean <= 0:
            continue
        cv = statistics.pstdev(prices) / mean
        if cv < 0.05:
            anchored += 1
    ratio = anchored / len(frequent)
    sev = _severity(ratio, _ANCHOR_THRESHOLDS)
    return BiasDiagnostic(
        "anchoring",
        sev,
        round(ratio, 3),
        f"{anchored}/{len(frequent)} frequently-traded symbols stayed within a "
        "<5% entry-price band.",
        {"anchored_symbols": anchored, "frequent_symbols": len(frequent)},
    )


class TradeJournalAuditor:
    """Audit a trader's executed roundtrips into a profile + bias diagnostics,
    and optionally persist them to the KG as queryable learning signals.

    Usage::

        auditor = TradeJournalAuditor()
        profile = auditor.audit("acct_42", roundtrips)
        if profile.bias("disposition_effect").severity == "high":
            ...  # risk officer weights the bear's stop-loss argument up
        auditor.persist(profile, backend)  # KG-2.26 nodes, optional
    """

    def audit(
        self, trader_id: str, roundtrips: list[Roundtrip | dict[str, Any]]
    ) -> TraderProfile:
        """Compute the profile + four bias diagnostics for ``trader_id``."""
        rts = [r if isinstance(r, Roundtrip) else Roundtrip(**r) for r in roundtrips]
        if not rts:
            return TraderProfile(
                trader_id=trader_id,
                total_roundtrips=0,
                win_rate=0.0,
                avg_holding_days=0.0,
                profit_loss_ratio=0.0,
                total_pnl=0.0,
                max_drawdown=0.0,
                avg_win_pnl=0.0,
                avg_loss_pnl=0.0,
            )

        rts_sorted = sorted(rts, key=lambda r: _to_dt(r.exit_time))
        wins = [r for r in rts if r.pnl > 0]
        losses = [r for r in rts if r.pnl < 0]
        win_rate = len(wins) / len(rts)
        avg_win = statistics.fmean(r.pnl for r in wins) if wins else 0.0
        avg_loss = statistics.fmean(r.pnl for r in losses) if losses else 0.0
        # PnL ratio = avg win / avg |loss| (classic edge ratio).
        pnl_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else float("inf")

        profile = TraderProfile(
            trader_id=trader_id,
            total_roundtrips=len(rts),
            win_rate=round(win_rate, 4),
            avg_holding_days=round(statistics.fmean(r.hold_days for r in rts), 3),
            profit_loss_ratio=round(pnl_ratio, 3)
            if pnl_ratio != float("inf")
            else pnl_ratio,
            total_pnl=round(sum(r.pnl for r in rts), 2),
            max_drawdown=round(_max_drawdown([r.pnl for r in rts_sorted]), 2),
            avg_win_pnl=round(avg_win, 2),
            avg_loss_pnl=round(avg_loss, 2),
            biases=[
                _disposition_effect(rts),
                _overtrading(rts),
                _momentum_chasing(rts),
                _anchoring(rts),
            ],
        )
        return profile

    def to_batch(self, profile: TraderProfile) -> Any:
        """Build an ``ExtractionBatch`` of the profile + bias nodes (KG-2.26).

        Shape: one ``:TraderProfile`` node, four ``:BehavioralBias`` nodes, each
        ``EXHIBITED_BY`` the profile. This is the same backend-agnostic node/edge
        shape every enrichment source emits, so it persists through ``write_batch``.
        """
        from agent_utilities.knowledge_graph.enrichment.models import (
            EnrichmentEdge,
            ExtractionBatch,
            GraphNode,
        )

        pid = f"trader_profile:{profile.trader_id}"
        nodes = [
            GraphNode(
                id=pid,
                type="TraderProfile",
                props={
                    "trader_id": profile.trader_id,
                    "win_rate": profile.win_rate,
                    "profit_loss_ratio": (
                        profile.profit_loss_ratio
                        if profile.profit_loss_ratio != float("inf")
                        else None
                    ),
                    "avg_holding_days": profile.avg_holding_days,
                    "total_pnl": profile.total_pnl,
                    "max_drawdown": profile.max_drawdown,
                    "total_roundtrips": profile.total_roundtrips,
                    "concept": "AU-KG.domains.trade-journal-bias-auditor",
                },
            )
        ]
        edges = []
        for b in profile.biases:
            bid = f"behavioral_bias:{profile.trader_id}:{b.name}"
            nodes.append(
                GraphNode(
                    id=bid,
                    type="BehavioralBias",
                    props={
                        "bias_name": b.name,
                        "severity": b.severity,
                        "metric": b.metric,
                        "evidence": b.evidence,
                        "concept": "AU-KG.domains.trade-journal-bias-auditor",
                    },
                )
            )
            edges.append(
                EnrichmentEdge(source=bid, target=pid, rel_type="EXHIBITED_BY")
            )
        return ExtractionBatch(category="trade_journal", nodes=nodes, edges=edges)

    def persist(self, profile: TraderProfile, backend: Any) -> tuple[int, int]:
        """Persist the profile + biases into the KG via ``write_batch``.

        Returns ``(nodes_written, edges_written)``. A ``None`` backend (offline)
        is a no-op returning ``(0, 0)`` — the audit still works without a graph.
        """
        if backend is None:
            return (0, 0)
        from agent_utilities.knowledge_graph.enrichment.registry import write_batch

        n, e = write_batch(backend, self.to_batch(profile))
        logger.info(
            "Persisted trader %s profile + %d biases: %d nodes, %d edges",
            profile.trader_id,
            len(profile.biases),
            n,
            e,
        )
        return n, e


def audit_trade_journal(
    trader_id: str,
    roundtrips: list[Roundtrip | dict[str, Any]],
    backend: Any = None,
) -> TraderProfile:
    """Convenience: audit roundtrips and (if a backend is given) persist to KG."""
    auditor = TradeJournalAuditor()
    profile = auditor.audit(trader_id, roundtrips)
    if backend is not None:
        auditor.persist(profile, backend)
    return profile
