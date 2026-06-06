"""
Copy-Trade Pipeline — CONCEPT:KG-2.6

Engine-grounded scanner → brain → execution → exit pipeline for prediction-market
copy-trading, wired over the EXISTING swarm-consensus machinery. This is the
orchestration layer for the "follow the consistently-profitable wallets" strategy:

* **Scanner** (:func:`score_market`) — ruthless filter on edge (gap), book depth,
  and time-to-resolution; kills ~90% of markets before any model spend.
* **Brain / execution** (:class:`CopyTradePipeline`) — runs the three copy-trade
  agents (arbitrage, convergence, whale-copy) and aggregates them through the
  same role-weighted :class:`SwarmConsensus` used everywhere else: 2-of-3 agree →
  full position, 1 → half, disagree → no trade. Position size is the engine's
  Kelly fraction (``client.finance.kelly_fraction`` / ``bayesian_kelly``), never a
  hand-rolled formula.
* **Exit** (:func:`exit_check`) — target-hit (≈85 % of expected move), volume-spike
  (3× in a 10-min window = smart money leaving), and stale-thesis (24 h no move).

The pipeline returns *intended* trades only — it never places live orders. Live
execution stays behind ``emerald-exchange`` RiskGuard's ``require_human_approval_live``.

Smart-money inputs (wallet ranking, convergence) come from emerald-exchange's
``emerald_wallet_intel`` tool group; this module accepts that data injected so it
stays unit-testable offline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.domains.finance.trading_swarm import (
    AgentSignal,
    SwarmConfig,
    SwarmConsensus,
    SwarmDecision,
    SwarmRole,
)

logger = logging.getLogger(__name__)

# Lazy, cached epistemic-graph client (mirrors forensic_screener). Probed once;
# ``None`` when the engine is unreachable so importing never needs a live engine.
_ENGINE_PROBED = False
_ENGINE_CLIENT: Any = None


def _engine() -> Any:
    global _ENGINE_PROBED, _ENGINE_CLIENT
    if _ENGINE_PROBED:
        return _ENGINE_CLIENT
    _ENGINE_PROBED = True
    try:
        from epistemic_graph.client import SyncEpistemicGraphClient

        _ENGINE_CLIENT = SyncEpistemicGraphClient.connect()
        logger.info("epistemic-graph engine connected for copy-trade sizing")
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, never invent
        logger.debug("epistemic-graph engine unavailable for copy-trade: %s", exc)
        _ENGINE_CLIENT = None
    return _ENGINE_CLIENT


def reset_engine_cache() -> None:
    """Reset the cached engine probe (used by tests to re-probe)."""
    global _ENGINE_PROBED, _ENGINE_CLIENT
    _ENGINE_PROBED = False
    _ENGINE_CLIENT = None


@dataclass
class CopyTradeConfig:
    """Tunable thresholds for the copy-trade pipeline.

    These defaults are deliberately conservative starting points — like the HFT
    handbooks warn, calibrate them on your own captures; do not trust any value
    a public article hands you.
    """

    # Scanner filters
    gap_min: float = 0.07  # min |estimate − price| edge; below this, costs eat it
    depth_min: float = 500.0  # min $ on both sides; thinner ⇒ you move the price
    hours_min: float = 4.0  # too late to enter below this
    hours_max: float = 168.0  # capital locked too long above this (7 days)
    # Consensus → position multiplier
    full_position_agreement: int = 2  # ≥2 of 3 agents agree ⇒ full size
    half_position_agreement: int = 1  # exactly 1 ⇒ half size
    # Sizing
    kelly_fraction: float = 0.25  # quarter-Kelly safety
    # Exit triggers
    target_capture: float = 0.85  # exit at 85 % of the expected move
    volume_spike_mult: float = 3.0  # 3× the avg ⇒ smart money moving
    stale_hours: float = 24.0  # no movement this long ⇒ thesis stale
    stale_move: float = 0.02  # "no movement" = |Δprice| < this


# ── Scanner ────────────────────────────────────────────────────────────────


def score_market(
    market: dict[str, Any],
    estimate: float,
    config: CopyTradeConfig | None = None,
) -> dict[str, Any] | None:
    """Score one market; return a queue entry, or ``None`` if it fails a filter.

    ``market`` carries: ``question``, ``midpoint``, ``bids_depth``, ``asks_depth``,
    ``hours_to_resolution``. ``estimate`` is your probability estimate for YES.
    """
    cfg = config or CopyTradeConfig()
    price = float(market["midpoint"])
    gap = abs(estimate - price)
    depth = min(
        float(market.get("bids_depth", 0.0)), float(market.get("asks_depth", 0.0))
    )
    hours = float(market.get("hours_to_resolution", 0.0))

    if gap < cfg.gap_min:
        return None  # edge too thin
    if depth < cfg.depth_min:
        return None  # can't fill without moving price
    if hours < cfg.hours_min:
        return None  # too late
    if hours > cfg.hours_max:
        return None  # too slow / capital locked

    return {
        "market": market.get("question", market.get("token", "?")),
        "token": market.get("token"),
        "price": price,
        "estimate": estimate,
        "gap": round(gap, 4),
        "depth": depth,
        "hours": hours,
        # expected-value proxy: edge × fillable depth
        "ev": round(gap * depth * 0.001, 3),
        "side": "YES" if estimate > price else "NO",
    }


# ── Consensus (reuses the swarm machinery) ──────────────────────────────────

# The three independent copy-trade agents, each mapped onto a swarm role so the
# existing role-weighted aggregation applies unchanged.
COPY_TRADE_AGENTS: dict[str, SwarmRole] = {
    "arbitrage": SwarmRole.QUANT_ANALYST,  # price gaps between related markets
    "convergence": SwarmRole.TREND_ANALYST,  # price moving toward estimate
    "whale_copy": SwarmRole.DIRECTOR,  # mirrors the target wallets
}


def aggregate_consensus(
    signals: list[AgentSignal],
    config: SwarmConfig | None = None,
) -> SwarmConsensus:
    """Role-weighted aggregation of the copy-trade agents' signals.

    Same weighted-score + risk-veto logic as :class:`TradingSwarm`, but for the
    fixed three-agent copy-trade panel.
    """
    cfg = config or SwarmConfig(min_agents_for_consensus=1)
    if not signals:
        return SwarmConsensus(
            decision=SwarmDecision.NO_CONSENSUS,
            weighted_score=0.0,
            agreement_ratio=0.0,
            signals=[],
        )
    total_w = 0.0
    weighted = 0.0
    for s in signals:
        w = cfg.role_weights.get(s.role, 1.0)
        weighted += s.direction * s.confidence * w
        total_w += w
    score = weighted / total_w if total_w > 0 else 0.0

    if score > 0:
        agreeing = sum(1 for s in signals if s.direction > 0)
    elif score < 0:
        agreeing = sum(1 for s in signals if s.direction < 0)
    else:
        agreeing = sum(1 for s in signals if s.direction == 0)
    agreement = agreeing / len(signals)

    if abs(score) < 0.1:
        decision = SwarmDecision.HOLD
    elif score > 0.5:
        decision = SwarmDecision.STRONG_BUY
    elif score > 0.1:
        decision = SwarmDecision.BUY
    elif score < -0.5:
        decision = SwarmDecision.STRONG_SELL
    else:
        decision = SwarmDecision.SELL
    majority = 1 if score > 0 else (-1 if score < 0 else 0)
    dissenters = [s.agent_id for s in signals if s.direction not in (majority, 0)]
    return SwarmConsensus(
        decision=decision,
        weighted_score=float(score),
        agreement_ratio=float(agreement),
        signals=signals,
        dissenting_agents=dissenters,
    )


def position_multiplier(
    consensus: SwarmConsensus, config: CopyTradeConfig | None = None
) -> float:
    """Map consensus agreement onto a position multiplier: 2 agree → 1.0,
    1 → 0.5, disagree → 0.0 (the article's consensus filter that killed 40 % of
    losing trades just by requiring agreement)."""
    cfg = config or CopyTradeConfig()
    majority = (
        1
        if consensus.weighted_score > 0
        else (-1 if consensus.weighted_score < 0 else 0)
    )
    if majority == 0:
        return 0.0
    n_agree = sum(1 for s in consensus.signals if s.direction == majority)
    if n_agree >= cfg.full_position_agreement:
        return 1.0
    if n_agree >= cfg.half_position_agreement:
        return 0.5
    return 0.0


# ── Exit logic ───────────────────────────────────────────────────────────────


def exit_check(
    position: dict[str, Any],
    current_price: float,
    volume_10m: float,
    avg_volume: float,
    config: CopyTradeConfig | None = None,
) -> str | None:
    """Return an exit reason, or ``None`` to hold.

    ``position`` carries ``entry``, ``target`` and ``hours_since_entry``.
    The top wallets take ~73 % of the potential profit and redeploy; 91 % of
    their exits happen before resolution — so we never just hold to settlement.
    """
    cfg = config or CopyTradeConfig()
    entry = float(position["entry"])
    target = float(position["target"])
    expected = target - entry
    # 1. target hit — captured the bulk of the expected move
    if expected != 0 and (current_price - entry) >= expected * cfg.target_capture:
        return "TARGET_HIT"
    # 2. volume spike — someone large is moving; be on the right side of the door
    if avg_volume > 0 and volume_10m > avg_volume * cfg.volume_spike_mult:
        return "VOLUME_EXIT"
    # 3. stale thesis — no movement for too long
    if (
        float(position.get("hours_since_entry", 0.0)) > cfg.stale_hours
        and abs(current_price - entry) < cfg.stale_move
    ):
        return "STALE_THESIS"
    return None


# ── Sizing (engine-grounded Kelly) ──────────────────────────────────────────


def size_position(
    estimate: float,
    price: float,
    multiplier: float,
    config: CopyTradeConfig | None = None,
    posterior: tuple[float, float] | None = None,
) -> float:
    """Kelly fraction for the trade, scaled by the consensus multiplier.

    Uses the engine's ``kelly_fraction`` (or ``bayesian_kelly`` when a Beta
    ``posterior=(alpha, beta)`` is supplied). Falls back to the closed-form
    f*=(q−c)/(1−c) only when the engine is offline, so unit tests still size.
    """
    cfg = config or CopyTradeConfig()
    if multiplier <= 0.0:
        return 0.0
    client = _engine()
    if client is not None:
        try:
            if posterior is not None:
                f = client.finance.bayesian_kelly(posterior[0], posterior[1], price, 50)
                f = min(f, 1.0) * cfg.kelly_fraction
            else:
                f = client.finance.kelly_fraction(estimate, price, cfg.kelly_fraction)
            return float(f) * multiplier
        except Exception as exc:  # noqa: BLE001
            logger.debug("engine Kelly unavailable, using local fallback: %s", exc)
    # local fallback (kept identical to the engine's point-Kelly formula)
    if price >= estimate or price <= 0.0 or price >= 1.0:
        return 0.0
    f_star = (estimate - price) / (1.0 - price)
    return max(0.0, min(f_star * cfg.kelly_fraction, 1.0)) * multiplier


# ── Pipeline ─────────────────────────────────────────────────────────────────


@dataclass
class CopyTradeIntent:
    """An intended (NOT executed) copy-trade decision."""

    market: str
    token: Any
    side: str
    decision: SwarmDecision
    position_fraction: float  # of bankroll; 0.0 ⇒ no trade
    multiplier: float
    agreement_ratio: float
    gap: float
    rationale: str = ""


@dataclass
class CopyTradePipeline:
    """Scanner → consensus → sizing pipeline producing intended trades.

    ``agent_evaluator`` is a callable ``(queue_entry) -> list[AgentSignal]`` that
    produces the three copy-trade agents' signals (arbitrage/convergence/whale).
    Inject it so the pipeline is testable without live MCP servers; in production
    it calls emerald-exchange's emerald_wallet_intel (convergence/whale) +
    market-data/quote tools (book/arb).
    """

    config: CopyTradeConfig = field(default_factory=CopyTradeConfig)
    swarm_config: SwarmConfig = field(
        default_factory=lambda: SwarmConfig(min_agents_for_consensus=1)
    )

    def run(
        self,
        markets: list[tuple[dict[str, Any], float]],
        agent_evaluator: Callable[[dict[str, Any]], list[AgentSignal]],
        posterior_fn: Callable[[dict[str, Any]], tuple[float, float] | None]
        | None = None,
    ) -> list[CopyTradeIntent]:
        """Run the full pipeline over ``(market, estimate)`` pairs."""
        intents: list[CopyTradeIntent] = []
        for market, estimate in markets:
            entry = score_market(market, estimate, self.config)
            if entry is None:
                continue  # killed by scanner
            signals = agent_evaluator(entry)
            consensus = aggregate_consensus(signals, self.swarm_config)
            mult = position_multiplier(consensus, self.config)
            posterior = posterior_fn(entry) if posterior_fn else None
            frac = size_position(estimate, entry["price"], mult, self.config, posterior)
            if frac <= 0.0:
                continue  # no edge after consensus/sizing
            intents.append(
                CopyTradeIntent(
                    market=entry["market"],
                    token=entry.get("token"),
                    side=entry["side"],
                    decision=consensus.decision,
                    position_fraction=round(frac, 4),
                    multiplier=mult,
                    agreement_ratio=round(consensus.agreement_ratio, 3),
                    gap=entry["gap"],
                    rationale=(
                        f"{consensus.decision.value} @ {entry['side']} "
                        f"(agree={consensus.agreement_ratio:.0%}, gap={entry['gap']})"
                    ),
                )
            )
        return intents


# ── Workflow registration (KG persistence) ──────────────────────────────────

COPY_TRADE_WORKFLOW = "copy_trade_prediction_market"


def build_copy_trade_workflow() -> Any:
    """Build the WorkflowSpec describing the scanner→brain→exec→exit flow."""
    from agent_utilities.knowledge_graph.enrichment.orchestration import WorkflowSpec

    return WorkflowSpec(
        name=COPY_TRADE_WORKFLOW,
        steps=["scan", "rank_wallets", "consensus", "size", "exit_monitor"],
        orchestrates=[
            # wallet-intel + venue I/O both folded into the emerald-exchange hub:
            # emerald_wallet_intel (rank_wallets / smart_money_convergence) and
            # the orderbook/quotes/risk tools.
            "tool:emerald-exchange",
            "tool:data-science-mcp",  # quant kernels (kelly/vpin)
            "agent:portfolio_manager",
        ],
    )


def seed_copy_trade_workflow(write_batch: Callable[[Any], Any]) -> Any:
    """Persist the copy-trade WorkflowSpec into the KG via ``write_batch``
    (the same path every other orchestration source uses)."""
    from agent_utilities.knowledge_graph.enrichment.orchestration import (
        workflow_to_batch,
    )

    spec = build_copy_trade_workflow()
    batch = workflow_to_batch(spec)
    return write_batch(batch)
