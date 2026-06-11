"""
Multi-Agent Trading Swarm — CONCEPT:KG-2.6

Provides specialized agent roles (Director, Quant Analyst, Risk Manager,
Execution Trader) with swarm orchestration for collaborative trading decisions.

Sources: QuantAgent multi-agent collaboration, AutoHedge risk-first swarm
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class SwarmRole(StrEnum):
    """Specialized roles within a trading swarm."""

    DIRECTOR = "director"
    QUANT_ANALYST = "quant_analyst"
    RISK_MANAGER = "risk_manager"
    EXECUTION_TRADER = "execution_trader"
    INDICATOR_SPECIALIST = "indicator_specialist"
    PATTERN_ANALYST = "pattern_analyst"
    TREND_ANALYST = "trend_analyst"
    SENTIMENT_ANALYST = "sentiment_analyst"
    BULL_RESEARCHER = "bull_researcher"
    BEAR_RESEARCHER = "bear_researcher"
    FUNDAMENTAL_ANALYST = "fundamental_analyst"
    NEWS_ANALYST = "news_analyst"


class SwarmDecision(StrEnum):
    """Possible swarm consensus decisions."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    NO_CONSENSUS = "no_consensus"


@dataclass
class AgentSignal:
    """A signal produced by a single swarm agent."""

    agent_id: str
    role: SwarmRole
    direction: int  # +1 buy, -1 sell, 0 hold
    confidence: float  # 0.0 to 1.0
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


@dataclass
class SwarmConsensus:
    """Aggregated consensus from all swarm agents."""

    decision: SwarmDecision
    weighted_score: float  # -1.0 to +1.0
    agreement_ratio: float  # 0.0 to 1.0
    signals: list[AgentSignal] = field(default_factory=list)
    dissenting_agents: list[str] = field(default_factory=list)
    risk_override: bool = False
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


@dataclass
class SwarmConfig:
    """Configuration for a trading swarm."""

    consensus_threshold: float = 0.6
    risk_veto_enabled: bool = True
    min_agents_for_consensus: int = 3
    # Calibration read-back (CONCEPT:KG-2.27): blend each role's weight toward
    # its agents' tracked calibration (Brier-based reputation) at aggregation
    # time. Safe-by-construction default: with no resolved calls the calibrated
    # weights EQUAL the base weights, so behaviour only shifts once real
    # outcomes are recorded; gate off to pin the static weights regardless.
    calibration_weighting: bool = True
    # EMA-style blend strength toward the calibrated weight (mirrors the
    # capability-index reward-EMA alpha): 0 = static weights, 1 = fully
    # calibrated. Kept moderate so one reputation swing can't flip a vote.
    calibration_alpha: float = 0.3
    role_weights: dict[str, float] = field(
        default_factory=lambda: {
            SwarmRole.DIRECTOR: 1.5,
            SwarmRole.QUANT_ANALYST: 1.2,
            SwarmRole.RISK_MANAGER: 1.3,
            SwarmRole.EXECUTION_TRADER: 1.0,
            SwarmRole.INDICATOR_SPECIALIST: 0.8,
            SwarmRole.PATTERN_ANALYST: 0.8,
            SwarmRole.TREND_ANALYST: 0.9,
            SwarmRole.SENTIMENT_ANALYST: 0.7,
            SwarmRole.BULL_RESEARCHER: 1.1,
            SwarmRole.BEAR_RESEARCHER: 1.1,
            SwarmRole.FUNDAMENTAL_ANALYST: 1.0,
            SwarmRole.NEWS_ANALYST: 0.8,
        }
    )


class SwarmAgent:
    """
    A single agent within a trading swarm.

    Each agent has a role, processes market data according to its specialization,
    and produces directional signals with confidence scores.
    """

    def __init__(self, agent_id: str, role: SwarmRole):
        self.agent_id = agent_id
        self.role = role
        self._signal_history: list[AgentSignal] = []

    def analyze(self, market_data: dict[str, Any]) -> AgentSignal:
        """
        Analyze market data and produce a signal.

        This is the base implementation using simple heuristics per role.
        In production, each role would delegate to specialized models or LLMs.
        """
        direction, confidence, reasoning = self._role_analysis(market_data)

        signal = AgentSignal(
            agent_id=self.agent_id,
            role=self.role,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
        )
        self._signal_history.append(signal)
        return signal

    def _role_analysis(self, data: dict[str, Any]) -> tuple[int, float, str]:
        """Role-specific analysis heuristics."""
        momentum = data.get("momentum", 0.0)
        volatility = data.get("volatility", 0.0)
        rsi = data.get("rsi", 50.0)
        trend = data.get("trend", 0.0)
        volume_signal = data.get("volume_signal", 0.0)

        if self.role == SwarmRole.QUANT_ANALYST:
            # Factor-based analysis
            score = momentum * 0.4 + trend * 0.3 + volume_signal * 0.3
            direction = 1 if score > 0.1 else (-1 if score < -0.1 else 0)
            confidence = min(1.0, abs(score) * 2)
            return direction, confidence, f"Factor score: {score:.3f}"

        elif self.role == SwarmRole.RISK_MANAGER:
            # Risk-first conservative approach
            if volatility > 0.03:
                return 0, 0.8, f"High volatility ({volatility:.3f}), recommending hold"
            if abs(momentum) > 0.05:
                direction = 1 if momentum > 0 else -1
                return direction, 0.5, f"Moderate risk, momentum {momentum:.3f}"
            return 0, 0.6, "Low conviction environment"

        elif self.role == SwarmRole.INDICATOR_SPECIALIST:
            # RSI-based analysis
            if rsi > 70:
                return -1, min(1.0, (rsi - 70) / 30), f"RSI overbought: {rsi:.1f}"
            elif rsi < 30:
                return 1, min(1.0, (30 - rsi) / 30), f"RSI oversold: {rsi:.1f}"
            return 0, 0.3, f"RSI neutral: {rsi:.1f}"

        elif self.role == SwarmRole.TREND_ANALYST:
            # Trend-following
            direction = 1 if trend > 0.01 else (-1 if trend < -0.01 else 0)
            confidence = min(1.0, abs(trend) * 10)
            return direction, confidence, f"Trend strength: {trend:.3f}"

        elif self.role == SwarmRole.PATTERN_ANALYST:
            # Pattern-based (simplified)
            combined = momentum * 0.5 + trend * 0.5
            direction = 1 if combined > 0.05 else (-1 if combined < -0.05 else 0)
            return (
                direction,
                min(1.0, abs(combined) * 5),
                f"Pattern score: {combined:.3f}",
            )

        elif self.role == SwarmRole.DIRECTOR:
            # High-level strategic view
            signals = [momentum, trend, volume_signal]
            avg = sum(signals) / len(signals) if signals else 0
            direction = 1 if avg > 0.02 else (-1 if avg < -0.02 else 0)
            return direction, min(1.0, abs(avg) * 5), f"Strategic view: {avg:.3f}"

        elif self.role == SwarmRole.EXECUTION_TRADER:
            # Execution-focused (timing and liquidity)
            if abs(volume_signal) > 0.5:
                direction = 1 if momentum > 0 else -1
                return (
                    direction,
                    0.7,
                    f"Liquidity available, volume signal: {volume_signal:.2f}",
                )
            return 0, 0.4, "Awaiting liquidity"

        elif self.role == SwarmRole.SENTIMENT_ANALYST:
            sentiment = data.get("sentiment", 0.0)
            direction = 1 if sentiment > 0.3 else (-1 if sentiment < -0.3 else 0)
            return direction, min(1.0, abs(sentiment)), f"Sentiment: {sentiment:.2f}"

        return 0, 0.0, "No analysis"

    @property
    def signal_count(self) -> int:
        return len(self._signal_history)


class TradingSwarm:
    """
    Multi-agent trading swarm that orchestrates specialized agents
    to reach collaborative trading decisions.

    Features:
    - Role-weighted consensus aggregation
    - Risk manager veto power
    - Configurable agreement thresholds
    - Full signal audit trail

    Usage:
        swarm = TradingSwarm.create_default()
        consensus = swarm.analyze({"momentum": 0.05, "rsi": 65, ...})
    """

    def __init__(
        self,
        agents: list[SwarmAgent] | None = None,
        config: SwarmConfig | None = None,
        calibration: Any | None = None,
    ):
        self.agents = agents or []
        self.config = config or SwarmConfig()
        self._consensus_history: list[SwarmConsensus] = []
        # Per-agent reputation tracker (CONCEPT:KG-2.27). Every analyze() logs
        # each agent's directional call; record_market_outcome() resolves them,
        # and the next aggregation blends the resulting calibration into the
        # role weights — the read-back loop that was previously write-only.
        if calibration is None:
            from agent_utilities.domains.finance.calibration_tracker import (
                CalibrationTracker,
            )

            calibration = CalibrationTracker()
        self.calibration = calibration

    @classmethod
    def create_default(cls, config: SwarmConfig | None = None) -> "TradingSwarm":
        """Create a default swarm with one agent per core role."""
        agents = [
            SwarmAgent("director_01", SwarmRole.DIRECTOR),
            SwarmAgent("quant_01", SwarmRole.QUANT_ANALYST),
            SwarmAgent("risk_01", SwarmRole.RISK_MANAGER),
            SwarmAgent("exec_01", SwarmRole.EXECUTION_TRADER),
            SwarmAgent("indicator_01", SwarmRole.INDICATOR_SPECIALIST),
            SwarmAgent("trend_01", SwarmRole.TREND_ANALYST),
        ]
        return cls(agents=agents, config=config)

    def add_agent(self, agent: SwarmAgent) -> None:
        """Add an agent to the swarm."""
        self.agents.append(agent)

    def _effective_role_weights(self) -> dict[str, float]:
        """Role weights for aggregation, calibration-blended when enabled.

        CONCEPT:KG-2.27 read-back: each role's static weight is EMA-blended
        toward :func:`calibrated_role_weights` (its agents' tracked Brier-based
        reputation) so historically-accurate voices count more. Guarded — any
        calibration failure falls back to the configured static weights.
        """
        base = dict(self.config.role_weights)
        if not self.config.calibration_weighting or self.calibration is None:
            return base
        try:
            from agent_utilities.domains.finance.calibration_tracker import (
                calibrated_role_weights,
            )

            calibrated = calibrated_role_weights(
                self.calibration,
                {a.agent_id: a.role for a in self.agents},
                base_weights=base,
            )
            alpha = max(0.0, min(1.0, float(self.config.calibration_alpha)))
            return {
                role: (1.0 - alpha) * base.get(role, 1.0) + alpha * cal
                for role, cal in calibrated.items()
            }
        except Exception as e:  # noqa: BLE001 — never let reputation break a vote
            logger.debug("calibration weighting unavailable: %s", e)
            return base

    def _record_calls(self, signals: list[AgentSignal], subject: str) -> None:
        """Log each directional signal as an open calibration call (KG-2.27)."""
        if self.calibration is None:
            return
        for s in signals:
            if s.direction == 0:
                continue
            try:
                self.calibration.record_call(
                    s.agent_id, direction=s.direction, confidence=s.confidence,
                    subject=subject,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("calibration call record failed: %s", e)

    def record_market_outcome(self, realized_direction: int, subject: str = "") -> int:
        """Resolve every agent's open call against the realized direction.

        The feedback half of the calibration loop (CONCEPT:KG-2.27): call this
        when the market outcome for ``subject`` is known; the very next
        :meth:`analyze` aggregates with reputation-adjusted weights. Returns
        the number of calls resolved.
        """
        if self.calibration is None:
            return 0
        resolved = 0
        for agent in self.agents:
            try:
                if self.calibration.record_outcome(
                    agent.agent_id, realized_direction=realized_direction,
                    subject=subject,
                ):
                    resolved += 1
            except Exception as e:  # noqa: BLE001
                logger.debug("calibration outcome record failed: %s", e)
        return resolved

    def analyze(self, market_data: dict[str, Any]) -> SwarmConsensus:
        """
        Run all agents and aggregate signals into a consensus decision.
        """
        signals = [agent.analyze(market_data) for agent in self.agents]

        # Open a calibration call per directional signal so future outcomes
        # build each agent's reputation (CONCEPT:KG-2.27).
        subject = str(market_data.get("symbol") or market_data.get("ticker") or "")
        self._record_calls(signals, subject)

        if len(signals) < self.config.min_agents_for_consensus:
            return SwarmConsensus(
                decision=SwarmDecision.NO_CONSENSUS,
                weighted_score=0.0,
                agreement_ratio=0.0,
                signals=signals,
            )

        # Weighted score aggregation (calibration-blended weights, KG-2.27)
        role_weights = self._effective_role_weights()
        total_weight = 0.0
        weighted_sum = 0.0
        for signal in signals:
            weight = role_weights.get(signal.role, 1.0)
            weighted_sum += signal.direction * signal.confidence * weight
            total_weight += weight

        weighted_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Agreement ratio
        if weighted_score > 0:
            agreeing = sum(1 for s in signals if s.direction > 0)
        elif weighted_score < 0:
            agreeing = sum(1 for s in signals if s.direction < 0)
        else:
            agreeing = sum(1 for s in signals if s.direction == 0)
        agreement_ratio = agreeing / len(signals) if signals else 0.0

        # Risk manager veto check
        risk_override = False
        if self.config.risk_veto_enabled:
            risk_signals = [s for s in signals if s.role == SwarmRole.RISK_MANAGER]
            for rs in risk_signals:
                if rs.direction == 0 and rs.confidence > 0.7:
                    risk_override = True
                    logger.warning(f"Risk manager veto: {rs.reasoning}")

        # Determine decision
        if risk_override:
            decision = SwarmDecision.HOLD
        elif abs(weighted_score) < 0.1:
            decision = SwarmDecision.HOLD
        elif weighted_score > 0.5:
            decision = SwarmDecision.STRONG_BUY
        elif weighted_score > 0.1:
            decision = SwarmDecision.BUY
        elif weighted_score < -0.5:
            decision = SwarmDecision.STRONG_SELL
        elif weighted_score < -0.1:
            decision = SwarmDecision.SELL
        else:
            decision = SwarmDecision.HOLD

        # Identify dissenters
        majority_dir = 1 if weighted_score > 0 else (-1 if weighted_score < 0 else 0)
        dissenters = [
            s.agent_id
            for s in signals
            if s.direction != majority_dir and s.direction != 0
        ]

        consensus = SwarmConsensus(
            decision=decision,
            weighted_score=float(weighted_score),
            agreement_ratio=float(agreement_ratio),
            signals=signals,
            dissenting_agents=dissenters,
            risk_override=risk_override,
        )
        self._consensus_history.append(consensus)
        return consensus

    def get_consensus_history(self) -> list[SwarmConsensus]:
        return list(self._consensus_history)

    @property
    def agent_count(self) -> int:
        return len(self.agents)

    @property
    def roles(self) -> list[SwarmRole]:
        return [a.role for a in self.agents]
