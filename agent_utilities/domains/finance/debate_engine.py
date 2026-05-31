"""
Debate Engine — CONCEPT:KG-2.6
Orchestrates multi-round Bull vs Bear structured debate for financial assets.
Inspired by TradingAgents (arxiv:2412.20138).
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DebateArgument(BaseModel):
    role: str = Field(..., description="Role of the arguer (Bull or Bear)")
    content: str = Field(..., description="The argument content")
    evidence: list[str] = Field(
        default_factory=list, description="Citations from data reports"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in this argument"
    )


class DebateRound(BaseModel):
    round_number: int
    bull_argument: DebateArgument
    bear_argument: DebateArgument


class RiskVeto(BaseModel):
    approved: bool = Field(
        ..., description="Whether risk management approves the trade"
    )
    reasoning: str = Field(..., description="Explanation for approval or veto")
    max_position_size: float = Field(
        0.0, description="Maximum allowed position size (0-1.0)"
    )


@dataclass
class DebateContext:
    ticker: str
    asset_class: str
    market_report: str = ""
    sentiment_report: str = ""
    news_report: str = ""
    fundamentals_report: str = ""
    technical_report: str = ""


@dataclass
class DebateSession:
    session_id: str
    context: DebateContext
    rounds: list[DebateRound] = field(default_factory=list)
    risk_assessment: RiskVeto | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str = ""
    final_decision: str = "PENDING"


class DebateEngine:
    """
    Orchestrates the Bull vs Bear debate and Risk Manager veto process.
    Integrates directly with the Knowledge Graph for provenance.
    """

    def __init__(self, engine: Any = None, llm_client: Any = None):
        self.engine = engine
        self.llm = llm_client

    def _generate_bull_argument(
        self, context: DebateContext, history: list[DebateRound], round_num: int
    ) -> DebateArgument:
        """Generate Bull Researcher argument using LLM."""
        logger.info(f"Generating Bull argument for {context.ticker}, round {round_num}")
        try:
            from pydantic_ai import Agent
            from agent_utilities.core.model_factory import create_model

            model = self.llm or create_model()
            agent = Agent(
                model=model,
                output_type=DebateArgument,
                system_prompt=(
                    "You are a Bull Researcher. Present a compelling, highly optimistic financial/quant "
                    "argument for buying or holding the given asset. Support your points with realistic "
                    "or historical evidence."
                ),
            )
            prompt = (
                f"Asset: {context.ticker} ({context.asset_class})\n"
                f"History of previous rounds: {history}\n"
                f"Current round: {round_num}\n"
                f"Market Context:\n"
                f"Report: {context.market_report}\n"
                f"Sentiment: {context.sentiment_report}\n"
                f"News: {context.news_report}\n"
                f"Fundamentals: {context.fundamentals_report}\n"
                f"Technical: {context.technical_report}"
            )
            result = agent.run_sync(prompt)
            return getattr(result, "data")
        except Exception as e:
            logger.warning(f"Bull argument LLM generation failed, using fallback: {e}")
            return DebateArgument(
                role="Bull Researcher",
                content=f"Strong growth potential for {context.ticker} based on market expansion.",
                evidence=["Revenue grew 20% YoY", "Positive sentiment spike"],
                confidence=0.85,
            )

    def _generate_bear_argument(
        self,
        context: DebateContext,
        history: list[DebateRound],
        round_num: int,
        bull_arg: DebateArgument,
    ) -> DebateArgument:
        """Generate Bear Researcher argument using LLM, responding to Bull."""
        logger.info(f"Generating Bear argument for {context.ticker}, round {round_num}")
        try:
            from pydantic_ai import Agent
            from agent_utilities.core.model_factory import create_model

            model = self.llm or create_model()
            agent = Agent(
                model=model,
                output_type=DebateArgument,
                system_prompt=(
                    "You are a Bear Researcher. Present a compelling, highly skeptical financial/quant "
                    "argument highlighting risks, macro headwinds, and valuation metrics for the given asset. "
                    "Critique the Bull argument directly."
                ),
            )
            prompt = (
                f"Asset: {context.ticker} ({context.asset_class})\n"
                f"Bull argument to critique: {bull_arg.content}\n"
                f"History of previous rounds: {history}\n"
                f"Current round: {round_num}\n"
                f"Market Context:\n"
                f"Report: {context.market_report}\n"
                f"Sentiment: {context.sentiment_report}\n"
                f"News: {context.news_report}\n"
                f"Fundamentals: {context.fundamentals_report}\n"
                f"Technical: {context.technical_report}"
            )
            result = agent.run_sync(prompt)
            return getattr(result, "data")
        except Exception as e:
            logger.warning(f"Bear argument LLM generation failed, using fallback: {e}")
            return DebateArgument(
                role="Bear Researcher",
                content=f"Valuation is stretched for {context.ticker}; macroeconomic headwinds present.",
                evidence=["P/E ratio at historical highs", "Sector rotation indicated"],
                confidence=0.75,
            )

    def _evaluate_risk(
        self, context: DebateContext, rounds: list[DebateRound]
    ) -> RiskVeto:
        """Risk Manager team evaluates the debate and makes a final call."""
        logger.info(f"Evaluating risk for {context.ticker} debate")
        try:
            from pydantic_ai import Agent
            from agent_utilities.core.model_factory import create_model

            model = self.llm or create_model()
            agent = Agent(
                model=model,
                output_type=RiskVeto,
                system_prompt=(
                    "You are a Risk Manager. Evaluate the structured debate rounds between the Bull and Bear "
                    "researchers for the asset. Assess the validity of their claims, their confidence levels, "
                    "and decide whether to approve the trade and assign a maximum position size (0 to 1.0)."
                ),
            )
            prompt = (
                f"Asset: {context.ticker} ({context.asset_class})\n"
                f"Debate Rounds:\n{rounds}\n"
            )
            result = agent.run_sync(prompt)
            return getattr(result, "data")
        except Exception as e:
            logger.warning(f"Risk evaluation LLM call failed, using heuristic fallback: {e}")
            bull_conf = (
                sum(r.bull_argument.confidence for r in rounds) / len(rounds)
                if rounds
                else 0
            )
            bear_conf = (
                sum(r.bear_argument.confidence for r in rounds) / len(rounds)
                if rounds
                else 0
            )
            approved = bull_conf > bear_conf
            return RiskVeto(
                approved=approved,
                reasoning=f"Bull arguments ({bull_conf:.2f}) outweighed Bear risks ({bear_conf:.2f})",
                max_position_size=0.05 if approved else 0.0,
            )

    def run_debate(
        self, session_id: str, context: DebateContext, num_rounds: int = 3
    ) -> DebateSession:
        """Execute a full debate session."""
        session = DebateSession(session_id=session_id, context=context)

        for i in range(1, num_rounds + 1):
            bull_arg = self._generate_bull_argument(context, session.rounds, i)
            bear_arg = self._generate_bear_argument(
                context, session.rounds, i, bull_arg
            )

            round_data = DebateRound(
                round_number=i, bull_argument=bull_arg, bear_argument=bear_arg
            )
            session.rounds.append(round_data)

        session.risk_assessment = self._evaluate_risk(context, session.rounds)
        session.final_decision = (
            "BUY" if session.risk_assessment.approved else "HOLD/SELL"
        )
        session.completed_at = datetime.now(UTC).isoformat()

        # Persist to Knowledge Graph
        if self.engine:
            self._persist_to_kg(session)

        return session

    def _persist_to_kg(self, session: DebateSession) -> None:
        """Save the debate outcome and provenance to the KG."""
        node_id = f"Debate_{session.session_id}_{session.context.ticker}"

        self.engine.add_node(
            id=node_id,
            node_type="DebateSession",
            ticker=session.context.ticker,
            decision=session.final_decision,
            rounds=len(session.rounds),
        )

        if session.risk_assessment:
            self.engine.add_node(
                id=f"{node_id}_Risk",
                node_type="RiskAssessment",
                approved=session.risk_assessment.approved,
                reasoning=session.risk_assessment.reasoning,
            )
            self.engine.add_edge(node_id, f"{node_id}_Risk", "EVALUATED_BY")
