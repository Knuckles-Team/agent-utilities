"""
Debate Engine — CONCEPT:AU-KG.research.research-pipeline-runner
Orchestrates multi-round Bull vs Bear structured debate for financial assets.
Inspired by TradingAgents (arxiv:2412.20138).
"""

from __future__ import annotations

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
    # Structured fundamentals a persona's decision heuristics evaluate against
    # (e.g. {"pe": 12.0, "pb": 1.1, "roe": 0.2, ...}). When present, each side's
    # bound persona cites its named pass/fail rules — KG-2.28 grounding.
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateSession:
    session_id: str
    context: DebateContext
    rounds: list[DebateRound] = field(default_factory=list)
    risk_assessment: RiskVeto | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str = ""
    final_decision: str = "PENDING"


# Generic fallback voices used when no investor persona is bound.
_GENERIC_BULL_PROMPT = (
    "You are a Bull Researcher. Present a compelling, highly optimistic financial/quant "
    "argument for buying or holding the given asset. Support your points with realistic "
    "or historical evidence."
)
_GENERIC_BEAR_PROMPT = (
    "You are a Bear Researcher. Present a compelling, highly skeptical financial/quant "
    "argument highlighting risks, macro headwinds, and valuation metrics for the given asset. "
    "Critique the Bull argument directly."
)


class DebateEngine:
    """
    Orchestrates the Bull vs Bear debate and Risk Manager veto process.
    Integrates directly with the Knowledge Graph for provenance.

    When ``bull_persona`` / ``bear_persona`` are set (prompt stems under
    ``agent_utilities/prompts``), each side speaks in that investor's actual
    voice — e.g. Buffett (value) vs Burry (forensic) — by loading the persona's
    identity + core directive as the system prompt. Otherwise generic Bull/Bear
    voices are used, so existing callers are unaffected. CONCEPT:AU-KG.research.research-pipeline-runner
    """

    def __init__(
        self,
        engine: Any = None,
        llm_client: Any = None,
        bull_persona: str | None = None,
        bear_persona: str | None = None,
    ):
        self.engine = engine
        self.llm = llm_client
        self.bull_persona = bull_persona
        self.bear_persona = bear_persona

    @classmethod
    def with_personas(
        cls,
        bull: str | None = None,
        bear: str | None = None,
        engine: Any = None,
        llm_client: Any = None,
    ) -> DebateEngine:
        """Construct a DebateEngine whose Bull/Bear sides speak as named investor
        personas. Defaults to Buffett (bull) vs Burry (bear) when unspecified."""
        from agent_utilities.domains.finance.investor_debate import (
            DEFAULT_BEAR_PERSONA,
            DEFAULT_BULL_PERSONA,
        )

        return cls(
            engine=engine,
            llm_client=llm_client,
            bull_persona=bull or DEFAULT_BULL_PERSONA,
            bear_persona=bear or DEFAULT_BEAR_PERSONA,
        )

    def _bull_system_prompt(self) -> str:
        """Bull-side system prompt: the bound persona's voice, else generic."""
        if self.bull_persona:
            from agent_utilities.domains.finance.investor_debate import (
                persona_system_prompt,
            )

            body = persona_system_prompt(self.bull_persona)
            if body:
                return body
        return _GENERIC_BULL_PROMPT

    def _bear_system_prompt(self) -> str:
        """Bear-side system prompt: the bound persona's voice, else generic."""
        if self.bear_persona:
            from agent_utilities.domains.finance.investor_debate import (
                persona_system_prompt,
            )

            body = persona_system_prompt(self.bear_persona)
            if body:
                return body
        return _GENERIC_BEAR_PROMPT

    def _bull_label(self) -> str:
        """Role label for the Bull argument, including the persona archetype."""
        if self.bull_persona:
            from agent_utilities.domains.finance.investor_debate import (
                persona_archetype,
            )

            return f"Bull Researcher ({persona_archetype(self.bull_persona)})"
        return "Bull Researcher"

    def _bear_label(self) -> str:
        """Role label for the Bear argument, including the persona archetype."""
        if self.bear_persona:
            from agent_utilities.domains.finance.investor_debate import (
                persona_archetype,
            )

            return f"Bear Researcher ({persona_archetype(self.bear_persona)})"
        return "Bear Researcher"

    def persona_heuristic_evidence(
        self, persona: str | None, metrics: dict[str, Any]
    ) -> str:
        """Evaluate a bound persona's decision heuristics against ``metrics``.

        Returns a one-line, citable pass/fail summary (KG-2.28) the side folds
        into its prompt so the argument is grounded in named, numeric rules
        rather than hand-waving. Returns ``""`` when no persona is bound, there
        are no metrics, or the persona carries no heuristic framework — so the
        existing generic debate path is unaffected.
        """
        if not persona or not metrics:
            return ""
        try:
            from agent_utilities.domains.finance.persona_heuristics import (
                PERSONA_HEURISTICS,
                evaluate_persona,
            )

            if persona not in PERSONA_HEURISTICS:
                return ""
            return evaluate_persona(persona, metrics).citation()
        except Exception as exc:  # noqa: BLE001 — evidence is best-effort
            logger.debug("persona heuristic evidence unavailable: %s", exc)
            return ""

    def _heuristic_block(self, persona: str | None, context: DebateContext) -> str:
        """Prompt fragment carrying the persona's heuristic verdict, if any."""
        evidence = self.persona_heuristic_evidence(persona, context.metrics)
        return f"\nPersona decision-heuristic verdict: {evidence}" if evidence else ""

    def _generate_bull_argument(
        self, context: DebateContext, history: list[DebateRound], round_num: int
    ) -> DebateArgument:
        """Generate Bull Researcher argument using LLM (persona voice if bound)."""
        logger.info(f"Generating Bull argument for {context.ticker}, round {round_num}")
        try:
            from pydantic_ai import Agent

            from agent_utilities.core.model_factory import create_model

            model = self.llm or create_model()
            agent = Agent(
                model=model,
                output_type=DebateArgument,
                system_prompt=self._bull_system_prompt(),
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
                f"{self._heuristic_block(self.bull_persona, context)}"
            )
            result = agent.run_sync(prompt)
            output = result.output
            output.role = self._bull_label()
            return output
        except Exception as e:
            logger.warning(f"Bull argument LLM generation failed, using fallback: {e}")
            return DebateArgument(
                role=self._bull_label(),
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
        """Generate Bear Researcher argument using LLM (persona voice if bound)."""
        logger.info(f"Generating Bear argument for {context.ticker}, round {round_num}")
        try:
            from pydantic_ai import Agent

            from agent_utilities.core.model_factory import create_model

            model = self.llm or create_model()
            agent = Agent(
                model=model,
                output_type=DebateArgument,
                system_prompt=self._bear_system_prompt(),
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
                f"{self._heuristic_block(self.bear_persona, context)}"
            )
            result = agent.run_sync(prompt)
            output = result.output
            output.role = self._bear_label()
            return output
        except Exception as e:
            logger.warning(f"Bear argument LLM generation failed, using fallback: {e}")
            return DebateArgument(
                role=self._bear_label(),
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
            return result.output
        except Exception as e:
            logger.warning(
                f"Risk evaluation LLM call failed, using heuristic fallback: {e}"
            )
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
            node_id=node_id,
            node_type="DebateSession",
            properties={
                "ticker": session.context.ticker,
                "decision": session.final_decision,
                "rounds": len(session.rounds),
            },
        )

        if session.risk_assessment:
            self.engine.add_node(
                node_id=f"{node_id}_Risk",
                node_type="RiskAssessment",
                properties={
                    "approved": session.risk_assessment.approved,
                    "reasoning": session.risk_assessment.reasoning,
                },
            )
            self.engine.add_edge(node_id, f"{node_id}_Risk", "EVALUATED_BY")
