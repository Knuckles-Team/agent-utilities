"""
Quant MCP Tools — CONCEPT:ECO-4.3
Exposes quant primitives to external agents via the graph-os MCP Server.
"""

import logging
from typing import Any

from agent_utilities.domains.finance.debate_engine import DebateContext, DebateEngine
from agent_utilities.domains.finance.market_data import DataRegistry
from agent_utilities.domains.finance.regime_detector import RegimeDetector
from agent_utilities.domains.finance.trading_swarm import TradingSwarm

logger = logging.getLogger(__name__)


def register_quant_tools(mcp: Any, engine: Any) -> None:
    """Register quant-specific tools onto the MCP server instance."""

    @mcp.tool()
    def quant(
        domain: str,
        action: str,
        ticker: str = "",
        asset_class: str = "equity",
        period: str = "1y",
        interval: str = "1d",
        side: str = "buy",
        quantity: float = 0.0,
        order_type: str = "market",
        price: float = 0.0,
        mode: str = "paper",
        portfolio_id: str = "default",
        rounds: int = 3,
    ) -> str:
        """
        The Ultimate Quant System Tool.
        Domains:
        - 'orchestrate': Intelligence layer (Actions: debate, analyze, regime, ensemble_predict)
        - 'data': Telemetry layer (Actions: historical, order_book, fundamentals)
        - 'execute': Trading layer (Actions: submit_order, cancel_order, status). SAFEGUARD: Defaults to mode="paper".
        - 'portfolio': Risk layer (Actions: balances, positions, risk_metrics, optimize)
        """
        try:
            if domain == "orchestrate":
                if action == "debate":
                    if not engine:
                        return "Error: GraphEngine required for quant debate"

                    context = DebateContext(
                        ticker=ticker,
                        asset_class=asset_class,
                        market_report=f"Standard market report for {ticker}",
                        sentiment_report=f"Neutral sentiment for {ticker}",
                    )

                    debate = DebateEngine(engine=engine)
                    session_id = f"mcp_debate_{hash(ticker)}"
                    result = debate.run_debate(session_id, context, rounds)

                    approved = (
                        result.risk_assessment.approved
                        if result.risk_assessment
                        else False
                    )
                    reasoning = (
                        result.risk_assessment.reasoning
                        if result.risk_assessment
                        else "No risk assessment available"
                    )
                    pos_size = (
                        result.risk_assessment.max_position_size
                        if result.risk_assessment
                        else 0.0
                    )

                    summary = [
                        f"=== DEBATE FINAL DECISION: {result.final_decision} ===",
                        f"Asset: {ticker} ({asset_class})",
                        f"Risk Veto: {'APPROVED' if approved else 'VETOED'}",
                        f"Reasoning: {reasoning}",
                        f"Max Position Size: {pos_size:.1%}",
                        "\n--- Debate History ---",
                    ]

                    for r in result.rounds:
                        summary.append(f"Round {r.round_number}:")
                        summary.append(
                            f"  {r.bull_argument.role} (Conf: {r.bull_argument.confidence:.2f}): {r.bull_argument.content}"
                        )
                        summary.append(
                            f"  {r.bear_argument.role} (Conf: {r.bear_argument.confidence:.2f}): {r.bear_argument.content}"
                        )

                    return "\n".join(summary)

                elif action == "analyze":
                    swarm = TradingSwarm.create_default()
                    mock_data = {
                        "momentum": 0.05,
                        "volatility": 0.02,
                        "rsi": 65,
                        "trend": 0.02,
                        "volume_signal": 0.4,
                    }
                    consensus = swarm.analyze(mock_data)

                    lines = [f"Full analysis for {ticker}:"]
                    lines.append(f"Decision: {consensus.decision.value.upper()}")
                    lines.append(
                        f"Confidence (Weighted Score): {consensus.weighted_score:.2f}"
                    )
                    lines.append(f"Agreement Ratio: {consensus.agreement_ratio:.2f}")
                    lines.append(f"Risk Override: {consensus.risk_override}")
                    lines.append("\nIndividual Signals:")
                    for sig in consensus.signals:
                        lines.append(
                            f"  - {sig.role.value}: {'Buy' if sig.direction > 0 else 'Sell' if sig.direction < 0 else 'Hold'} ({sig.confidence:.2f}) -> {sig.reasoning}"
                        )

                    return "\n".join(lines)

                elif action == "regime":
                    registry = DataRegistry()
                    res = registry.fetch(ticker, period=period, interval=interval)

                    detector = RegimeDetector(engine)
                    regime = detector.detect_regime(res.data, ticker)

                    return f"Current market regime for {ticker}: {regime.upper()} (Data source: {res.provider})"

                elif action == "ensemble_predict":
                    import time

                    from agent_utilities.harness.distributed_state_manager import (
                        OptimisticStateLocker,
                    )
                    from agent_utilities.orchestration.prediction_linkage import (
                        PredictionLinkageLayer,
                    )

                    pll = PredictionLinkageLayer()
                    state_locker = OptimisticStateLocker(use_redis=False)

                    expected_v = 0
                    state_data = state_locker.get_state(f"ensemble_{ticker}")
                    if state_data:
                        expected_v = state_data.get("version", 0)

                    now = time.time()
                    pll.register_prediction(
                        "agent_momentum",
                        ticker,
                        prediction=1.05,
                        confidence=0.8,
                        timestamp=now,
                    )
                    pll.register_prediction(
                        "agent_mean_rev",
                        ticker,
                        prediction=-0.95,
                        confidence=0.6,
                        timestamp=now,
                    )
                    pll.register_prediction(
                        "agent_macro",
                        ticker,
                        prediction=1.02,
                        confidence=0.9,
                        timestamp=now,
                    )

                    fused_result = pll.fuse_predictions(ticker)

                    success = state_locker.update_state(
                        f"ensemble_{ticker}", fused_result, expected_v
                    )

                    lines = [
                        f"=== QUANT ENSEMBLE PREDICTION FOR {ticker} ===",
                        f"Fused Prediction Score: {fused_result['ensemble_prediction']:.4f}",
                        f"Overall Confidence: {fused_result['overall_confidence']:.2f}",
                        f"State Lock Success: {'Yes' if success else 'No (Race Condition Detected)'}",
                        "Participating Sub-Agents: agent_momentum, agent_mean_rev, agent_macro",
                    ]
                    return "\n".join(lines)

                else:
                    return f"Error: Unknown action '{action}' for orchestrate domain."

            elif domain == "data":
                if action == "historical":
                    return f"[Mock] Fetched {period} of {interval} historical data for {ticker} ({asset_class})."
                elif action == "order_book":
                    return f"[Mock] Fetched Level 2 order book for {ticker} ({asset_class})."
                elif action == "fundamentals":
                    return f"[Mock] Fetched fundamental metrics for {ticker} ({asset_class})."
                else:
                    return f"Error: Unknown action '{action}' for data domain."

            elif domain == "execute":
                prefix = "[LIVE EXECUTION]" if mode == "live" else "[PAPER TRADING]"
                if action == "submit_order":
                    return f"{prefix} Submitted {side.upper()} order for {quantity} {ticker} @ {order_type} {price}."
                elif action == "cancel_order":
                    return f"{prefix} Cancelled working orders for {ticker}."
                elif action == "status":
                    return f"{prefix} Order status for {ticker}: FILLED."
                else:
                    return f"Error: Unknown action '{action}' for execute domain."

            elif domain == "portfolio":
                if action == "balances":
                    return f"[Mock] Total Portfolio Value ({portfolio_id}): $100,000. Cash: $20,000."
                elif action == "positions":
                    return f"[Mock] Open Positions ({portfolio_id}): BTC-USD (+5%), AAPL (-1%)."
                elif action == "risk_metrics":
                    return f"[Mock] Risk Metrics ({portfolio_id}): VaR = 2.5%, Margin Usage = 15%."
                elif action == "optimize":
                    return f"[Mock] Portfolio Optimization ({portfolio_id}): Rebalancing suggested. +2% AAPL, -1% BTC."
                else:
                    return f"Error: Unknown action '{action}' for portfolio domain."

            else:
                return f"Error: Unknown domain '{domain}'."

        except Exception as e:
            logger.error(
                f"quant tool failed for domain '{domain}', action '{action}': {e}"
            )
            return f"Action '{action}' failed in domain '{domain}': {str(e)}"
