"""
Strategy Engine — CONCEPT:KG-2.6
Lifecycle tracking for trading strategies (Draft -> Backtest -> Paper -> Live).
Inspired by qlib workflow orchestration and freqtrade strategy patterns.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class StrategyStage(StrEnum):
    DRAFT = "draft"
    BACKTESTING = "backtesting"
    PAPER_TRADING = "paper_trading"
    LIVE = "live"
    RETIRED = "retired"


@dataclass
class StrategyMetrics:
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0


class StrategyEngine:
    """
    Manages the lifecycle of trading strategies, ensuring they pass
    configurable gates before promotion.
    """

    def __init__(self, engine: Any):
        self.engine = engine

    def register_strategy(self, name: str, code_ref: str, author: str) -> str:
        """Register a new strategy draft."""
        strategy_id = f"Strat_{name.replace(' ', '_')}"

        self.engine.add_node(
            id=strategy_id,
            node_type="TradingStrategy",
            name=name,
            version="1.0.0",
            status=StrategyStage.DRAFT,
            code_ref=code_ref,
            author=author,
        )
        logger.info(f"Registered new strategy: {strategy_id}")
        return strategy_id

    def record_backtest(self, strategy_id: str, metrics: StrategyMetrics) -> bool:
        """Record backtest results and evaluate for promotion to paper trading."""

        # Link backtest to strategy
        bt_id = f"BT_{strategy_id}_{hash(str(metrics.sharpe))}"
        self.engine.add_node(
            id=bt_id,
            node_type="BacktestResult",
            sharpe=metrics.sharpe,
            max_drawdown=metrics.max_drawdown,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
        )
        self.engine.add_edge(bt_id, strategy_id, "VALIDATES")

        # Promotion gate heuristics
        promotable = (
            metrics.sharpe > 1.0
            and metrics.max_drawdown > -0.15
            and metrics.total_trades > 50
        )

        if promotable:
            self.engine.query_cypher(
                "MATCH (s:TradingStrategy {id: $sid}) SET s.status = $status",
                {"sid": strategy_id, "status": StrategyStage.PAPER_TRADING},
            )
            logger.info(f"Strategy {strategy_id} promoted to PAPER_TRADING")

        return promotable
