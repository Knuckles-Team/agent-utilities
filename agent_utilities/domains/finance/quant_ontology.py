"""
Quant Ontology — CONCEPT:KG-2.6
FIBO-aligned Knowledge Graph ontology for quantitative finance primitives.
"""

from enum import StrEnum
from typing import Any


class MarketRegime(StrEnum):
    BULL = "bull_market"
    BEAR = "bear_market"
    SIDEWAYS = "sideways_market"
    CRISIS = "crisis_market"
    HIGH_VOLATILITY = "high_volatility"


class QuantOntology:
    """
    Registers and manages the quant finance ontology schemas in the Knowledge Graph.
    Ensures standard node types (Signal, Order, Strategy, Factor) and relationships.
    """

    @staticmethod
    def register(engine: Any) -> None:
        """Register the FIBO-aligned quant ontology schemas in the engine."""

        # 1. Strategy & Hypothesis Lifecycle
        engine.add_node(
            "Schema:TradingHypothesis",
            "OntologySchema",
            properties=["id", "title", "expected_edge", "asset_class", "status"],
        )
        engine.add_node(
            "Schema:BacktestResult",
            "OntologySchema",
            properties=["id", "sharpe", "max_drawdown", "win_rate", "profit_factor"],
        )
        engine.add_node(
            "Schema:TradingStrategy",
            "OntologySchema",
            properties=["id", "name", "version", "status", "regime_preference"],
        )

        # 2. Market State & Alpha
        engine.add_node(
            "Schema:MarketRegime",
            "OntologySchema",
            properties=["id", "regime_type", "confidence", "start_time"],
        )
        engine.add_node(
            "Schema:AlphaFactor",
            "OntologySchema",
            properties=["id", "name", "ic_mean", "ic_ir", "turnover"],
        )
        engine.add_node(
            "Schema:TradingSignal",
            "OntologySchema",
            properties=["id", "asset", "direction", "conviction", "source"],
        )

        # 3. Execution & Portfolio
        engine.add_node(
            "Schema:Order",
            "OntologySchema",
            properties=["id", "asset", "type", "side", "qty", "limit_price", "status"],
        )
        engine.add_node(
            "Schema:Portfolio",
            "OntologySchema",
            properties=["id", "nav", "cash", "margin_utilization", "var_95"],
        )

        # 4. Prediction Markets & Forecasts
        engine.add_node(
            "Schema:PredictionMarket",
            "OntologySchema",
            properties=[
                "id",
                "platform",
                "event",
                "implied_probability",
                "settlement_time",
            ],
        )
        engine.add_node(
            "Schema:EnsembleForecast",
            "OntologySchema",
            properties=[
                "id",
                "model_family",
                "member_count",
                "mean",
                "probability",
                "target_event",
            ],
        )

        # Define formal relationships
        relations = [
            ("TradingHypothesis", "TESTED_IN", "BacktestResult"),
            ("BacktestResult", "VALIDATES", "TradingStrategy"),
            ("TradingStrategy", "GENERATES_SIGNAL", "TradingSignal"),
            ("AlphaFactor", "DRIVES", "TradingSignal"),
            ("TradingSignal", "EXECUTES_AS", "Order"),
            ("MarketRegime", "INFLUENCES", "TradingStrategy"),
            ("Order", "BELONGS_TO", "Portfolio"),
            # Prediction Market Relations
            ("EnsembleForecast", "DRIVES", "TradingSignal"),
            ("TradingSignal", "TARGETS", "PredictionMarket"),
            ("PredictionMarket", "PRICES", "TradingHypothesis"),
        ]

        for source, rel, target in relations:
            engine.add_edge(
                f"Schema:{source}",
                f"Schema:{target}",
                "ALLOWED_RELATION",
                relation_type=rel,
            )
