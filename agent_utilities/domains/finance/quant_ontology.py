"""
Quant Ontology — CONCEPT:AU-KG.research.research-pipeline-runner
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
        """Register the FIBO-aligned quant ontology schemas in the engine.

        Dead-code gotcha found alongside the ``add_node(id=...)`` kwarg-drift
        bug (kg-exhaustive-batchD.md): every call here used to pass
        ``properties=[...]`` — a bare list of field-name strings — where
        ``IntelligenceGraphEngine.add_node`` requires ``properties:
        dict[str, Any] | None`` (it does ``props["type"] = node_type`` on
        whatever is passed, which raises on a list). Not currently reachable
        (nothing in the repo calls ``QuantOntology.register``), but fixed
        defensively so it doesn't crash the moment something wires it up.
        Each schema's field-name list is now nested under a ``fields`` key.
        """

        # 1. Strategy & Hypothesis Lifecycle
        engine.add_node(
            "Schema:TradingHypothesis",
            "OntologySchema",
            properties={
                "fields": ["id", "title", "expected_edge", "asset_class", "status"]
            },
        )
        engine.add_node(
            "Schema:BacktestResult",
            "OntologySchema",
            properties={
                "fields": [
                    "id",
                    "sharpe",
                    "max_drawdown",
                    "win_rate",
                    "profit_factor",
                ]
            },
        )
        engine.add_node(
            "Schema:TradingStrategy",
            "OntologySchema",
            properties={
                "fields": ["id", "name", "version", "status", "regime_preference"]
            },
        )

        # 2. Market State & Alpha
        engine.add_node(
            "Schema:MarketRegime",
            "OntologySchema",
            properties={"fields": ["id", "regime_type", "confidence", "start_time"]},
        )
        engine.add_node(
            "Schema:AlphaFactor",
            "OntologySchema",
            properties={"fields": ["id", "name", "ic_mean", "ic_ir", "turnover"]},
        )
        engine.add_node(
            "Schema:TradingSignal",
            "OntologySchema",
            properties={"fields": ["id", "asset", "direction", "conviction", "source"]},
        )

        # 3. Execution & Portfolio
        engine.add_node(
            "Schema:Order",
            "OntologySchema",
            properties={
                "fields": [
                    "id",
                    "asset",
                    "type",
                    "side",
                    "qty",
                    "limit_price",
                    "status",
                ]
            },
        )
        engine.add_node(
            "Schema:Portfolio",
            "OntologySchema",
            properties={
                "fields": ["id", "nav", "cash", "margin_utilization", "var_95"]
            },
        )

        # 4. Prediction Markets & Forecasts
        engine.add_node(
            "Schema:PredictionMarket",
            "OntologySchema",
            properties={
                "fields": [
                    "id",
                    "platform",
                    "event",
                    "implied_probability",
                    "settlement_time",
                ]
            },
        )
        engine.add_node(
            "Schema:EnsembleForecast",
            "OntologySchema",
            properties={
                "fields": [
                    "id",
                    "model_family",
                    "member_count",
                    "mean",
                    "probability",
                    "target_event",
                ]
            },
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
