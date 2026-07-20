from __future__ import annotations

"""Finance Schema Pack — financial services domain profile.

CONCEPT:AU-KG.research.research-pipeline-runner, CONCEPT:AU-KG.query.vendor-agnostic-traversal, CONCEPT:AU-AHE.harness.self-improvement-overview

Optimized for financial agents managing instruments, transactions,
accounts, regulations, trading pipelines, risk assessment, and
backtesting. Aligned with FIBO ontology.

Expanded with trading pipeline primitives (KG-2.6), risk scoring
ontology (KG-2.7), and backtest evaluation nodes (AHE-3.8).
"""


from ..knowledge_graph import RegistryEdgeType, RegistryNodeType
from ..schema_pack import BacklinkBoostStrategy, SchemaPack, SchemaPackMode


class FinanceSchemaPack(SchemaPack):
    """Finance domain pack: instruments, trading, risk, and backtesting.

    CONCEPT:AU-KG.research.research-pipeline-runner, CONCEPT:AU-KG.query.vendor-agnostic-traversal, CONCEPT:AU-AHE.harness.self-improvement-overview

    Activates the full financial domain including:
    - Core financial entities (instruments, transactions, accounts)
    - Trading pipeline (signals, orders, positions, portfolios, strategies)
    - Risk scoring (assessments, factors, mitigations)
    - Backtest evaluation (runs, metrics)
    """

    name: str = "finance"
    description: str = (
        "Financial services profile. Activates financial instrument, "
        "transaction, account, regulation, trading pipeline, risk scoring, "
        "and backtesting types. Aligned with FIBO ontology for securities, "
        "compliance, regulatory tracking, and quantitative analysis."
    )
    mode: SchemaPackMode = SchemaPackMode.ADDITIVE
    node_types: set[RegistryNodeType] = {
        # Core financial entities
        RegistryNodeType.FINANCIAL_INSTRUMENT,
        RegistryNodeType.FINANCIAL_TRANSACTION,
        RegistryNodeType.ACCOUNT,
        RegistryNodeType.REGULATION,
        RegistryNodeType.ORGANIZATION,
        RegistryNodeType.PERSON,
        RegistryNodeType.DOCUMENT,
        RegistryNodeType.DECISION,
        RegistryNodeType.EVIDENCE,
        RegistryNodeType.SYSTEM,
        # Trading pipeline (CONCEPT:AU-KG.research.research-pipeline-runner)
        RegistryNodeType.TRADING_SIGNAL,
        RegistryNodeType.ORDER,
        RegistryNodeType.POSITION,
        RegistryNodeType.PORTFOLIO,
        RegistryNodeType.STRATEGY,
        RegistryNodeType.TIME_SERIES_FORECAST,
        RegistryNodeType.VERSIONED_TRADE_COMMIT,
        RegistryNodeType.EXECUTION_GUARD,
        RegistryNodeType.UNIFIED_TRADING_ACCOUNT,
        # Data connectors (CONCEPT:AU-ECO.messaging.native-backend-abstraction)
        RegistryNodeType.DATA_CONNECTOR,
        RegistryNodeType.DATA_FETCH_RECORD,
        # Risk scoring (CONCEPT:AU-KG.research.research-pipeline-runner)
        RegistryNodeType.RISK_ASSESSMENT,
        RegistryNodeType.RISK_FACTOR,
        RegistryNodeType.RISK_MITIGATION,
        # Backtest evaluation (CONCEPT:AU-AHE.evaluation.backtest-harness)
        RegistryNodeType.BACKTEST_RUN,
        RegistryNodeType.BACKTEST_METRIC,
        # Microstructure signals + trade journal (CONCEPT:AU-AHE.assimilation.microstructure-signal-fusion, CONCEPT:AU-AHE.assimilation.decision-distillation)
        RegistryNodeType.MICROSTRUCTURE_SIGNAL,
        RegistryNodeType.TRADE_JOURNAL,
        # Curated book/PDF trading knowledge taxonomy (CONCEPT:AU-AHE.assimilation.trading-curator)
        RegistryNodeType.TRADING_KNOWLEDGE,
        RegistryNodeType.STRATEGY_CONCEPT,
        RegistryNodeType.RISK_CONCEPT,
        RegistryNodeType.EXECUTION_CONCEPT,
    }
    edge_types: set[RegistryEdgeType] = {
        # Core edges
        RegistryEdgeType.HAS_FINANCIAL_INSTRUMENT,
        RegistryEdgeType.EXECUTED_TRANSACTION,
        RegistryEdgeType.BELONGS_TO_ORGANIZATION,
        RegistryEdgeType.DECIDED_BY,
        RegistryEdgeType.RESULTED_IN,
        RegistryEdgeType.OWNS_SYSTEM,
        RegistryEdgeType.DEPENDS_ON_SYSTEM,
        RegistryEdgeType.WAS_ATTRIBUTED_TO,
        RegistryEdgeType.CITES_SOURCE,
        # Trading pipeline edges (CONCEPT:AU-KG.research.research-pipeline-runner)
        RegistryEdgeType.GENERATED_SIGNAL,
        RegistryEdgeType.PLACED_ORDER,
        RegistryEdgeType.OPENED_POSITION,
        RegistryEdgeType.BELONGS_TO_PORTFOLIO,
        RegistryEdgeType.EXECUTES_STRATEGY,
        RegistryEdgeType.BACKTESTED_WITH,
        RegistryEdgeType.FORECASTED,
        RegistryEdgeType.VERSIONED_IN,
        RegistryEdgeType.GUARDED_BY,
        # Data connector edges (CONCEPT:AU-ECO.messaging.native-backend-abstraction)
        RegistryEdgeType.FETCHED_FROM,
        RegistryEdgeType.FALLS_BACK_TO,
        # Risk scoring edges (CONCEPT:AU-KG.research.research-pipeline-runner)
        RegistryEdgeType.ASSESSED_RISK,
        RegistryEdgeType.HAS_RISK_FACTOR,
        RegistryEdgeType.MITIGATED_BY,
        RegistryEdgeType.PROPAGATES_RISK_TO,
        # Backtest evaluation edges (CONCEPT:AU-AHE.evaluation.backtest-harness)
        RegistryEdgeType.EVALUATED_STRATEGY,
        RegistryEdgeType.HAS_METRIC,
        RegistryEdgeType.COMPARED_TO_BENCHMARK,
    }
    retrieval_boosts: dict[str, float] = {
        # Core boosts
        "has_financial_instrument": 1.4,
        "executed_transaction": 1.3,
        "belongs_to_organization": 1.2,
        "decided_by": 1.3,
        # Trading pipeline boosts — high value for provenance
        "generated_signal": 1.5,
        "executes_strategy": 1.5,
        "belongs_to_portfolio": 1.4,
        "opened_position": 1.3,
        "forecasted": 1.4,
        "versioned_in": 1.6,
        "guarded_by": 1.5,
        # Risk propagation — highest boost for transitive chains
        "propagates_risk_to": 1.6,
        "has_risk_factor": 1.4,
        # Backtest evaluation — important for strategy lineage
        "evaluated_strategy": 1.5,
        "backtested_with": 1.4,
        # Data connector fallback chains
        "falls_back_to": 1.2,
    }
    backlink_boost_strategy: BacklinkBoostStrategy = BacklinkBoostStrategy.GLOBAL
    backlink_boost_factor: float = 0.12
