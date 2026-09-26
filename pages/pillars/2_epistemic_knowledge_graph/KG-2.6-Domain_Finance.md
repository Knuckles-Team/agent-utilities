# Financial Trading Pipeline (CONCEPT:AU-KG.research.research-pipeline-runner)

## Overview
FIBO-aligned KG primitives for the full trading lifecycle: Signal → Order → Position → Portfolio → Strategy. OWL-promoted with transitive provenance chains.

## Implementation Details
- **Source Code**: ``agent_utilities/models/knowledge_graph.py``, ``agent_utilities/knowledge_graph/ontology_company_infra.ttl``
- **Pillar**: KG

### Core OWL Classes (Added in AU-AHE.assimilation.autonomous-trading-ecosystem updates)
- `:ExchangeBackend`: Abstracted financial exchange connections (`ccxt`, `alpaca`, `paper`).
- `:TradingStrategy`: Quantitative strategy lifecycle nodes.
- `:TradingSignal`: Alpha signals and factor predictions.
- `:PortfolioPosition`: Active instrument holdings.
- `:VersionedOrder`: Immutable order execution audit trail.
- `:RiskSnapshot`: Point-in-time risk measurements (Drawdown, P&L, Regime State).
- `:TradingDebate`: Multi-agent hypothesis consensus objects.
- `:BacktestResult`: Validation metrics for quantitative strategies.

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*

# Quantitative Frameworks (CONCEPT:AU-KG.research.research-pipeline-runner)

## Overview
Advanced quantitative logic for automated trading systems, now offloaded to the **Rust `epistemic-graph` compute engine** for high-performance, stateless execution.
- **AlphaCombinationEngine**: 11-step regression methodology for statistically independent signal weighting (Information Ratio optimization).
- **EmpiricalKellyOptimizer**: Uncertainty-adjusted position sizing using Monte Carlo simulations.
- **FractionalKellyOptimizer**: Position sizing scaling factor for high-variance environments.
- **CircuitBreaker**: Risk management hard stop drawdown limit.
- **Microstructure**: Level 1 Order Book Imbalance (OBI), volume-weighted Micro-Price, Convergence Filters, and Brier Score Validation.
- **StatisticalArbitrage**: Cointegration analysis and Ornstein-Uhlenbeck stochastic mean-reversion MLE parameter estimation.

## Implementation Details
- **Source Code**: the math is epistemic-graph's (`FinanceMarket`, `FinanceSignalModels` and the `Finance*` kernels such as `FinanceAlphaCombinationEngine`, `FinanceEmpiricalKelly`, `FinanceOrderBookImbalance`, `FinanceOuCalibrate` and `FinanceBrierScore`). agent-utilities keeps only the finance agent roles in ``agent_utilities/domains/finance/`` (EH-423 / AUD-30).
- **Pillar**: KG
- **Architecture Note**: agent-utilities computes no finance math. Agent roles call the epistemic-graph client; market data and order execution belong to the `emerald-exchange` and `market-data-mcp` connectors, and a live order exists only as a human-approved D18 change set.

# Risk Scoring Ontology (CONCEPT:AU-KG.research.research-pipeline-runner)

## Overview
Domain-agnostic risk assessment with `RiskAssessmentNode`, `RiskFactorNode`, `RiskMitigationNode`. OWL `propagatesRiskTo` enables transitive upstream risk chain inference.

## Implementation Details
- **Source Code**: ``agent_utilities/models/knowledge_graph.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Vectorized Context-Window Filtering (CONCEPT:AU-KG.research.research-pipeline-runner)

## Overview
Semantically prunes non-relevant subgraph context before swapping models on token overflow. Implemented as ``prune_context_by_semantic_distance()``.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/memory/agent_context.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*

See the [Epistemic Knowledge Graph pillar](../2_epistemic_knowledge_graph.md)
for the canonical capability inventory.
