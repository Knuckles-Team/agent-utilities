# Financial Trading Pipeline (CONCEPT:KG-2.6)

## Overview
FIBO-aligned KG primitives for the full trading lifecycle: Signal → Order → Position → Portfolio → Strategy. OWL-promoted with transitive provenance chains.

## Implementation Details
- **Source Code**: ``agent_utilities/models/knowledge_graph.py``, ``agent_utilities/knowledge_graph/ontology_company_infra.ttl``
- **Pillar**: KG

### Core OWL Classes (Added in EE-001 updates)
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

# Quantitative Frameworks (CONCEPT:KG-2.6)

## Overview
Advanced quantitative logic for automated trading systems, now offloaded to the **Rust `epistemic-graph` compute engine** for high-performance, stateless execution.
- **AlphaCombinationEngine**: 11-step regression methodology for statistically independent signal weighting (Information Ratio optimization).
- **EmpiricalKellyOptimizer**: Uncertainty-adjusted position sizing using Monte Carlo simulations.
- **FractionalKellyOptimizer**: Position sizing scaling factor for high-variance environments.
- **CircuitBreaker**: Risk management hard stop drawdown limit.
- **Microstructure**: Level 1 Order Book Imbalance (OBI), volume-weighted Micro-Price, Convergence Filters, and Brier Score Validation.
- **StatisticalArbitrage**: Cointegration analysis and Ornstein-Uhlenbeck stochastic mean-reversion MLE parameter estimation.

## Implementation Details
- **Source Code**: ``agent_utilities/domains/finance/signal_fusion.py``, ``agent_utilities/domains/finance/portfolio_optimizer.py``, ``agent_utilities/domains/finance/microstructure.py``, ``agent_utilities/domains/finance/cross_market_arb.py``
- **Pillar**: KG
- **Architecture Note**: The Python layer acts as a lightweight orchestrator and thin proxy. All heavy numerical lifting (MVO, Risk Parity, Black-Litterman, HMM regime detection) is delegated to the `epistemic-graph-server` via Unix Domain Socket (UDS) RPC. Python dependencies like `numpy` and `scipy` have been strictly purged from the `agent-utilities` ecosystem to maintain a minimal surface area.

# Risk Scoring Ontology (CONCEPT:KG-2.6)

## Overview
Domain-agnostic risk assessment with `RiskAssessmentNode`, `RiskFactorNode`, `RiskMitigationNode`. OWL `propagatesRiskTo` enables transitive upstream risk chain inference.

## Implementation Details
- **Source Code**: ``agent_utilities/models/knowledge_graph.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# [Vectorized Context-Window Filtering](pillars/2_epistemic_knowledge_graph/KG-2.50-Vectorized_Context-Window_Filtering.md) (CONCEPT:KG-2.6)

## Overview
Semantically prunes non-relevant subgraph context before swapping models on token overflow.

## Implementation Details
- **Source Code**: ``agent_utilities/graph/context_filter.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
