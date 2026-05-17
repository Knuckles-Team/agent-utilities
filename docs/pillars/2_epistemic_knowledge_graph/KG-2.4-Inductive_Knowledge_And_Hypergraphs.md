# Cross-Pillar Synergy Engine (CONCEPT:KG-2.4)

## Overview
Discovers non-obvious functional synergies between the 5 Unified Pillars by analyzing concept bridges, computing pillar coupling metrics, and suggesting missing relationships. Leverages the Analogy Engine (KG-2.15), SKOS taxonomy, and transitive OWL properties. OWL property: `hasSynergyWith` (symmetric).

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/synergy_engine.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Formal Relations Engine (CONCEPT:KG-2.6)

## Overview
Mathematical relation properties (Reflexive, Symmetric, Transitive closures) and Equivalence Classes from MCS Ch 4. Provides zero-shot entity resolution by formally defining equivalence sets.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/core/formal_relations.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# State Machine Invariant Engine (CONCEPT:KG-2.6)

## Overview
Deterministic Finite Automata (DFA) abstractions and provable state invariants from MCS Ch 6. Formally validates transitions against structural invariants, preventing infinite loops.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/core/state_machines.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Markov Transition Forecasting (CONCEPT:KG-2.6)

## Overview
Markov Chain transition matrices over agent interaction traces (Vectorized Topologies) from MCS Ch 21. Calculates stationary distribution (Eigenvector) to predict statistical failure nodes.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/core/formal_reasoning_core.py``
- **Pillar**: KG

## Core Capabilities
The ``MarkovTransitionModel`` provides:
- **Trace ingestion** — Build transition matrices from sequential state observations
- **Stationary distribution** — Power iteration to find long-run state probabilities
- **Sink node prediction** — Identify absorbing/terminal states
- **Top-k next-state prediction** — Predict most likely successor states (used by ``PreemptiveCacheEngine``)
- **Chapman-Kolmogorov n-step forecasting** — Multi-step transition probabilities via matrix powers
- **State-specific forecasting** — Probability distribution over states after n steps from a given starting state

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*

# Markov Regime Detection (CONCEPT:KG-2.6)

## Overview
Market regime detection and forecasting using Markov Chains for financial time-series analysis. Classifies market states (Bull/Bear/Sideways) from returns data and builds probabilistic transition models for regime forecasting, trading signal generation, and walk-forward backtesting.

## Architecture

```
Raw Returns → MarketRegimeDetector → State Labels
                                          │
                                          ▼
              MarkovRegimeModel ← MarkovTransitionModel
                    │                     │
                    ▼                     ▼
             Regime Forecast        KG Persistence
                    │          (FinanceEngineMixin)
                    ▼
            Trading Signal / Walk-Forward Backtest
```

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/core/markov_regime.py``
- **Engine Integration**: ``agent_utilities/knowledge_graph/orchestration/engine_finance.py``
- **Domain Models**: ``agent_utilities/models/domains/finance.py``
- **Pillar**: KG + Finance Domain

## Key Classes

### `MarketRegimeDetector`
Detects market regimes from returns time-series. Supports two rolling return methods:
- **rolling_sum**: Simple sum of daily returns (fast, good approximation for small returns)
- **compounding**: Proper `∏(1+r_i) - 1` (more accurate for volatile assets like crypto)

### `MarkovRegimeModel`
End-to-end Markov Chain model:
- `fit(returns)` — Detect regimes → build transition matrix
- `forecast(state, n_steps)` — N-step regime probabilities
- `generate_signal(state)` — Bull_prob - Bear_prob trading signal
- `walk_forward_backtest(returns, lookback)` — Rolling re-estimation with no lookahead bias
- `to_kg_properties()` — Serialize model state for KG persistence

### `HiddenMarkovRegimeModel`
Gaussian Hidden Markov Model for latent regime detection (requires `hmmlearn`):
- `fit(returns)` — Baum-Welch EM with multiple random restarts
- `decode(returns)` — Viterbi decoding of most likely regime sequence
- `predict_proba(returns)` — Posterior regime probabilities per timestep

## Asset-Class-Specific Defaults

| Asset Class  | Bull Threshold | Bear Threshold | Window |
|-------------|---------------|----------------|--------|
| Equities    | +2.0%         | -2.0%          | 20     |
| Crypto      | +5.0%         | -5.0%          | 14     |
| Forex       | +0.5%         | -0.5%          | 30     |
| Commodities | +1.5%         | -1.5%          | 25     |
| Fixed Income| +0.3%         | -0.3%          | 40     |

All thresholds are configurable at instantiation.

## KG Node Types
- `MarkovRegimeStateNode` — A detected regime observation
- `MarkovTransitionMatrixNode` — A serialized transition matrix
- `RegimeSignalNode` — A generated trading signal

## KG Edge Types
- `TradingStrategy -[:MODELS_REGIME]-> MarkovTransitionMatrix`
- `MarkovTransitionMatrix -[:DETECTED_REGIME]-> MarkovRegimeState`
- `TradingStrategy -[:GENERATES_SIGNAL]-> RegimeSignal`

## Service Registry
Registered as discoverable services in the ``ServiceRegistry`` (CONCEPT:ORCH-1.4):
- `markov_regime_detection` (Layer: domain, Domain: finance)
- `hmm_regime_detection` (Layer: domain, Domain: finance)
