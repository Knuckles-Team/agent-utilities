# Backtest Evaluation Harness (CONCEPT:AU-AHE.evaluation.backtest-harness)

## Overview
Strategy evaluation harness with SQLite storage, walk-forward validation windows, benchmark comparison, and KG integration via `BacktestRunNode`/`BacktestMetricNode`.

## Implementation Details
- **Source Code**: ``agent_utilities/harness/continuous_evaluation_engine.py``
- **Pillar**: AHE

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Horizon-Aware Task Curriculum (CONCEPT:AU-AHE.evaluation.backtest-harness)

## Overview
Progressive horizon scheduling with macro-action composition, subgoal checkpoints, and configurable promotion policies (threshold/plateau/adaptive). Based on Long-Horizon Training research (CONCEPT:AU-AHE.evaluation.backtest-harness).

## Implementation Details
- **Source Code**: ``agent_utilities/graph/horizon_curriculum.py``
- **Pillar**: AHE

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Temporal Drift & EWC Consolidation (CONCEPT:AU-AHE.evaluation.backtest-harness)

## Overview
Tracks knowledge drift via cosine distance/coefficient of variation and applies Elastic Weight Consolidation (EWC++) to prevent catastrophic forgetting (CONCEPT:AU-AHE.evaluation.backtest-harness).

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/memory/optimization_engine.py``
- **Pillar**: AHE

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Structured Retry Manager (CONCEPT:AU-ORCH.execution.execution-budget-caps)

## Overview
Shell-based success checks, on-failure hooks, and configurable timeouts for structured retry logic. Retry outcomes feed into TeamConfig reward signaling (AHE-3.3) for routing improvement. Adapted from Goose's retry.rs.

## Implementation Details
- **Source Code**: ``agent_utilities/security/execution_stability_engine.py``
- **Pillar**: AHE

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Agentic Engineering Patterns (CONCEPT:AU-AHE.harness.evolutionary-aggregation)

## Overview
Out-of-the-box support for TDD Cycles (Red-Green-Refactor), First Run Tests, Agentic Manual Testing, Code Walkthroughs, and Interactive Explanations.

## Implementation Details
- **Source Code**: ``agent_utilities/harness/engineering.py``
- **Pillar**: AHE

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
