# Backtest Evaluation Harness (CONCEPT:AHE-3.4)

## Overview
Strategy evaluation harness with SQLite storage, walk-forward validation windows, benchmark comparison, and KG integration via `BacktestRunNode`/`BacktestMetricNode`.

## Implementation Details
- **Source Code**: ``agent_utilities/harness/backtest_harness.py``
- **Pillar**: AHE

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Horizon-Aware Task Curriculum (CONCEPT:AHE-3.4)

## Overview
Progressive horizon scheduling with macro-action composition, subgoal checkpoints, and configurable promotion policies (threshold/plateau/adaptive). Based on Long-Horizon Training research (CONCEPT:AHE-3.4).

## Implementation Details
- **Source Code**: ``agent_utilities/graph/horizon_curriculum.py``
- **Pillar**: AHE

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Temporal Drift & EWC Consolidation (CONCEPT:AHE-3.4)

## Overview
Tracks knowledge drift via cosine distance/coefficient of variation and applies Elastic Weight Consolidation (EWC++) to prevent catastrophic forgetting (CONCEPT:AHE-3.4).

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/ewc.py``
- **Pillar**: AHE

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Structured Retry Manager (CONCEPT:ORCH-1.3)

## Overview
Shell-based success checks, on-failure hooks, and configurable timeouts for structured retry logic. Retry outcomes feed into TeamConfig reward signaling (AHE-3.3) for routing improvement. Adapted from Goose's retry.rs.

## Implementation Details
- **Source Code**: ``agent_utilities/graph/retry_manager.py``
- **Pillar**: AHE

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Agentic Engineering Patterns (CONCEPT:AHE-3.2)

## Overview
Out-of-the-box support for TDD Cycles (Red-Green-Refactor), First Run Tests, Agentic Manual Testing, Code Walkthroughs, and Interactive Explanations.

## Implementation Details
- **Source Code**: ``agent_utilities/harness/engineering.py``
- **Pillar**: AHE

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
