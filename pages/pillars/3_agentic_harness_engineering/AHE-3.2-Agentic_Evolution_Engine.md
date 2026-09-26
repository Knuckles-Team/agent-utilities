# Agent Config Versioning (CONCEPT:AU-AHE.harness.evolutionary-aggregation)

## Overview
Immutable config snapshots with forward-only rollback, structured diffs, and SUPERSEDES edge chains. Ported from MATE's AgentConfigVersion pattern. OWL-inferred `configDrift` and `stableConfig`.

## Implementation Details
- **Source Code**: ``agent_utilities/observability/config_versioning.py``
- **Pillar**: AHE

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Agent-Interpretable Model Evolver (CONCEPT:AU-AHE.evaluation.interpretability-tests)

## Overview
Autoresearch loop that evolves scikit-learn-compatible model classes optimized for dual objectives: predictive accuracy and LLM readability via `__str__()`. Pareto frontier tracking, reward decomposition (AHE-3.10), and KG-native evolutionary lineage. Based on arXiv:2605.03808. MCP-delegated model fitting via `data-science-mcp`.

## Implementation Details
- **Source Code**: ``agent_utilities/harness/imodel_evolver.py``
- **Pillar**: AHE

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# LLM-Graded Interpretability Tests (CONCEPT:AU-AHE.evaluation.interpretability-tests)

## Overview
6-category, 200-test protocol measuring whether an LLM can simulate model predictions, feature effects, and counterfactuals from `__str__()` alone. Reward hacking detection, numerical tolerance grading, and EvalRunner (AHE-3.12) integration. Based on arXiv:2605.03808.

## Implementation Details
- **Source Code**: ``agent_utilities/harness/continuous_evaluation_engine.py``
- **Pillar**: AHE

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
