# Learned Agent Routing (CONCEPT:ORCH-1.8)

## Overview
Jointly optimizes decomposition depth, worker choice, and inference budget from execution traces. Three policies: RuleBasedPolicy (keyword pattern matching), TraceLearnedPolicy (softmax scoring from historical traces with EMA quality tracking), CostAwareRouter (Pareto-optimal cost/accuracy filtering). Derived from Uno-Orchestra (arXiv:2605.05007v1).

## Implementation Details
- **Source Code**: ``agent_utilities/graph/routing_policy.py``
- **Pillar**: ORCH

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
