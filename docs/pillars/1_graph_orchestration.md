# Pillar 1: Graph Orchestration Engine

## Overview

The **Graph Orchestration Engine** represents the foundational execution layer of the `agent-utilities` ecosystem. Moving away from rigid LLM chains and monolithic prompt contexts, this pillar implements a Hierarchical Task Network (HTN) backed by Pydantic Graph, transitioning linear execution into dynamic, topological routing.

## Why We Built This (Rationale)

As our agent ecosystem scaled to include dozens of domain specialists (Python, TS, CI/CD, DB) and hundreds of MCP tools, we encountered three critical failure modes:
1. **Prompt Bloat & Context Pollution**: Injecting all available tools into a single prompt exceeded context limits and degraded LLM reasoning accuracy.
2. **Sequential Bottlenecks**: Large features were executed linearly, squandering the opportunity for parallel discovery and implementation.
3. **Catastrophic Forgetting & Loop Cycles**: Agents would forget successful tool combinations or fall into infinite retry loops without an enforced architectural guardrail.

## How It Works (Implementation)

The architecture solves these bottlenecks through several interdependent primitives:

### Registry Hot Cache & Unified Specialists (ORCH-1.2)
We collapsed the artificial boundary between `prompt` and `mcp` agents into a singular `specialist` type. The **Registry Hot Cache** maintains an O(1) session-scoped index of these specialists. Instead of passing 50+ specialists to the orchestrator, it filters down to the Top-7 relevant specialists per query, reducing prompt token bloat by ~7x.

### Spec-Driven Development Pipeline (ORCH-1.7)
The orchestrator implements a multi-stage SDD pipeline:
- **Discovery & Requirements**: Generates structured `Spec` models with measurable success criteria.
- **Task Decomposition**: Emits a `Tasks` dependency graph, identifying which subtasks can be executed in parallel (e.g., frontend and backend).
- **Parallel Dispatch**: Fuses tasks out to specific `specialist` workers, leveraging the `Execution Visibility Graph` to constrain context so a backend specialist only sees backend-related prior steps.

### Learned Agent Routing & Execution Budgets (ORCH-1.8 & ORCH-1.3)
Routing isn't static. `TraceLearnedPolicy` uses softmax scoring over historical `ExecutionTrace` records with an exponential moving average (EMA) to actively down-weight specialists with low success rates. `ExecutionBudget` acts as an absolute cost governor, preempting infinite loops by enforcing USD/token constraints at the dispatcher step.

## Benefits Introduced

- **Cost Efficiency**: By utilizing `Confidence-Gated Model Routing`, trivial queries fallback to smaller models (`gpt-4o-mini`), saving reasoning tokens for complex HTN planning.
- **Architectural Safety**: `Subagent Lifecycle Patterns` and recursive execution constraints ensure the system fails gracefully and retries contextually rather than spinning in infinite loops.
- **Test-Time Scaling**: The system achieves zero-shot generalization by spawning parallel agent rollouts and selecting the optimal path via dynamic subgraph convergence and evolutionary aggregation.

## Key Concepts Leveraged
- **ORCH-1.0**: Dynamic Subgraph Orchestrator
- **ORCH-1.1**: Agentic Planning Engine (Planning)
- **ORCH-1.2**: Agentic Planning Engine (Routing)
- **ORCH-1.3**: Execution Budgets & State Safety
- **ORCH-1.7**: Spec-Driven Development
- **ORCH-1.8**: Learned Agent Routing
