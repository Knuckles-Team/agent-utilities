# Graph Orchestration

This directory contains the `pydantic-graph` implementation for high-fidelity agent orchestration.

## Overview

The `graph/` module implements a **Hierarchical State Machine (HSM)** that routes user queries through a multi-stage pipeline of specialized agents.

## Core Nodes

- **Router**: Topology selection and initial planning.
- **Planner**: Re-planning with feedback on failure.
- **Dispatcher**: Dynamic routing and parallel execution management.
- **Researcher**: Deep discovery across workspace and web.
- **Architect**: System design and approach definition.
- **Verifier**: Quality gate and scoring.
- **Synthesizer**: Final response composition.

## Key Files

- `steps.py`: Atomic functional definitions for each graph node.
- `executor.py`: Logic for spawning and executing dynamic MCP/A2A specialists.
- `runner.py`: The entry point for executing a graph session with full lifespan management.
- `builder.py`: Factory logic for assembling the graph from workspace metadata and KG registries.
- `hsm.py`: Hierarchical State Machine primitives and state invariant guards.

## Maintenance

- **State**: The `GraphState` (in `state.py`) is the shared memory of the graph. Ensure it remains serializable and compact.
- **Events**: Always emit lifecycle events using `emit_graph_event` for UI transparency.
- **Resilience**: Any new node should implement retry and fallback logic consistent with the `expert_executor_step`.
