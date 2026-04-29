# Spec-Driven Development (SDD) Orchestrator

> CONCEPT:AU-009 — Spec-Driven Development

## Overview

The SDD orchestrator (`sdd/orchestrator.py`) implements a **specification-first development pipeline** where formal specs drive plan generation, task decomposition, and parallel execution. This aligns with the `.specify` standard (spec-kit).

## Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐
│  Specs   │ →  │  Plans   │ →  │  Tasks   │ →  │  Parallel │
│ (.specify)│    │ (impl)   │    │ (atomic) │    │  Execution│
└──────────┘    └──────────┘    └──────────┘    └───────────┘
```

### Phase 1: Specification Ingestion

Reads `.specify/` directory structures containing:
- `requirements.md`: Functional requirements
- `constraints.md`: Non-functional requirements
- `acceptance.md`: Acceptance criteria

### Phase 2: Plan Generation

Uses the planning agent to generate implementation plans from specs:
- Dependency analysis
- Topological sorting of tasks
- Risk assessment

### Phase 3: Task Decomposition

Breaks plans into atomic, executable tasks:
- Each task maps to a single file or function
- Tasks are tagged with CONCEPT markers for traceability
- Dependencies between tasks are explicit

### Phase 4: Parallel Execution

Dispatches independent tasks to specialist agents:
- DAG-based scheduling
- Parallel execution of independent tasks
- Synchronization barriers between dependent layers

## Usage

```python
from agent_utilities.sdd.orchestrator import SDDOrchestrator

orchestrator = SDDOrchestrator(
    spec_dir=".specify/",
    workspace="/path/to/project",
)

# Run the full pipeline
results = await orchestrator.run()
```

## Integration with CONCEPT Markers

SDD tasks are tagged with `CONCEPT:` markers that link:
- Specification → Implementation plan → Code → Tests → Documentation

This creates a full traceability chain from requirement to verification.
