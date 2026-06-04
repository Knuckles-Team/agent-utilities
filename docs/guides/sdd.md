# Spec-Driven Development (SDD) Orchestrator

> CONCEPT:AHE-3.0 — Spec-Driven Development

## Overview

The SDD orchestrator (`sdd/orchestrator.py`) implements a **specification-first development pipeline** where formal specs drive plan generation, task decomposition, and parallel execution. This aligns with the `.specify` standard (spec-kit).

## Pipeline

## SDD Lifecycle Diagram

```mermaid
graph TD
    subgraph Phase1 [1. Specification Ingestion]
        direction TB
        Req[ORCH-1.5: requirements.md] --> ORCH-1.5: Spec[Spec]
        Cons[ORCH-1.5: constraints.md] --> Spec
        Acc[ORCH-1.5: acceptance.md] --> Spec
    end

    subgraph Phase2 [2. Plan Generation]
        Spec --> PlanAgent[ORCH-1.2: Planning Agent]
        PlanAgent --> ImplPlan[ORCH-1.5: Implementation Plan]
    end

    subgraph Phase3 [3. Task Decomposition]
        ImplPlan --> TaskAgent[ORCH-1.1: Task Decomposer]
        TaskAgent --> TaskA[ORCH-1.1: Task A]
        TaskAgent --> TaskB[ORCH-1.1: Task B]
        TaskAgent --> TaskC[ORCH-1.1: Task C]
    end

    subgraph Phase4 [4. Parallel Execution]
        TaskA --> PySpecialist[ORCH-1.2: Python Specialist]
        TaskB --> TSSpecialist[ORCH-1.2: TypeScript Specialist]
        TaskC --> DevOpsSpecialist[ORCH-1.2: DevOps Specialist]

        PySpecialist --> Joiner[ORCH-1.0: Execution Joiner]
        TSSpecialist --> Joiner
        DevOpsSpecialist --> Joiner
    end

    Joiner --> Verifier[AHE-3.1: TDD Verification - tests pass]

    style Phase1 fill:#f5f5f5,stroke:#666
    style Phase2 fill:#dae8fe,stroke:#6c8ebf
    style Phase3 fill:#e1d5e7,stroke:#9673a6
    style Phase4 fill:#d5e8d4,stroke:#82b366
    style Verifier fill:#fff2cc,stroke:#d6b656
```

### Phase 1: Specification Ingestion

Reads/writes the `.specify/` directory structure managed by `SDDManager`
(`sdd/__init__.py`). Artifacts are persisted as Markdown (with JSON sidecars
for design docs):
- `constitution.md`: Project constitution (vision, principles, tech stack)
- `design/<feature_id>/design.md`: KG-gated design (Extend-Before-Invent)
- `specs/<feature_id>/spec.md`: User stories + acceptance criteria + NFRs
- `specs/<feature_id>/plan.md`: Implementation plan (approach, risks)
- `specs/<feature_id>/tasks.md`: Decomposed tasks (spec-kit `[P]` parallel markers)

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

## Integration with HSM and Knowledge Graph

The SDD pipeline is deeply integrated with the core architecture:

- **HSM Dispatcher**: Task execution is routed through the main Hierarchical State Machine (CONCEPT:ORCH-1.0). Each task is mapped to a Specialist Superstate (e.g., Python Coder) which enters its own execution loop.
- **Knowledge Graph (CONCEPT:ORCH-1.0)**: The generated Spec, Implementation Plan, and individual Tasks are persisted into the Knowledge Graph as nodes. This provides long-term context, allowing the system to reference past design decisions during future tasks.

## Real-World Usage Example

```python
import asyncio
from agent_utilities.sdd.orchestrator import SDDOrchestrator
from agent_utilities.models import AgentDeps

async def main():
    # AgentDeps carries the workspace path and the agentic `patterns`
    # helpers (first_run_tests, tdd_red/green/refactor_phase, etc.).
    deps = AgentDeps(...)

    # Initialize the SDD orchestrator. `spec_generator` is an optional
    # async callable that turns a goal string into a Spec; when omitted,
    # a minimal default Spec is used.
    orchestrator = SDDOrchestrator(deps, spec_generator=None)

    # Run the full workflow for a goal:
    #   baseline tests -> spec -> TDD RED -> implement -> TDD GREEN -> REFACTOR
    refactored_code = await orchestrator.run_sdd("Add a rate limiter to the API")

    print("SDD workflow complete. Final implementation:\n", refactored_code)

if __name__ == "__main__":
    asyncio.run(main())
```

## Integration with CONCEPT Markers

SDD tasks are tagged with `CONCEPT:` markers that link:
- Specification → Implementation plan → Code → Tests → Documentation

This creates a full traceability chain from requirement to verification.
