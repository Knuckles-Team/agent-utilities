# Spec-Driven Development (SDD) Orchestrator

> CONCEPT:AU-009 — Spec-Driven Development

## Overview

The SDD orchestrator (`sdd/orchestrator.py`) implements a **specification-first development pipeline** where formal specs drive plan generation, task decomposition, and parallel execution. This aligns with the `.specify` standard (spec-kit).

## Pipeline

## SDD Lifecycle Diagram

```mermaid
graph TD
    subgraph Phase1 [1. Specification Ingestion]
        direction TB
        Req[requirements.md] --> Spec[Spec]
        Cons[constraints.md] --> Spec
        Acc[acceptance.md] --> Spec
    end

    subgraph Phase2 [2. Plan Generation]
        Spec --> PlanAgent[Planning Agent]
        PlanAgent --> ImplPlan[Implementation Plan]
    end

    subgraph Phase3 [3. Task Decomposition]
        ImplPlan --> TaskAgent[Task Decomposer]
        TaskAgent --> TaskA[Task A]
        TaskAgent --> TaskB[Task B]
        TaskAgent --> TaskC[Task C]
    end

    subgraph Phase4 [4. Parallel Execution]
        TaskA --> PySpecialist[Python Specialist]
        TaskB --> TSSpecialist[TypeScript Specialist]
        TaskC --> DevOpsSpecialist[DevOps Specialist]

        PySpecialist --> Joiner[Execution Joiner]
        TSSpecialist --> Joiner
        DevOpsSpecialist --> Joiner
    end

    Joiner --> Verifier[Spec Verifier]

    style Phase1 fill:#f5f5f5,stroke:#666
    style Phase2 fill:#dae8fe,stroke:#6c8ebf
    style Phase3 fill:#e1d5e7,stroke:#9673a6
    style Phase4 fill:#d5e8d4,stroke:#82b366
    style Verifier fill:#fff2cc,stroke:#d6b656
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

## Integration with HSM and Knowledge Graph

The SDD pipeline is deeply integrated with the core architecture:

- **HSM Dispatcher**: Task execution is routed through the main Hierarchical State Machine (AU-002). Each task is mapped to a Specialist Superstate (e.g., Python Coder) which enters its own execution loop.
- **Knowledge Graph (AU-003)**: The generated Spec, Implementation Plan, and individual Tasks are persisted into the Knowledge Graph as nodes. This provides long-term context, allowing the system to reference past design decisions during future tasks.

## Real-World Usage Example

```python
import asyncio
from agent_utilities.sdd.orchestrator import SDDOrchestrator
from agent_utilities.core.workspace import get_agent_workspace

async def main():
    workspace = get_agent_workspace()

    # Initialize the SDD Pipeline
    orchestrator = SDDOrchestrator(
        spec_dir=workspace / ".specify",
        workspace=workspace,
    )

    # Run the full pipeline: Ingest -> Plan -> Task -> Execute -> Verify
    results = await orchestrator.run()

    if results.verified:
        print(f"Implementation complete and verified. Score: {results.verification_score}")
    else:
        print("Verification failed. Check the tasks for feedback gradients.")

if __name__ == "__main__":
    asyncio.run(main())
```

## Integration with CONCEPT Markers

SDD tasks are tagged with `CONCEPT:` markers that link:
- Specification → Implementation plan → Code → Tests → Documentation

This creates a full traceability chain from requirement to verification.
