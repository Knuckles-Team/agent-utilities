# Spec-Driven Development (SDD) Orchestrator

> CONCEPT:AU-AHE.harness.harness-evolution — Spec-Driven Development

## Overview

The SDD orchestrator (`sdd/orchestrator.py`) implements a **specification-first development pipeline** where formal specs drive plan generation, task decomposition, and parallel execution. This aligns with the `.specify` standard (spec-kit).

## Pipeline

## SDD Lifecycle Diagram

<div class="admonition architecture" markdown>
<p class="admonition-title">Four phases: ingest, plan, decompose, execute in parallel</p>

**1. Specification ingestion.** `requirements.md`, `constraints.md`, and
`acceptance.md` all feed one `Spec`. **2. Plan generation.** The Planning
Agent turns that spec into an Implementation Plan. **3. Task
decomposition.** A Task Decomposer splits the plan into Task A, B, and C.
**4. Parallel execution.** Each task goes to its own specialist (Python,
TypeScript, DevOps); all three join at an Execution Joiner, which feeds
TDD verification (tests pass).
</div>

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

- **HSM Dispatcher**: Task execution is routed through the main Hierarchical State Machine (CONCEPT:AU-ORCH.execution.inject-signal-board-observations). Each task is mapped to a Specialist Superstate (e.g., Python Coder) which enters its own execution loop.
- **Knowledge Graph (CONCEPT:AU-ORCH.execution.inject-signal-board-observations)**: The generated Spec, Implementation Plan, and individual Tasks are persisted into the Knowledge Graph as nodes. This provides long-term context, allowing the system to reference past design decisions during future tasks.

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
