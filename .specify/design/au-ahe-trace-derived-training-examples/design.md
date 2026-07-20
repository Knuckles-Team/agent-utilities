# Design: KG-Trace-Derived Native Program Training Evidence

## Intent

Close the loop between `RunTrace`/`ToolCall`/`OutcomeEvaluation` provenance and
the engine-owned `ProgramOptimize` job. A target's successful traces become governed
demonstration references; failures remain negative evidence and are never selected as
outputs to imitate.

## Composition

`trace_examples.py` reads the existing trace ontology with bounded queries and returns
provider-neutral rows. `program_optimization.run_program_optimization` blends those
rows with the caller's labeled corpus. `optimization_backend.OptimizationRequest`
converts every input, output, feedback, trace, evidence locus, and policy identity into
an opaque reference and attests that raw personal/local data is not persisted.

`GraphComputeEngine.optimize_program` submits `JobKind::ProgramOptimize`, boundedly
polls the durable job, rejects failed/cancelled/timed-out states, and validates the exact
typed result row schema. The engine authority-rebinds policy and persists the resulting
claim/evidence trajectory.

## Invariants

- No prompt, model output, identity, endpoint, credential, or local path enters the
  durable request or result.
- No Python optimizer or alternate provider path exists.
- Every query, corpus, job poll, candidate set, and result is bounded.
- A missing engine capability or invalid native result fails closed.
- Source mutation remains behind held-out evaluation, promotion, and approval gates.

## Interfaces

- Scheduled: `KG_OPTIMIZATION_ENABLED`, `KG_OPTIMIZATION_INTERVAL`.
- On demand: `graph_orchestrate action=optimize_component`.
- Core: `program_optimization.run_component_optimization`.
- Engine: `submit_program_optimization`, `program_optimization_status`,
  `program_optimization_result`, and `optimize_program`.
