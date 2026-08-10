# Design Document: Wire the quantum surface into the existing ToolCall/RunTrace provenance model (Q9)

CONCEPT:AU-KG.temporal.quantum-run-provenance

> `agent_utilities/observability/quantum_trace.py`

## Decision — one new typed node (`:QuantumJob`), grounded the same way existing provenance is

`plans/au-eg-program/program/quantum-external-providers.md` §1.5 states Q9
"wires quantum into the EXISTING `:ToolCall`/`RunTrace` provenance model — it
does not build a new one." This module adds exactly one new typed node,
`:QuantumJob`, grounded to the calling run the SAME way
`agent_utilities/runtime/provenance.py`'s `:WorkspaceObservation` is grounded
to its `:WorkspaceAction` (a `HAS_*`-shaped edge off the run/action, not a
parallel provenance system):

```text
(:RunTrace {id: trace:<opaque-ref>}) -[:HAS_QUANTUM_RUN]-> (:QuantumJob)
```

A `:QuantumJob` carries the full Q0/Q9 metadata the engine's `Method::Quantum`
response already returns verbatim (backend id, formalism, seed, shots, circuit
hash, exact, noise model id, fidelity hint, wall time, peak memory) plus the
planner's R0-R5 audit trail (`eg_quantum_core::planner::PlannerDecision.audit`)
so an explicit R5 `backend_id` override is durably auditable from the KG side,
per the program's "always honoured but MUST be audited" requirement. Reuses
the shared trace schema (`trace_ontology.py`'s `TRACE_SCHEMA_VERSION`,
`content_digest`, `trace_id`) rather than inventing a parallel identifier
scheme. Best-effort, like every other provenance write in this codebase: a
cold or absent KG must never break a `graph_quantum` call.

**The rejected alternative** was a standalone quantum-specific trace/logging
system. That would duplicate `:ToolCall`/`RunTrace`'s existing schema-version,
digest, and sequencing machinery, and — critically — would not compose with
the KG's existing "what is durably in flight, waiting on what" query surface,
which is the entire reason a KG-native provenance model exists in this repo.

## Risk Assessment

- **Blast Radius**: `agent_utilities/observability/quantum_trace.py` only;
  read by any future KG query joining `:RunTrace` to `:QuantumJob`.
- **Backward Compatible**: Yes — additive node/edge type, no existing
  provenance shape changed.
- **Known weak point**: like all best-effort provenance writes in this
  codebase, a `:QuantumJob` write failure is silent (logged, not raised) — a
  quantum call can succeed with no provenance record if the KG is briefly
  unavailable.
