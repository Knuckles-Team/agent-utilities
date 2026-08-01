# Design Document: The lightweight OWL closure tick reasons engine-native, never requiring an owlready2 backend on the hot path

CONCEPT:AU-KG.ontology.owl-closure-native

> `agent_utilities/knowledge_graph/maintenance/owl_closure.py`.

## Decision — the background closure tick runs `OWLBridge.run_cycle(lightweight=True)` with `owl_backend=None`, so the engine reasons over the live graph directly and owlready2 is never touched on this path

`owl_closure.py:103-107` states the decision plainly: "the lightweight closure
reasons engine-native (`client.rdf.owl_reason` over the live graph), so it
needs NO owlready2 backend: the bridge runs `run_cycle(lightweight=True)` with
`owl_backend=None` and the engine materializes the OWL/RDFS+ closure.
owlready2 is a true last-resort fallback (used only for the full-DL cycle),
kept out of this hot path." The surrounding code (`owl_closure.py:99-117`)
never raises — every failure path (no graph, no `OWLBridge` importable)
returns a structured `_empty_summary(...)` sentinel with a `status` of
`"skipped"`/`"error"`, and the SHACL `conforms` check defaults to `True` when
validation is unavailable "so the closure never blocks on a missing
validator."

**The rejected alternative is requiring the owlready2 backend for every
closure tick** — the pluggable `OWLBackend` abstraction
(`.specify/design/kgo-har-pluggable-owl-backend/design.md`) makes this
possible, but doing it unconditionally would mean the periodic background
closure — running on every tick, on potentially every deployment profile —
pays the cost (and the dependency requirement) of an in-process reasoner even
when the engine can answer the SAME entailment question natively. Reserving
owlready2's full Description Logic reasoning for the EXPLICIT full-DL cycle
(a caller-requested, heavier operation) keeps the routine background tick
cheap and dependency-light, while still allowing the stronger reasoning mode
when a caller actually needs it.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/maintenance/owl_closure.py`, the
  background closure tick's promoted-node/inferred-edge counts.
- **Backward Compatible**: Yes — `lightweight=True` is the existing tick
  behavior; `owl_backend=None` here is a resource-usage optimization, not a
  behavior change to the entailments produced (same inference-dict shape as
  the owlready2 path, per `.specify/design/kgo-e1-schema-packs/design.md`'s
  pack-owl-closure pointer).
- **Known weak point**: the engine-native reasoner covers OWL 2 EL+/RL, a
  strict subset of full Description Logic — an entailment that requires full
  DL reasoning is invisible to the lightweight tick and only surfaces if a
  caller explicitly runs the heavier `lightweight=False` cycle.
