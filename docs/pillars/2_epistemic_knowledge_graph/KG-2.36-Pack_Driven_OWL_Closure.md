# KG-2.36 — Pack-Driven OWL Closure

**Pillar:** 2 — Epistemic Knowledge Graph · **Status:** live

## What

A pack can declare its edge types as OWL object-properties — **transitive**,
**symmetric**, or **inverse-of** another edge. When the pack activates, these
declarations are unioned into the OWL reasoning sets so the existing
promote→reason→downfeed cycle materialises multi-hop and inverse edges **for free**:
e.g. the `research-state` pack makes `supports_belief` transitive (so `A→B→C` yields
`A→C`) and `cites_source` inverse-of `cited_by_paper` (so a citation gets its
back-edge automatically).

## Why — the "free value-add"

This is a capability a flat regex brain layer (gbrain) structurally cannot provide.
Because we already run an OWL reasoner with a closure cycle, exposing pack edges to
it is nearly free and turns extracted edges (KG-2.33) into a reasoned graph: support
chains, citation networks, and symmetric relations close automatically.

## How / Wiring

- `models/schema_pack.py`: `OwlObjectProperty` + `owl_object_properties` field +
  `get_owl_closure_sets()`.
- `knowledge_graph/core/owl_bridge.py`: `__init__` stores the pack closure sets;
  `_python_reasoning` and `_rust_reasoning` **union** them into the transitive/
  symmetric sets, and `_inverse_inferences()` emits inverse edges. Idempotent — the
  `_downfeed_inferences` dedup guard makes re-running the cycle a fixpoint.

> **Wire-First note:** the default lightweight reasoning path does **not** read
> `ontology.ttl`; it uses in-memory Python/Rust sets. The load-bearing change is the
> union into those sets, not a TTL edit.

## Tests

`tests/knowledge_graph/test_pack_owl_closure.py` (transitive, inverse, symmetric, fixpoint, core no-op).
