# AU-KG.ontology.schema-pack-lifecycle-audit — Schema-Pack Lifecycle and Audit

**Pillar:** 2 — Epistemic Knowledge Graph · **Status:** live

## What

Active-pack resolution and lifecycle that makes the domain Schema Pack reachable
end-to-end, plus an observe-only candidate-type audit:

- **Loader** — resolves the active pack by precedence `explicit > GRAPH_SCHEMA_PACK
  env > config.json (graph.schema_pack) > core`; unknown names warn and fall back to
  `core` (never raise). `set_active_pack()` re-wires live consumers and the engine
  rebuilds its retriever (carrying the new `pack.signature()` so cached results from
  a prior pack can't leak — the gbrain `knobs_hash` analogue).
- **Candidate audit** — when an EXCLUSIVE pack is active and a write introduces a
  type outside the active set, it is recorded (privacy-hashed by default) for review.
  Mirrors gbrain's `candidate-audit.ts`. Observe-only: it never blocks a write.

## Why

Packs previously existed as models but were constructed *pack-blind* in the engine,
so none of their signals were reachable. This is the Wire-First prerequisite that
turns every other Schema-Pack 2.0 capability on. The audit surfaces vocabulary drift
without forcing a rigid schema.

## How / Wiring

- `models/schema_pack_loader.py`, `models/schema_pack_audit.py`.
- `knowledge_graph/core/engine.py`: `__init__` resolves the active pack, builds
  `HybridRetriever(self, schema_pack=...)`, registers `_on_schema_pack_change`, and
  calls `_audit_candidate_type` from `add_node`/`link_nodes`.
- Entry points: `graph_configure(action="schema_pack", config_key=<name>)` get/set;
  `graph_configure(action="schema_candidates")` review.

## Tests

`tests/knowledge_graph/test_schema_pack_loader.py`, `tests/knowledge_graph/test_schema_pack_audit.py`.
