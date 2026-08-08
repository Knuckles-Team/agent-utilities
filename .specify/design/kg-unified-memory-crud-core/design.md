# Design Document: `graph_memory`'s recall/store/link route into the SAME `graph_write` core the REST twins and harness tools use — no fourth memory surface

CONCEPT:AU-KG.memory.unified-memory-crud-core

> Realised by `agent_utilities/mcp/tools/engine_surface_tools.py:883-888`
> (`_MEMORY_CRUD_ACTIONS` and the decision comment), `:891-931`
> (`_memory_crud` dispatcher) and `:2050` (the short-circuit call site).
> Introduced by commit `89076844` ("feat(ontology-federation)").

## Decision — a new caller-facing surface must route into an existing core, not implement the mutation itself

Three ways to mutate agent memory already existed: the REST
`/graph/write/memory[/recall]` endpoints, the harness `kg_memory_recall` /
`kg_memory_store` tools, and the underlying `graph_write` core they both use.
Adding the `graph_memory` MCP tool's `recall`/`store`/`link` actions raised the
question of whether it should talk to storage directly.

It does not. The comment at the dispatcher states the rule and the count:

> *"route into the SAME `graph_write` tool the REST ... twins and the harness
> ... tools already use — one core, no fourth memory surface."*

**The rejected alternative is not hypothetical: a separate engine-side memory
surface (EG-318) was available and explicitly not used.** The commit diff
carries a comment distinguishing this routing from it. That is the sharper
version of the decision — the choice was not "reimplement or reuse", it was
"reuse the core we already converge on, or adopt a second legitimate-looking
one that happens to be closer to the engine".

The reason a fourth surface is worth actively refusing is that memory mutation
carries policy, not just storage. Bi-temporal stamping, supersede-instead-of-
overwrite, soft delete, provenance, dedup — each surface that writes memory
directly must re-implement all of it, and any surface that implements it
slightly differently produces memory records that are subtly inconsistent with
the others. Those divergences are invisible at write time and surface much
later as contradictory recall. Routing every caller through one core means the
policy is written once and cannot drift between entrypoints.

This is the chokepoint form of the pattern: the guarantee comes from there
being exactly one place the mutation can happen, not from each entrypoint
remembering to do the right thing.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/tools/engine_surface_tools.py`. The
  `graph_write` core and the REST/harness twins are unchanged — that is the
  point.
- **Backward Compatible**: Yes — additive MCP actions over an existing core.
- **Known weak point**: the invariant is maintained by convention. Nothing
  structurally prevents a fifth surface from calling the engine's memory API
  directly, and the EG-318 surface that was declined here remains available to
  anyone who finds it. The guarantee would be enforceable — by making the
  direct engine memory API private to the core — but currently is not.
