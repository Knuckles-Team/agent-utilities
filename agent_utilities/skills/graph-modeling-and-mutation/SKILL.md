---
name: graph-modeling-and-mutation
skill_type: skill
description: >-
  Design canonical ontology-backed graph models and perform governed mutations
  through Graph-OS object and memory surfaces. Use for nodes, edges, memories,
  concepts, property or value types, ontology functions, derivations, object
  sets, permissions, indexes, linked materialization, reversible changes, or
  transactional write planning. For low-level native engine compute, storage,
  transaction, or cluster primitives, use graph-engine-and-modalities.
---

# Graph modeling and mutation

Translate a domain change into canonical types, validate it, and apply the
smallest governed mutation with a verification and rollback plan.

## Action reference

| Tool | Actions | Notes |
|---|---|---|
| `graph_write` | `add_node`, `add_edge`, `delete_node`, `delete_edge`, `register_external_graph`, `bulk_ingest`, `compare_and_set` (atomic conditional update — applies `updates` only if every field in `conditions` still matches), `store_memory`, `recall_memory`, `recall_media`, `log_chat`, `submit_sdd`, `register_execution`, `check_loop` | the primary KG mutation interface |
| `graph_memory` | engine methods (1:1, dashes→underscores): `create_summary`, `consolidate`, `maintain`, `add_scene_object`, `world_transform`, `start_trajectory`, `append_step`, `discounted_return`, `get_*`; unified memory-CRUD `store` (`agent_id`+`content`[+`memory_type`,`tags`]), `recall` (`query`), `link` | episodic→semantic consolidation, the spatial scene graph, RL trajectories; the unified CRUD actions route into the SAME `graph_write` memory core as the REST `/graph/write/memory` twins |
| `graph_ontology` + object layer | `graph_ontology`: `load`/`list`/`get`/`update`/`delete`/`validate`/`activate`/`deactivate`/`sync_packages` (SHACL-validated, versioned, native-reasoner-loaded); `ontology_property_types`/`ontology_value_types` (type registry: list/describe/validate/coerce); `ontology_interface` (`implementers`/`conforms`/`owl`, `registry='enterprise'` for standard contracts); `ontology_sampling_profile` (task-aware LLM sampling profiles — list/describe/resolve/set/evolve/owl); `ontology_function` (typed versioned functions, `list`/`invoke`); `ontology_derive` (compute derived properties live at read time); `ontology_link_materialize` (reify a many-to-many link as a junction triple); `ontology_leanix_sync` (mirror the live LeanIX metamodel as OWL/RDF, `dry_run` first); `object_edits` (durable object-edit ledger — record/revert/history/as_of, optimistic `expect`); `object_index` (search-index lifecycle — sync/reindex/status); `object_permissioning` (`redact`/`restricted_view`/`mark`, ambient actor); `object_set` (Foundry-style object sets — `of_type`/`search`/`filter`/`pivot`/`aggregate`/`union`/`intersect`/`subtract`) | the whole ontology + Foundry-style object layer, one skill wraps all thirteen tools |
| `concept_registry` (text → structured concepts) | parses unstructured text (transcripts/papers/specs) into `Concept`/`Reference` nodes + `SUB_CLASS_OF`/`DEFINED_IN` edges, then compiles idempotent parameterized `MERGE` Cypher for `graph_write` | 3-step pipeline: parse extracted terms → structure into node/edge schemas → compile the transaction; validate parameter-backed bindings before writing (no injection) |

### Reversible derivations (truth maintenance)

A derived/computed node (a mined finding, a computed capability-index entry, a
materialized view) registered as a live materialization auto-marks `Stale` the
moment a committed change touches something it depends on — dependency-directed,
paraconsistent (a contradiction elsewhere never cascades into unrelated
stale-ing). Requires the opt-in `epistemic-tms` engine feature (both actions
below degrade to `{"error": ...}` without it).

1. Write the derivation with its provenance intact: the node itself
   (`graph_write(action="add_node", ...)`) plus explicit `DERIVED_FROM`/
   `GENERATED_BY` edges to every fact it actually depends on
   (`graph_write(action="add_edge", ..., rel_type="DERIVED_FROM")`). The engine
   reads exactly these edges — not a caller-supplied list — to build the
   dependency set, so get them right.
2. Register once: `engine_query(action="register_materialization",
   params_json='{"derived_id": "<id>"}')` → `{"id", "depends_on", ...}` — treat a
   shorter-than-expected `depends_on` list as a sign the dependency edges weren't
   written correctly.
3. Check freshness later: `engine_query(action="materialization_status",
   params_json='{"id": "<id>"}')` → `"Fresh"` / `"Stale"` / `"Retracted"` / `null`
   (never registered). On `Stale`, re-run whatever produced the derivation and
   re-register — there is no automatic recompute (see limits below).

Honest limits: the live CDC hook that marks dependents `Stale` fires on exactly
two mutation shapes — `RemoveNode`/`RemoveEdge` (→ `Deleted`) and
`CompareAndSetNodeFields` (→ `Updated`). A plain `AddNode` on a dependency is
**not** mapped (no pre-image capture on that path), so adding a brand-new fact
never staleness-marks a materialization that would logically depend on it — only
removing or CAS-updating an existing dependency does. **Nothing currently
consumes staleness automatically** — the engine computes the staled-id set on
every qualifying mutation but drops it after logging; there is no scheduler,
watcher, or Loop stage that reacts to a `Stale` materialization and recomputes it
(a tracked, open gap, not something to paper over). The truth-maintenance index
is in-memory, process-global, and resets on restart — re-register after a
restart if you need continued tracking.

## Workflow

### 1. Model before mutating

- Search the existing ontology and object model for an equivalent type or
  property.
- Extend the canonical model instead of creating a parallel vocabulary.
- Specify identity, required properties, relationship direction, cardinality,
  tenant scope, and lifecycle.
- Distinguish asserted facts from derived facts and memories.

Use `graph_ontology`, `ontology_property_types`, `ontology_value_types`, and
`ontology_interface` to inspect or define the model. Use
`ontology_sampling_profile` to ground a proposed type in representative data.

### 2. Select the mutation surface

| Change | Primary operation |
|---|---|
| Canonical nodes or relationships | `graph_write` |
| Episodic, semantic, spatial, or RL memories | `graph_memory` |
| Concept registration | `concept_registry` |
| Object edits and sets | `object_edits`, `object_set` |
| Object indexes | `object_index` |
| Object access rules | `object_permissioning` |
| Derived values or links | `ontology_derive`, `ontology_link_materialize` |
| Executable ontology logic | `ontology_function` |
| External architecture sync | `ontology_leanix_sync` |

Keep a simple, reversible update direct. Delegate a schema migration or
multi-stage derivation through `graph_workflows` with explicit preconditions,
postconditions, and approval boundaries.

### 3. Validate and preview

- Validate ontology connectivity and constraints before the data write.
- Confirm referenced nodes exist and identifiers are stable.
- Preview affected object counts with a bounded read.
- Define rollback or compensating operations before a high-impact change.

### 4. Apply atomically

For a design, review, or preview-only request, return the proposed model,
affected counts, verification, and rollback plan, then stop before this stage.

- Use a transaction or compare-and-set operation when concurrent writers can
  race.
- Attach provenance, actor, observed time, and evidence to material changes.
- Keep derived artifacts linked to their inputs so they can be recomputed or
  retracted.
- Never broaden permissions as an incidental side effect.

### 5. Verify

- Read the changed objects through the normal query surface.
- Re-run constraints and check that unrelated objects did not change.
- Confirm retries are idempotent.
- Record the mutation outcome and any remaining manual action.

Use an economy model for inventory and deterministic conformance checks. Use a
stronger reasoning model for ontology tradeoffs, migrations, and blast-radius
judgment.

## Guardrails

- Do not bypass policy, approval, or tenant boundaries.
- Do not use unbounded object sets for mutations.
- Do not embed credentials or private source data in ontology definitions.
- Do not claim success from an accepted job alone; verify the resulting state.
