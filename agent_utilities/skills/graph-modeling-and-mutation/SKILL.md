---
name: graph-modeling-and-mutation
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
