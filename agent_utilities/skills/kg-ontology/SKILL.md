---
name: kg-ontology
skill_type: skill
description: >-
  The full ontology + object layer — host/validate OWL/RDF ontologies, manage
  property/value types, interfaces, functions, derived properties, link materialization,
  LeanIX metamodel sync, and the Foundry-style object layer (edits ledger, index,
  permissioning, object sets). Use for ontology/type/object-model work — "load this
  ontology", "validate a value type", "materialize a link", "redact an object", "build an
  object set".
license: MIT
tags: [graph-os, ontology, objects]
tier: core
wraps: [graph_ontology, ontology_property_types, ontology_value_types, ontology_interface, ontology_sampling_profile, ontology_function, ontology_derive, ontology_link_materialize, ontology_leanix_sync, object_edits, object_index, object_permissioning, object_set]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-ontology

This skill fronts the whole ontology/object surface:
- **`graph_ontology`** — hosted-ontology lifecycle CRUD (`load`/`list`/`get`/`update`/`delete`/`validate`/`activate`/`deactivate`/`sync_packages`; SHACL-validated, versioned, native-reasoner-loaded).
- **`ontology_property_types`** / **`ontology_value_types`** — the type registry (list/describe/validate/coerce).
- **`ontology_interface`** — interfaces: `implementers`, `conforms`, `owl` (`registry='enterprise'` for standard contracts).
- **`ontology_sampling_profile`** — task-aware LLM sampling profiles (list/describe/resolve/set/evolve/owl).
- **`ontology_function`** — typed versioned functions (`list`/`invoke`).
- **`ontology_derive`** — compute derived properties live at read time.
- **`ontology_link_materialize`** — reify a many-to-many link as a junction triple.
- **`ontology_leanix_sync`** — mirror the live LeanIX metamodel as OWL/RDF (`dry_run` first).
- **`object_edits`** — durable object-edit ledger (record/revert/history/as_of, optimistic `expect`).
- **`object_index`** — search-index lifecycle (`sync`/`reindex`/`status`).
- **`object_permissioning`** — `redact`/`restricted_view`/`mark` (ambient actor).
- **`object_set`** — Foundry-style object sets (`of_type`/`search`/`filter`/`pivot`/`aggregate`/`union`/`intersect`/`subtract`).

## Invoke
- **MCP:** `load_tools(tools=["graph_ontology"])` (or any listed verb), then call it.
- **REST twin:** `POST /graph/ontology`, `POST /ontology/<...>`, `POST /object/<...>` with a JSON body carrying `action` + args.

## Example
```
graph_ontology(action="load", source="/path/to/domain.ttl", source_type="file")
object_set(action="filter", ...)
```
