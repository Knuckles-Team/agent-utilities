---
name: kg-reversible-derive
skill_type: skill
description: >-
  Registers a derived/computed node (a mined finding, a computed capability-index entry,
  a materialized view) as a LIVE truth-maintenance materialization, straight off its own
  provenance edges — so it auto-marks Stale when a fact it depends on changes, instead of
  silently going wrong. Use after writing ANY derived node when you want it to stay
  honest as its inputs change — "make this derivation reversible", "track staleness of
  this computed result", "did the facts behind this materialized thing change".
license: MIT
tags: [graph-os, epistemic, reversible-intelligence, truth-maintenance, engine, provenance]
tier: core
wraps: [graph_write, engine_query]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-reversible-derive

X-6 "reversible intelligence": a derived node registered this way is tracked on the
engine's `TruthMaintenance` index and flips to `Stale` the moment a committed change
touches something it depends on — dependency-directed, paraconsistent (a contradiction
elsewhere never cascades into unrelated stale-ing).

## Procedure

1. **Write the derivation** with its provenance intact — the node itself
   (`graph_write(action="add_node", ...)`) plus explicit `:DerivedFrom`/`:GeneratedBy`
   edges to every fact it actually depends on
   (`graph_write(action="add_edge", source_id="<derived_id>", target_id="<dep_id>",
   rel_type="DERIVED_FROM")`). The engine reads exactly these edges (plus an
   `invalidation_deps` property, if you set one) to build the dependency set — it does
   **not** trust a caller-supplied list, so get the edges right.
2. **Register it once**: `engine_query(action="register_materialization",
   params_json='{"derived_id": "<derived_id>"}')`. Returns `{"id", "depends_on",
   "generating_activity"}` — `depends_on` is the dependency set the engine ACTUALLY
   resolved from step 1's edges, so treat a shorter-than-expected list as a sign the
   `DERIVED_FROM`/`GENERATED_BY` edges weren't written correctly.
3. **Check freshness later**: `engine_query(action="materialization_status",
   params_json='{"id": "<derived_id>"}')` → `{"status": "Fresh" | "Stale" |
   "Retracted" | null}` (`null` = never registered).
4. **On `Stale`**: re-run whatever produced the derivation and re-register (step 2
   again) — there is no automatic recompute today (see limitations).

## Invoke

- **MCP:** `load_tools(tools=["graph_write", "engine_query"])`.
- **REST twin:** `POST /engine/query` with `{"action": "register_materialization", "params_json": "{\"derived_id\": \"...\"}"}`.

## Example

```jsonc
// 1. write the derived node + its real dependency edges
graph_write(action="add_node", node_id="cap_index:svc-payments",
            node_type="CapabilityIndexEntry", properties='{"score": 0.91}')
graph_write(action="add_edge", source_id="cap_index:svc-payments",
            target_id="service:payments", rel_type="DERIVED_FROM")
graph_write(action="add_edge", source_id="cap_index:svc-payments",
            target_id="capability:process-refunds", rel_type="DERIVED_FROM")

// 2. register it as a live materialization
engine_query(action="register_materialization",
             params_json='{"derived_id": "cap_index:svc-payments"}')
// -> {"id": "cap_index:svc-payments",
//     "depends_on": ["service:payments", "capability:process-refunds"],
//     "generating_activity": null}

// 3. later — did anything it depends on change?
engine_query(action="materialization_status",
             params_json='{"id": "cap_index:svc-payments"}')
// -> {"status": "Fresh"}   (or "Stale" once service:payments or the capability changed)
```

## Honest limitations

- Requires the opt-in `epistemic-tms` engine feature (not in the default `full` build) —
  both actions degrade to `{"error": ...}` without it.
- The live CDC hook that marks dependents `Stale` fires on exactly two mutation shapes:
  `RemoveNode`/`RemoveEdge` (→ `Deleted`) and `CompareAndSetNodeFields` (→ `Updated`). A
  plain `AddNode` on a dependency is **not** mapped — there is no pre-image capture on
  that path — so adding a brand-new fact never staleness-marks a materialization that
  would logically depend on it; only removing or CAS-updating an existing dependency does.
- **Nothing currently consumes staleness automatically.** The engine computes the staled-
  id set on every qualifying mutation but drops it after logging — there is no
  scheduler, watcher, or `graph_loops` stage that reacts to a materialization going
  `Stale` and recomputes it for you. Step 4 above (poll `materialization_status`, then
  manually redo the derivation) is the whole story today; building an automatic
  recompute-on-stale loop is a real, open gap (tracked in the evolution roadmap as
  closing the TMS control loop), not something this skill can paper over.
- The truth-maintenance index itself is in-memory, process-global, and resets on
  restart — a registered materialization does not survive an engine restart; re-register
  after a restart if you need continued tracking.
