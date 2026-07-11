---
name: kg-placement-advisor
skill_type: skill
description: >-
  Triggers the workload-aware placement mining + measured-canary control loop: mines
  agent-trace co-occurrence (tenant/tool/entity/modality access skew) into typed,
  evidenced PlacementProposals (shard_split / replica / cache_prewarm / materialized_join
  / embedding_refresh / index_change), runs them through the SAME claim-flywheel +
  action-policy governance as any other autonomous change, canaries an approved change,
  and promotes or rolls back on the measured SLO delta. Use to right-size sharding/
  caching/indexing from REAL observed access patterns — "run the placement mining
  loop", "propose a shard/cache/index change from load", "canary this placement change".
license: MIT
tags: [graph-os, evolution, placement, mining, canary, sharding, engine]
tier: core
wraps: [graph_loops, graph_query, graph_orchestrate, engine_resharding]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-placement-advisor

The X-5 placement-mining pipeline (CONCEPT:AU-KG.evolution.placement-mining-canary-loop)
mirrors `kg-mining-flywheel`'s pattern exactly: mine → typed `PlacementProposal` →
persist as a `Claim` (`status="proposal"`, always, `is_verified=False`) →
`PromotionGovernanceValidator.validate()` → `action_policy.decide(kind=
"apply_placement_change")` (shipped `approval_required` by default — nothing
auto-applies) → only if allowed, a **measured canary** (apply small, measure the SLO
delta, promote or roll back) → on promote, the change reaches the engine's
`PlacementCatalog` admin path (`ReshardingClient.catalog_assign`/`catalog_remove`, the
SAME admin surface `engine_resharding` exposes — no second placement authority).

`graph_loops(action="placement_control")` is the **manual trigger** for one pass of
this whole pipeline — call it directly instead of hand-running the mining/governance/
canary steps yourself.

## Procedure

1. **Trigger one governed pass**: `graph_loops(action="placement_control",
   placement_scan_limit=200, placement_canary_tolerance=0.10)`.
   - `placement_scan_limit` — provenance row-scan cap for the mining pass.
   - `placement_canary_tolerance` — fraction the canary metric may regress by and
     still be promoted (SLO noise tolerance), default `0.10`.
2. **Inspect what it proposed**: `graph_query(cypher="MATCH (c:Claim) WHERE
   c.source = 'placement_mining' RETURN c ORDER BY c.created_at DESC LIMIT 20")`
   (adjust the property filter to your deployment's actual claim shape — cross-check
   with `kg-mining-flywheel`'s `ClaimLifecycleEvent` query for the same claim ids).
3. **If a proposal is stuck awaiting approval** (the default `approval_required`
   policy tier for `apply_placement_change`): review it via the fleet approval queue
   and grant it with `graph_orchestrate(action="grant_approval", ...)` — see
   `kg-orchestrate`.
4. **Confirm the landed placement**: `engine_resharding(action="catalog_list",
   params_json="{}")` — lists the live `(graph, shard)` catalog assignments the
   `PlacementCatalog` currently holds. Requires the `kg:admin` scope (`resharding`
   is an ADMIN domain).

## Invoke

- **MCP:** `load_tools(tools=["graph_loops", "graph_query", "graph_orchestrate", "engine_resharding"])`.
- **REST twin:** `POST /graph/loops` with `{"action": "placement_control", "placement_scan_limit": 200, "placement_canary_tolerance": 0.1}`.

## Example

```jsonc
// 1. trigger one pass
graph_loops(action="placement_control", placement_scan_limit=200, placement_canary_tolerance=0.10)
// -> {"action": "placement_control", "result": {"mined": 3, "proposed": 3,
//     "validated": 2, "canaried": 1, "promoted": 0, "rolled_back": 0, "errors": []}}

// 2. confirm current catalog state (requires kg:admin)
engine_resharding(action="catalog_list", params_json="{}")
// -> {"assignments": [{"graph": "code:agent-utilities", "shard": 2}, ...]}
```

## Honest limitations

- **This is the manual trigger, not an automatic one.** Nothing in the codebase runs a
  placement-mining pass on a schedule — this skill (or a future scheduler) is the only
  way to fire a pass today. If you want continuous placement tuning, you must call this
  action periodically yourself (e.g. via `kg-schedules`) — there is no `KG_LOOP_*`-style
  always-on flag for it.
- `engine_resharding` is an **ADMIN domain** (`tenants`/`resharding`/`consensus`/`rbac`/
  `admin`) — `catalog_list`/`catalog_assign`/`catalog_remove` are denied to an acting
  identity without the `kg:admin` scope/role, fail-closed.
- The default `apply_placement_change` action-policy tier is `approval_required` —
  expect most proposals to sit in the approval queue rather than auto-promote; that is
  intentional, not a bug.
- The exact `:Claim`/property shape a placement proposal persists under is internal and
  may evolve — treat the Cypher in step 2 as a starting query to adapt, not a guaranteed
  stable schema.

## Delegation

For a fully autonomous placement-tuning cadence, wire this trigger into
`kg-schedules` (`graph_schedules action=run_now` on a named schedule) rather than
calling `placement_control` by hand every time.
