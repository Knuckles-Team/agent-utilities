---
name: kg-mining-flywheel
skill_type: skill
description: >-
  Runs the governed mining→claim→review→act flywheel: a research Loop cycle mines
  Episode/ToolCall/OutcomeEvaluation provenance and discovery content, turns findings
  above a confidence floor into reviewable Claims with a real, terminal-retraction-aware
  lifecycle (proposed→validated→accepted→deprecated/retracted), gated by the SAME
  promotion-governance + action-policy checks every autonomous action passes through —
  then inspect the resulting lifecycle. Use to grow and govern the KG's own belief base
  from evidence — "mine findings into reviewable claims", "run the discovery flywheel",
  "what got proposed/accepted/retracted this cycle".
license: MIT
tags: [graph-os, evolution, mining, claim-flywheel, loops, governance]
tier: core
wraps: [graph_loops, graph_query]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-mining-flywheel

The `ClaimFlywheel` (CONCEPT:AU-KG.evolution.mining-flywheel, workstream C4/C6) is a
five-state lifecycle over mining-produced Claims: `proposed → validated → accepted →
deprecated / retracted` (any pre-terminal state may also retract directly).
**`RETRACTED` is terminal and sticky** — a rejected finding is never silently
re-proposed on a later mining pass over the same content-addressed finding id.

It is a **thin overlay**, not a second governance stack: a claim only reaches
`validated` because `PromotionGovernanceValidator` said so, and only reaches
`accepted` because `action_policy.decide()` independently allowed it — the same
gates every other autonomous action passes through. Every transition is an
append-only `ClaimLifecycleEvent` node (never a silent mutation of the Claim's own
fields), so the full history is queryable.

**There is no direct `flywheel.propose()`/`.accept()` tool call** — the flywheel only
runs as a byproduct of a research Loop cycle's `trace_mining` and `insight_validation`
stages (both default **ON**). This skill's job is to trigger a cycle and then read the
lifecycle it produced.

## Procedure

1. **Submit (or reuse) a research Loop**: `graph_loops(action="submit", kind="research",
   objective="...")` — or skip this and just call `run`/`drive` against an already-active
   Loop (`graph_loops(action="list")` to find one).
2. **Advance a cycle**: `graph_loops(action="run", max_topics=5)` (advances every active
   Loop one cycle) or `graph_loops(action="drive", loop_id="...")` (drives ONE Loop to
   completion). The returned report has `trace_mining`/`insight_validation` sub-reports
   (best-effort — a failing stage never aborts the cycle) plus a `metrics.stage_ms`
   timing breakdown.
3. **Inspect what the flywheel actually produced**:
   `graph_query(cypher="MATCH (e:ClaimLifecycleEvent) WHERE e.claim_id = $id RETURN e ORDER BY e.at", params='{"id": "<claim_id>"}')`
   or, to see everything proposed/accepted this run, query `:Claim` nodes by
   `status`/`is_verified` directly.
4. **Cross-reference with the justification layer**: once you have a claim id,
   `kg-epistemic-answer` (`engine_query action=explain_belief`) gives you its full
   justification tree; `kg-evidence-cite` gives you its source loci.

## Invoke

- **MCP:** `load_tools(tools=["graph_loops", "graph_query"])`.
- **REST twin:** `POST /graph/loops` with `{"action": "run", "max_topics": 5}`; `POST /graph/query`.

## Example

```jsonc
// 1. submit a research loop (or reuse an active one)
graph_loops(action="submit", kind="research", objective="mine repeated tool-call failures")
// -> {"loop": {"id": "loop:research:abc", ...}}

// 2. advance it one cycle — trace_mining + insight_validation run by default
graph_loops(action="run", max_topics=5)
// -> {"trace_mining": {...}, "insight_validation": {...},
//     "belief_revision": {...}, "metrics": {"stage_ms": {...}}, "errors": []}

// 3. what happened to a specific mined claim?
graph_query(cypher="MATCH (e:ClaimLifecycleEvent) WHERE e.claim_id = $id RETURN e.from_state, e.to_state, e.at ORDER BY e.at",
            params='{"id": "claim:mine:abc123"}')
// -> [{"from_state": null, "to_state": "proposed", "at": ...},
//     {"from_state": "proposed", "to_state": "validated", "at": ...},
//     {"from_state": "validated", "to_state": "accepted", "at": ...}]
```

## Honest limitations

- No standalone tool reaches the flywheel's own `propose`/`validate`/`accept`/`reject`/
  `deprecate`/`retract` methods directly — only running a whole Loop cycle exercises the
  state machine. If you need finer control than "run a cycle and see what came out,"
  that control does not exist on any AU/MCP surface today.
- Both stages are gated `KG_LOOP_TRACE_MINING`/`KG_LOOP_INSIGHT_VALIDATION` (default ON)
  and can be overridden per-`run` call via `graph_loops`'s own params where exposed
  (`mine_discovery` is the one explicitly parameterized override on the tool today;
  `trace_mining`/`insight_validation` follow the server-side config default and are not
  independently toggleable per MCP call).
- `trace_mining` is explicitly safety-critical and propose-only regardless of its flag —
  it never calls `OutcomeRouter.record()` before `action_policy.decide()` gates it.
- This is a governance-and-observability skill, not a raw mining tool — for ad hoc
  association-rule/clustering/anomaly mining outside the flywheel's governance, use
  `kg-modality-mining` (`graph_mine`) instead.
