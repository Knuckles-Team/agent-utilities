---
name: kg-epistemic-answer
skill_type: skill
description: >-
  Answers a question and returns its full epistemic justification, not just the row —
  calibrated confidence + provenance/evidence refs + bitemporal valid/tx time (the
  KnowledgeBatch "currency upgrade"), the belief's justification tree (Asserted /
  DerivedSupport / DerivedContradiction / BayesianUpdate), the acceptance capstone
  (believed? since when? on what evidence? what would flip it?), a bitemporal
  what-changed diff between two transaction times, and — when policy hid something —
  which rows were denied and why. Use when an answer alone isn't enough — "why do we
  believe this", "how confident are we and why", "what would invalidate this",
  "what changed between these two points in time", "why was this row filtered out".
license: MIT
tags: [graph-os, epistemic, engine, belief, provenance, confidence, bitemporal]
tier: core
wraps: [graph_query, graph_ask, engine_query, graph_epistemic]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-epistemic-answer

A plain `graph_query`/`graph_ask` gives you rows. This skill layers the engine's
**epistemic currency** on top of those rows via the generic `engine_query` passthrough
(`QueryClient` on the `epistemic-graph` engine) — the same engine-native machinery the
X-7 context compiler and S1 `KnowledgeBatch` reads from, exposed here directly so you
can interrogate ONE answer's justification instead of a whole retrieval bundle.

Four layers, each a real `engine_query` action (`engine_query(action=<name>,
params_json='{"...": ...}')`), progressively deeper and progressively more feature-gated:

1. **Currency upgrade — `explain_provenance_by_ids`** (CONCEPT:AU-KB-CURRENCY / Seam 1).
   Take any id list from ANY prior read (a Cypher `MATCH`, a `graph_ask` result) and
   "upgrade" it to calibrated, cited, time-versioned rows: `{"rows": [{"id", "kind",
   "score", "confidence", "valid_time", "tx_time", "source_refs", "policy_labels",
   "evidence_spans"}, ...], "resolved": bool}`. `resolved: false` means the connected
   engine wasn't built with the base `epistemic` feature — every score/confidence/time
   field is still populated regardless; only the citation/policy columns go empty.
   **In the default `full` build already** — no opt-in flag needed.
2. **Justification tree — `explain_belief`** (`{"node_id": "..."}`). The full
   un-flattened `JustificationGraph` rooted at that node: `{"root": {"claim", "rule",
   "confidence", "premises": [<same shape>, ...]}}`, `rule` one of `"Asserted"` /
   `"DerivedSupport"` / `"DerivedContradiction"` / `"BayesianUpdate"`. **Also in the
   default `full` build.**
3. **Acceptance capstone — `epistemic_status`** (EPI-P3-5, `{"node_id": "..."}`). The
   single typed call that answers "what do we believe, why, on exactly which evidence,
   under whose authority, at what time, with what uncertainty, and what would
   invalidate it." Sibling: **`what_changed`** (`{"tx_from": int, "tx_to": int}`) — a
   whole-graph bitemporal diff between two transaction times, for "what changed
   between deploy N and N+1" rather than one claim's history. **Requires the opt-in
   `epistemic-tms` engine feature — NOT in the default `full` build.** A degrade
   (`{"error": "..."}`) means the connected engine lacks it; treat that as "capstone
   unavailable here," not a crash.
4. **Policy diagnostic — `explain_policy`** (`{"plan": {"ops": [...]}}` — the same
   externally-tagged `Op` list `graph_search`'s planner uses). Runs the plan against
   both the caller's RLS-filtered view and the unfiltered one, returning
   `{"visible_ids": [...], "policy_denied_ids": [...]}` — use this when an expected row
   is missing and you need to know whether policy hid it.

## Two additive surfaces (WS-1a)

- **`include_epistemic` on the read path** — `graph_query(cypher=..., include_epistemic=true)`
  and `graph_ask(question=..., include_epistemic=true)` (cypher dialect only) skip the
  two-step "get ids, then currency-upgrade them" dance above: each result row IS
  already an `EpistemicRow` (confidence/bitemporal window/evidence/policy labels
  alongside the row's own `properties`), resolved server-side in the SAME query call.
  Use this when you just want provenance-aware rows; use the `explain_provenance_by_ids`
  two-step above when you already have an id list from elsewhere.
- **`graph_epistemic`** — a purpose-named wrapper over layers 2-4 above PLUS
  `resolve_conflict` (`{"node_ids": [...], "semantics": "grounded"}` — argumentation-
  based resolution over a set of contradicting claims), so you don't have to remember
  the underlying `engine_query` action names: `graph_epistemic(action="why", node_id=...)`
  = `explain_belief`, `action="status"` = `epistemic_status`, `action="what_changed"`,
  `action="resolve_conflict"`. Same degrade contract (a clean `{"error": ...}` on an
  engine build/config that lacks the action).

## Invoke

- **MCP:** `load_tools(tools=["graph_query", "graph_ask", "engine_query", "graph_epistemic"])`.
- Get the base rows: `graph_query(cypher="MATCH (c:Claim) WHERE c.topic = 'X' RETURN c.id AS id")`
  or `graph_ask(question="...", execute=true)`.
- Currency-upgrade the ids: `engine_query(action="explain_provenance_by_ids", params_json='{"ids": ["claim:1", "claim:2"]}')`
  — or skip straight to provenance-aware rows with `graph_query(cypher="...", include_epistemic=true)`.
- Deep-dive one node: `graph_epistemic(action="why", node_id="claim:1")` (or
  `engine_query(action="explain_belief", params_json='{"node_id": "claim:1"}')`).
- Capstone (if built with `epistemic-tms`): `graph_epistemic(action="status", node_id="claim:1")`.
- **REST twin:** `POST /engine/query` with `{"action": "explain_belief", "params_json": "{\"node_id\": \"claim:1\"}"}`,
  or `POST /epistemic` with `{"action": "why", "node_id": "claim:1"}`.

## Example

```jsonc
// 1. base rows
graph_query(cypher="MATCH (c:Claim {status:'accepted'}) RETURN c.id AS id LIMIT 5")
// -> [{"id": "claim:mine:abc123"}, ...]

// 2. currency upgrade — confidence + provenance + bitemporal time for those exact ids
engine_query(action="explain_provenance_by_ids",
             params_json='{"ids": ["claim:mine:abc123"]}')
// -> {"rows": [{"id": "claim:mine:abc123", "kind": "Claim", "score": 0.0,
//               "confidence": 0.82, "valid_time": 1752000000, "tx_time": 1752000042,
//               "source_refs": ["mining:pass:7"], "policy_labels": [],
//               "evidence_spans": ["ev:1", "ev:2"]}], "resolved": true}

// 3. why do we believe it, and what would flip it (needs epistemic-tms)
engine_query(action="epistemic_status", params_json='{"node_id": "claim:mine:abc123"}')
// -> {"error": "..."} on a build without epistemic-tms — degrade cleanly, don't retry blindly.
```

## Honest limitations

- `explain_provenance_by_ids`/`explain_belief`/`explain_plan`/`explain_policy` are in
  the default `full` engine build. `epistemic_status`/`what_changed` need the opt-in
  `epistemic-tms` feature (not folded into `full`) — check for a degrade response before
  assuming the capstone ran.
- `ExplainBelief`'s policy-aware redaction (`disclosure_level` — `Full` / `Skeleton` /
  `ExistenceOnly`, `epistemic-redaction` engine feature) IS now reachable: pass
  `disclosure_level` to `engine_query(action="explain_belief", ...)` or
  `graph_epistemic(action="why", disclosure_level="Skeleton", ...)`. Omitting it means
  full disclosure (the engine's own default). `kg-compliance`'s bulk `export` action
  applies this per-node across a whole id set.
- These are read-only diagnostics over what the mining/loop/ingestion pipelines already
  wrote — they explain existing belief structure, they do not create it. To grow the
  belief graph, see `kg-mining-flywheel`.

## Delegation

If graph-os is reachable, prefer letting the local LLM assemble a multi-node
justification report via `graph_orchestrate(action="execute_agent")` rather than
hand-chaining many `engine_query` calls — use this skill directly for a single
node/id-list lookup.
