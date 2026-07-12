---
name: kg-modality-txn
skill_type: skill
description: >-
  Run server-side ACID transactions across modalities on the epistemic-graph
  engine — begin/commit/rollback with optimistic concurrency control. Use when
  you need atomic multi-step or cross-modal writes ("do these writes atomically",
  "transaction", "commit or rollback", "OCC", "all-or-nothing update").
license: MIT
tags: [graph-os, engine, modality, txn, acid, transactions]
tier: modality
wraps: [engine_txn]
metadata:
  author: Genius
  version: '0.1.0'
---

# KG Modality — Transactions (server-side ACID / OCC)

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `engine_txn` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["engine_txn"])` once per session (as below), then proceed exactly as documented; or (2) call the `act` intent verb with the same natural-language request — the resolver routes to `engine_txn` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


Fronts the epistemic-graph engine's **`txn`** domain: server-side ACID
transactions using optimistic concurrency control (OCC). A transaction can span
modalities — graph nodes/edges, tabular, time-series, blob — and commits
all-or-nothing, so multi-step mutations stay consistent under concurrency
without pessimistic locking.

This is the **modality** tier — a thin wrapper over the low-level engine surface
(`engine_txn`), action-routed 1:1 over the `epistemic_graph` client's
`TxnClient`. The action set is discovered from the client (begin, commit,
rollback, and the staged read/write ops within a transaction); call with an
empty `action` to list the live set.

## How to reach it

**Via the multiplexer:**
1. `load_tools(tools=["engine_txn"])`.
2. `engine_txn(action="", params_json="{}")` — list actions.
3. `engine_txn(action="begin", params_json="{...}", graph="")` — invoke, then
   stage ops and `commit`/`rollback` using the returned transaction handle.
4. `unload_tools(...)` when done.

**Direct MCP on graph-os:** `engine_txn` is a registered tool; per-method verbose
tools appear under `MCP_TOOL_MODE=verbose|both`.

**REST twin:** `POST /engine/txn` with body
`{"action": "<method>", "params_json": "{...}", "graph": ""}`.

## Example

```jsonc
// discover the transaction actions the live engine supports
engine_txn(action="", params_json="{}")

// begin → stage → commit (exact args + handle shape come from the action list)
engine_txn(action="begin", params_json="{}")
engine_txn(action="commit", params_json="{\"txn_id\": \"…\"}")
```

On an OCC conflict the commit is rejected — retry the transaction rather than
partially applying it.
