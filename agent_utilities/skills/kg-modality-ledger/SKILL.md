---
name: kg-modality-ledger
skill_type: skill
description: >-
  Read and manage the epistemic-graph engine's append-only audit ledger —
  get/apply/clear ledger entries for tamper-evident change history. Use when you
  need an audit trail or provenance log ("show the audit ledger", "what changed
  and when", "append-only history", "apply a ledger record").
license: MIT
tags: [graph-os, engine, modality, ledger, audit, provenance]
tier: modality
wraps: [engine_ledger]
metadata:
  author: Genius
  version: '0.1.0'
---

# KG Modality — Ledger (append-only audit log)

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `engine_ledger` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["engine_ledger"])` once per session (as below), then proceed exactly as documented; or (2) call the `write` intent verb with the same natural-language request — the resolver routes to `engine_ledger` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


Fronts the epistemic-graph engine's **`ledger`** domain: an append-only audit
ledger recording mutations to the graph for tamper-evident provenance and
compliance. You can read (`get`) the ledger, `apply` records, and `clear` it
(admin) — giving agents a durable "who changed what, when" history distinct from
the live graph state.

This is the **modality** tier — a thin wrapper over the low-level engine surface
(`engine_ledger`), action-routed 1:1 over the `epistemic_graph` client's
`LedgerClient`. The action set is discovered from the client (get/apply/clear);
call with an empty `action` to list the live set.

## How to reach it

**Via the multiplexer:**
1. `load_tools(tools=["engine_ledger"])`.
2. `engine_ledger(action="", params_json="{}")` — list actions.
3. `engine_ledger(action="get", params_json="{...}", graph="")` — invoke.
4. `unload_tools(...)` when done.

**Direct MCP on graph-os:** `engine_ledger` is a registered tool; per-method
verbose tools appear under `MCP_TOOL_MODE=verbose|both`.

**REST twin:** `POST /engine/ledger` with body
`{"action": "<method>", "params_json": "{...}", "graph": ""}`.

## Example

```jsonc
// discover the ledger actions the live engine supports
engine_ledger(action="", params_json="{}")

// read the audit ledger (exact args come from the action list)
engine_ledger(action="get", params_json="{\"limit\": 100}")
```

For higher-level, KG-native audit records (ExecutionSummary / action outcomes),
see the core feedback/persistence skills — this domain is the raw engine ledger.
