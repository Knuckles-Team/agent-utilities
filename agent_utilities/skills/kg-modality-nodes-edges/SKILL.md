---
name: kg-modality-nodes-edges
skill_type: skill
description: >-
  Low-level core graph CRUD on the epistemic-graph engine — nodes, edges, graph
  operations, and lifecycle (prune/decay/evict, batch update, context views,
  (de)serialize). Use when you need byte-level graph mutation/read below the
  curated graph_* verbs ("add/get/batch nodes", "temporal edge invalidate",
  "AST parse/index", "prune or evict the graph", "raw graph CRUD").
license: MIT
tags: [graph-os, engine, modality, nodes, edges, graph, lifecycle, crud]
tier: modality
wraps: [engine_nodes, engine_edges, engine_graph, engine_lifecycle]
metadata:
  author: Genius
  version: '0.1.0'
---

# KG Modality — Nodes / Edges / Graph / Lifecycle (core CRUD)

Fronts the four epistemic-graph engine domains that make up **core graph CRUD**
at the wire level:

- **`nodes`** — node CRUD, batch/union reads, degree/neighbour queries.
- **`edges`** — edge CRUD, temporal invalidate/supersede, batch reads.
- **`graph`** — graph algorithms, AST parse/index, semantic + embedding compute.
- **`lifecycle`** — prune/decay/evict, `batch_update`, context view,
  (de)serialize.

These are the low-level primitives beneath the curated high-level `graph_*`
verbs (`kg-query`, `kg-write`, etc.). Reach for this modality skill when you
need direct, method-level control the synthesized verbs don't expose — bulk
batch operations, temporal edge supersession, AST indexing, or lifecycle
maintenance. For everyday reads/writes, prefer the core `graph_*` skills.

This is the **modality** tier — thin wrappers, each action-routed 1:1 over the
`epistemic_graph` client (`NodeClient`, `EdgeClient`, `GraphOperationsClient`,
`LifecycleClient`). The action set is discovered from the client, so it never
drifts — call any tool with an empty `action` to list the live set.

## How to reach it

**Via the multiplexer:**
1. `load_tools(tools=["engine_nodes", "engine_edges", "engine_graph", "engine_lifecycle"])`.
2. `engine_nodes(action="", params_json="{}")` — list actions.
3. `engine_nodes(action="get_batch", params_json="{...}", graph="")` — invoke.
4. `unload_tools(...)` when done.

**Direct MCP on graph-os:** `engine_nodes` / `engine_edges` / `engine_graph` /
`engine_lifecycle` are registered tools; per-method verbose tools appear under
`MCP_TOOL_MODE=verbose|both`.

**REST twins:** `POST /engine/nodes`, `/engine/edges`, `/engine/graph`,
`/engine/lifecycle` with body
`{"action": "<method>", "params_json": "{...}", "graph": ""}`.

## Example

```jsonc
// discover the node actions the live engine supports
engine_nodes(action="", params_json="{}")

// batch-read nodes by id, then a temporal edge invalidate (args from action list)
engine_nodes(action="get_batch", params_json="{\"ids\": [\"n:1\", \"n:2\"]}")
engine_edges(action="invalidate", params_json="{\"edge_id\": \"e:9\"}")
```
