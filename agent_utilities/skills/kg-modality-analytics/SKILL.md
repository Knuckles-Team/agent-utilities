---
name: kg-modality-analytics
skill_type: skill
description: >-
  Run graph analytics and data-science kernels on the epistemic-graph engine —
  centrality, (personalized) PageRank, plus estimators, numeric primitives, and
  training kernels. Use when you need graph algorithms or in-engine ML/stats
  ("rank the most central nodes", "run PageRank", "fit an estimator", "compute
  on the graph without exporting to numpy").
license: MIT
tags: [graph-os, engine, modality, analytics, datascience, pagerank, ml]
tier: modality
wraps: [engine_analytics, engine_datascience]
metadata:
  author: Genius
  version: '0.1.0'
---

# KG Modality — Analytics & Data Science

Fronts two epistemic-graph engine domains that compute over graph and tabular
data **inside the engine** (no numpy round-trip):

- **`analytics`** — centrality measures and (personalized) PageRank over the
  live graph.
- **`datascience`** — estimators, numeric primitives, and training kernels (the
  engine's native compute-kernel backend).

This is the **modality** tier: thin wrappers over the low-level engine surface,
each action-routed 1:1 over the `epistemic_graph` client (`AnalyticsClient`,
`DataScienceClient`). The action set is discovered from the client, so it never
drifts — call with an empty `action` to list what the live engine exposes.

## How to reach it

**Via the multiplexer:**
1. `load_tools(tools=["engine_analytics", "engine_datascience"])`.
2. `engine_analytics(action="", params_json="{}")` — list actions.
3. `engine_analytics(action="pagerank", params_json="{...}", graph="")` — invoke.
4. `unload_tools(...)` when done.

**Direct MCP on graph-os:** `engine_analytics` / `engine_datascience` are
registered tools; per-method verbose tools appear under `MCP_TOOL_MODE=verbose|both`.

**REST twins:** `POST /engine/analytics` and `POST /engine/datascience` with body
`{"action": "<method>", "params_json": "{...}", "graph": ""}`.

## Example

```jsonc
// discover the analytics actions the live engine supports
engine_analytics(action="", params_json="{}")

// personalized PageRank seeded on a node set (exact args from the action list)
engine_analytics(action="pagerank",
  params_json="{\"seeds\": [\"node:1\"], \"damping\": 0.85}")
```

For heavier statistical/ML workflows, prefer the `datascience` estimators over
exporting data — the training kernels run in-engine on the same substrate.
