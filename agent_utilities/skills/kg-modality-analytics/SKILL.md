---
name: kg-modality-analytics
skill_type: skill
description: >-
  Run graph analytics, data-science kernels, and KAN graph-learning on the
  epistemic-graph engine — centrality, (personalized) PageRank, estimators,
  numeric primitives, training kernels, and a learned per-feature edge-function
  link predictor. Use when you need graph algorithms, in-engine ML/stats, or
  link prediction ("rank the most central nodes", "run PageRank", "fit an
  estimator", "fit/predict a graph-learning model", "compute on the graph
  without exporting to numpy").
license: MIT
tags: [graph-os, engine, modality, analytics, datascience, pagerank, ml, graphlearn]
tier: modality
wraps: [engine_analytics, engine_datascience, engine_graphlearn]
metadata:
  author: Genius
  version: '0.2.0'
---

# KG Modality — Analytics & Data Science

Fronts three epistemic-graph engine domains that compute over graph and
tabular data **inside the engine** (no numpy round-trip):

- **`analytics`** — centrality measures and (personalized) PageRank over the
  live graph.
- **`datascience`** — estimators, numeric primitives, and training kernels (the
  engine's native compute-kernel backend).
- **`graphlearn`** (AU-P0-6) — a pure-Rust KAN (Kolmogorov-Arnold) link
  predictor learned over the resident graph: `fit`/`predict` with queryable,
  interpretable per-feature edge functions (not a black-box scorer).

This is the **modality** tier: thin wrappers over the low-level engine surface,
each action-routed 1:1 over the `epistemic_graph` client (`AnalyticsClient`,
`DataScienceClient`, `GraphLearnClient`). The action set is discovered from the
client, so it never drifts — call with an empty `action` to list what the live
engine exposes.

## How to reach it

**Via the multiplexer:**
1. `load_tools(tools=["engine_analytics", "engine_datascience", "engine_graphlearn"])`.
2. `engine_analytics(action="", params_json="{}")` — list actions.
3. `engine_analytics(action="pagerank", params_json="{...}", graph="")` — invoke.
4. `unload_tools(...)` when done.

**Direct MCP on graph-os:** `engine_analytics` / `engine_datascience` /
`engine_graphlearn` are registered tools; per-method verbose tools appear under
`MCP_TOOL_MODE=verbose|both`.

**REST twins:** `POST /engine/analytics`, `POST /engine/datascience`, and
`POST /engine/graphlearn` with body
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
