---
name: kg-modality-mining
skill_type: skill
description: >-
  Run data-mining over the epistemic-graph engine — association-rule mining
  (Apriori/FP-Growth/Eclat), clustering (DBSCAN/hierarchical/GMM/k-medoids), and
  anomaly detection (z-score/isolation-forest/LOF/one-class-SVM), compute-near-data,
  plus the deep-learning family (LSTM forecast, MLP classify, autoencoder anomaly,
  gradient-boosting, neural embedding) delegated to data-science-mcp. Use for
  "find frequent itemsets", "mine association rules", "cluster these nodes/
  vectors", "detect anomalies/outliers in this series or graph", "forecast this
  series with a neural model", "deep-classify these rows".
license: MIT
tags: [graph-os, engine, modality, mining, clustering, anomaly, data-mining, deep-learning]
tier: modality
wraps: [engine_mining, graph_mine, graph_mine_deep]
metadata:
  author: Genius
  version: '0.2.0'
---

# KG Modality — Mining

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `engine_mining`, `graph_mine`, `graph_mine_deep` are held back from the default tool list (nothing removed — REST + `_execute_tool` still reach them exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["engine_mining"])` once per session (as below), then proceed exactly as documented; or (2) call the `ask` intent verb with the same natural-language request — the resolver routes to `engine_mining` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


Fronts the epistemic-graph engine's `mining` domain (CONCEPT:EG-KG.mining.frequent-itemset-mining):
association-rule mining, clustering, and anomaly detection running **inside the
engine**, compute-near-data, with an optional writeback of results as first-class
graph nodes.

Two surfaces over the same engine `MiningClient`:

- **`engine_mining`** — the raw modality-tier action router (1:1 over the
  client's methods; empty `action` lists what the live engine supports).
- **`graph_mine`** — the friendlier, action-routed wrapper with three fixed
  actions and richer input shapes (baskets, feature matrices, or a
  graph/vector `source` selector):
  - **`associate`** — frequent-itemset + rules (Apriori/FP-Growth/Eclat: support,
    confidence, lift). Provide `transactions` (baskets of item labels) OR a
    graph-derived `source` (`node_label`, `direction`, `item_field`, `relation`,
    `limit`). `writeback: true` ⇒ `:AssociationRule` nodes.
  - **`cluster`** — DBSCAN (default) / hierarchical / GMM / k-medoids over a
    `features` row matrix OR a vector `source` (the stored embeddings of those
    nodes). `writeback: true` ⇒ `:Cluster` nodes linked to members.
  - **`anomaly`** — z-score (default) / isolation-forest / LOF / one-class-SVM
    over `features`, a 1-D `values` series (timeseries RCA), or a vector
    `source`. `writeback: true` ⇒ `:Anomaly` nodes linked to their source.

Degrades cleanly on a no-mining engine build (returns a structured error, never
a crash).

- **`graph_mine_deep`** (CONCEPT:AU-KG.mining.dsm-forecast-delegation) — the
  heavy-Python / deep-learning family the pure-Rust engine deliberately does
  NOT implement (no torch/GPU in-engine): dispatches to `agents/data-science-mcp`
  over MCP and folds the result back into the KG as typed nodes.
  Actions: `deep_forecast` (LSTM sequence forecaster), `deep_classify` (MLP
  classifier), `autoencoder_anomaly` (reconstruction-error outlier detection),
  `xgboost` (gradient-boosting classifier), `embed` (neural embedding of
  numeric feature rows). Every action accepts raw rows OR a graph-derived
  `source` selector; `writeback: true` materializes typed result nodes
  (`:Forecast`/`:Classification`/`:Anomaly`/`:Embedding`) linked back to their
  source. Degrades cleanly (`{available:false, error:...}`) when
  data-science-mcp is unreachable or its `[training]` extra isn't installed.

## How to reach it

**Via the multiplexer:** `load_tools(tools=["graph_mine"])` (or `engine_mining`
for the raw action router, or `graph_mine_deep` for the deep-learning
delegation), then call; `unload_tools(...)` when done.

**REST twins:** `POST /engine/mining` (raw), `POST /api/mining/{associate,
cluster,anomaly}` (natural per-action body, same `_execute_tool` core as
`graph_mine`), and `POST /api/mining/deep/{deep_forecast,deep_classify,
autoencoder_anomaly,xgboost,embed}` (same core as `graph_mine_deep`).

## Example

```jsonc
// association rules from baskets
graph_mine(action="associate",
  params_json="{\"transactions\":[[\"bread\",\"milk\"],[\"bread\",\"butter\"]],\"min_support\":0.5}")

// cluster the stored embeddings of a node set, writing back :Cluster nodes
graph_mine(action="cluster",
  params_json="{\"source\":{\"node_label\":\"Doc\"},\"algorithm\":\"kmedoids\",\"k\":3,\"writeback\":true}")

// anomaly detection over a metric series
graph_mine(action="anomaly",
  params_json="{\"values\":[1,1,1,100],\"algorithm\":\"zscore\"}")
```
