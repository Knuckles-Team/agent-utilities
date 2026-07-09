---
name: kg-modality-mining
skill_type: skill
description: >-
  Run data-mining over the epistemic-graph engine — association-rule mining
  (Apriori/FP-Growth/Eclat), clustering (DBSCAN/hierarchical/GMM/k-medoids), and
  anomaly detection (z-score/isolation-forest/LOF/one-class-SVM), compute-near-data.
  Use for "find frequent itemsets", "mine association rules", "cluster these nodes/
  vectors", "detect anomalies/outliers in this series or graph".
license: MIT
tags: [graph-os, engine, modality, mining, clustering, anomaly, data-mining]
tier: modality
wraps: [engine_mining, graph_mine]
metadata:
  author: Genius
  version: '0.1.0'
---

# KG Modality — Mining

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

## How to reach it

**Via the multiplexer:** `load_tools(tools=["graph_mine"])` (or `engine_mining`
for the raw action router), then call; `unload_tools(...)` when done.

**REST twins:** `POST /engine/mining` (raw) and `POST /api/mining/{associate,
cluster,anomaly}` (natural per-action body, same `_execute_tool` core as
`graph_mine`).

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
