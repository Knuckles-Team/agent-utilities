---
name: kg-promql
description: >-
  Thin verb over the engine's observability metrics — run instant or ranged PromQL
  queries. Use for metric queries — "query this PromQL", "CPU over the last hour",
  "instant value of a metric".
license: MIT
tags: [graph-os, engine, observability, metrics]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-promql

`graph_promql` (CONCEPT:KG-2.310) queries engine metrics with PromQL. `action='instant'` (single evaluation at `time`, default now) or `action='range'` (over `start`..`end` at `step`). Extra engine kwargs via `params_json`. Degrades cleanly when the engine build has no metrics surface.

## Invoke
- **MCP:** `load_tools(tools=["graph_promql"])`, then `graph_promql(query="up", action="instant")`.
- **REST twin:** `POST /graph/promql` with `{"query": "rate(http_requests_total[5m])", "action": "range", "start": "...", "end": "...", "step": "30s"}`.

## Example
```
graph_promql(query="engine_queue_depth", action="instant")
```
