---
name: kg-promql
skill_type: skill
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

> **Condensed intent-surface note (Seam 8).** Under the small/cheap-LLM profile (`MCP_TOOL_MODE=intent`), `graph_promql` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_promql"])` once per session (as below), then proceed exactly as documented; or (2) call the `ask` intent verb with the same natural-language request — the resolver routes to `graph_promql` for you and returns the result plus a routing justification. The default `MCP_TOOL_MODE=condensed` is completely unaffected.


`graph_promql` (CONCEPT:AU-KG.coordination.engine-message-broker) queries engine metrics with PromQL. `action='instant'` (single evaluation at `time`, default now) or `action='range'` (over `start`..`end` at `step`). Extra engine kwargs via `params_json`. Degrades cleanly when the engine build has no metrics surface.

## Invoke
- **MCP:** `load_tools(tools=["graph_promql"])`, then `graph_promql(query="up", action="instant")`.
- **REST twin:** `POST /graph/promql` with `{"query": "rate(http_requests_total[5m])", "action": "range", "start": "...", "end": "...", "step": "30s"}`.

## Example
```
graph_promql(query="engine_queue_depth", action="instant")
```
