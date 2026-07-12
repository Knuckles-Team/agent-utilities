---
name: kg-observe
skill_type: skill
description: >-
  Reasons over the KG-native observability subgraph — traces, online scores, assertion
  verdicts, prompt versions — for root-cause and regression analysis an opaque trace store
  can't do. Use for eval/trace analytics — "root-cause these failures", "which prompt
  version regressed", "cluster failing traces".
license: MIT
tags: [graph-os, observe, eval]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-observe

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `graph_observe` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_observe"])` once per session (as below), then proceed exactly as documented; or (2) call the `why` intent verb with the same natural-language request — the resolver routes to `graph_observe` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


`graph_observe` (CONCEPT:AU-KG.ingest.observability-queries-opik-cannot) queries the trace/score subgraph. Actions: `trace_rootcause` (failed assertions + low scores joined to their trace's agent, grouped; `query`=agent/capability filter), `prompt_regression` (mean score per prompt version — which regressed), `failure_cluster` (failing traces clustered by the failed assertion — systemic breaks across agents).

## Invoke
- **MCP:** `load_tools(tools=["graph_observe"])`, then `graph_observe(action="trace_rootcause", query="chat-agent")`.
- **REST twin:** `POST /graph/observe` with `{"action": "prompt_regression"}`.

## Example
```
graph_observe(action="failure_cluster")
```
