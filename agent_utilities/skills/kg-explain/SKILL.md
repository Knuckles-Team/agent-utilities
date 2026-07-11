---
name: kg-explain
skill_type: skill
description: >-
  The universal context plane — routes a question to its domain provider and returns ONE
  grounded, cited answer (code, ops, deploy, entity/tickets/process). Use to ask
  "why/how/status" about the system — "why is the ops queue backing up", "is my change
  live", "explain this entity".
license: MIT
tags: [graph-os, explain, context]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-explain

> **Condensed intent-surface note (Seam 8).** Under the small/cheap-LLM profile (`MCP_TOOL_MODE=intent`), `graph_explain` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_explain"])` once per session (as below), then proceed exactly as documented; or (2) call the `why` intent verb with the same natural-language request — the resolver routes to `graph_explain` for you and returns the result plus a routing justification. The default `MCP_TOOL_MODE=condensed` is completely unaffected.


`graph_explain` (CONCEPT:AU-KG.retrieval.route-question-its-domain) routes `target='domain:intent'` to the right provider and returns a cited answer. Domains: `code`, `ops` (live task-queue), `deploy` (is my change live, AU-KG.retrieval.kg-2), `entity`/`tickets`/`deploys`/`process` (AU-KG.retrieval.kg-3). `action='explain'` for the answer, `action='context'` for a synthesized context bundle, `target='domains'` to list providers.

## Invoke
- **MCP:** `load_tools(tools=["graph_explain"])`, then `graph_explain(action="explain", target="ops:why")`.
- **REST twin:** `POST /graph/explain` with `{"action": "explain", "target": "deploy:status", "query": "..."}`.

## Example
```
graph_explain(action="explain", target="deploy:status", query="graph-os host change")
```
