---
name: kg-explain
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

`graph_explain` (CONCEPT:KG-2.136) routes `target='domain:intent'` to the right provider and returns a cited answer. Domains: `code`, `ops` (live task-queue), `deploy` (is my change live, KG-2.138), `entity`/`tickets`/`deploys`/`process` (KG-2.139). `action='explain'` for the answer, `action='context'` for a synthesized context bundle, `target='domains'` to list providers.

## Invoke
- **MCP:** `load_tools(tools=["graph_explain"])`, then `graph_explain(action="explain", target="ops:why")`.
- **REST twin:** `POST /graph/explain` with `{"action": "explain", "target": "deploy:status", "query": "..."}`.

## Example
```
graph_explain(action="explain", target="deploy:status", query="graph-os host change")
```
