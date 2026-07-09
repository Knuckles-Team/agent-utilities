---
name: kg-traces
skill_type: skill
description: >-
  Thin verb over the engine's distributed-trace surface — search traces by
  service/operation/free-form filter, or fetch one by trace id. Use for trace lookups —
  "find traces for this service", "get trace <id>", "slow spans".
license: MIT
tags: [graph-os, engine, observability, traces]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-traces

`graph_traces` (CONCEPT:AU-KG.coordination.engine-message-broker) searches or fetches distributed traces. `action='search'` (filter by `service`/`operation`/free-form `query`, capped by `limit`) or `action='get'` (a single `trace_id`). Extra engine kwargs via `params_json`. Degrades cleanly when the engine build has no trace surface.

## Invoke
- **MCP:** `load_tools(tools=["graph_traces"])`, then `graph_traces(action="search", service="graph-os", limit=20)`.
- **REST twin:** `POST /graph/traces` with `{"action": "get", "trace_id": "..."}`.

## Example
```
graph_traces(action="search", operation="ingest", query="error")
```
