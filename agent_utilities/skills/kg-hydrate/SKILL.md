---
name: kg-hydrate
skill_type: skill
description: >-
  Full-hydrates the KG from an external source (or all configured sources) — a thin alias
  of source_sync with mode=full. Use to (re)mirror a connector's data — "hydrate leanix",
  "full sync all sources", "rebuild from the source".
license: MIT
tags: [graph-os, ingestion, hydrate]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-hydrate

`graph_hydrate` re-mirrors an external source into the KG at `mode=full`. `source` is any registered connector (e.g. `leanix`, `servicenow`, `gitlab`), or `all` to fan out to the fleet-wide sweep (CONCEPT:AU-KG.ingest.enterprise-source-extractor). It delegates to the same unified `source_sync` core, so there is no divergent hydration logic — use `kg-etl`/`source_sync` for delta/reconcile modes.

## Invoke
- **MCP:** `load_tools(tools=["graph_hydrate"])`, then `graph_hydrate(source="leanix")`.
- **REST twin:** `POST /graph/hydrate` with `{"source": "all"}`.

## Example
```
graph_hydrate(source="servicenow")
```
