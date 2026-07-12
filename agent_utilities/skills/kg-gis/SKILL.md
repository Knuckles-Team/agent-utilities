---
name: kg-gis
skill_type: skill
description: >-
  Thin verb over the engine's geospatial (GIS) surface — routing, tiles, nearest-
  neighbour, and named geo-tasks. Use for geospatial ops — "route between these points",
  "nearest to this coordinate", "fetch a tile".
license: MIT
tags: [graph-os, engine, gis, geospatial]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-gis

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `graph_gis` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_gis"])` once per session (as below), then proceed exactly as documented; or (2) call the `ask` intent verb with the same natural-language request — the resolver routes to `graph_gis` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


`graph_gis` (CONCEPT:AU-KG.coordination.engine-message-broker) is action-routed 1:1 over the engine geo methods: `route` (`from`+`to`[+`profile`]), `tile` (`z/x/y`), `nearest` (`lat`+`lon`[+`limit`]), `geo_task` (a named geospatial job). All structured args go via `params_json`. Degrades cleanly when the engine build has no GIS surface.

## Invoke
- **MCP:** `load_tools(tools=["graph_gis"])`, then `graph_gis(action="route", params_json='{"from":[51.5,-0.1],"to":[48.8,2.3]}')`.
- **REST twin:** `POST /graph/gis` with `{"action": "nearest", "params_json": "{\"lat\":51.5,\"lon\":-0.1,\"limit\":5}"}`.

## Example
```
graph_gis(action="tile", params_json='{"z":10,"x":511,"y":340}')
```
