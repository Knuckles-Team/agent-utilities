---
name: kg-schedules
skill_type: skill
description: >-
  Manages the durable scheduler — list, enable/disable, prioritize, retune the interval
  of, or run-now recurring schedules. Use for periodic jobs — "list schedules", "disable
  the sweep", "run this schedule now", "change the interval".
license: MIT
tags: [graph-os, schedules]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-schedules

> **Condensed intent-surface note (Seam 8).** Under the small/cheap-LLM profile (`MCP_TOOL_MODE=intent`), `graph_schedules` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_schedules"])` once per session (as below), then proceed exactly as documented; or (2) call the `manage` intent verb with the same natural-language request — the resolver routes to `graph_schedules` for you and returns the result plus a routing justification. The default `MCP_TOOL_MODE=condensed` is completely unaffected.


`graph_schedules` drives the unified scheduling engine (`:Schedule` nodes enqueue jobs). Actions: `list`, `enable`, `disable`, `prioritize` (bucket 0-3 or `critical|high|normal|background`), `set_interval` (new `interval_s`), `run_now`.

## Invoke
- **MCP:** `load_tools(tools=["graph_schedules"])`, then `graph_schedules(action="list")`.
- **REST twin:** `POST /graph/schedules` with `{"action": "run_now", "name": "rss-sweep"}`.

## Example
```
graph_schedules(action="set_interval", name="rss-sweep", interval_s=900)
```
