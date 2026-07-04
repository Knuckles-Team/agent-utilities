---
name: kg-schedules
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

`graph_schedules` drives the unified scheduling engine (`:Schedule` nodes enqueue jobs). Actions: `list`, `enable`, `disable`, `prioritize` (bucket 0-3 or `critical|high|normal|background`), `set_interval` (new `interval_s`), `run_now`.

## Invoke
- **MCP:** `load_tools(tools=["graph_schedules"])`, then `graph_schedules(action="list")`.
- **REST twin:** `POST /graph/schedules` with `{"action": "run_now", "name": "rss-sweep"}`.

## Example
```
graph_schedules(action="set_interval", name="rss-sweep", interval_s=900)
```
