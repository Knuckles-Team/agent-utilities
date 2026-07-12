---
name: kg-modality-timeseries
skill_type: skill
description: >-
  Append and query native time-series data in the epistemic-graph engine's TSDB
  — append points, range/window scans, as-of lookups, and gap-filling. Use when
  you need time-series storage or analytics next to the graph ("record metric
  points", "query a time range", "as-of value at time T", "gap-fill a series").
license: MIT
tags: [graph-os, engine, modality, timeseries, tsdb, metrics]
tier: modality
wraps: [engine_timeseries]
metadata:
  author: Genius
  version: '0.1.0'
---

# KG Modality — Time Series (native TSDB)

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `engine_timeseries` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["engine_timeseries"])` once per session (as below), then proceed exactly as documented; or (2) call the `write` intent verb with the same natural-language request — the resolver routes to `engine_timeseries` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


Fronts the epistemic-graph engine's **`timeseries`** domain: a native
time-series database co-located with the graph. Supports append, range and
window scans, as-of (point-in-time) lookups, and gap-filling — so temporal
metrics live on the same substrate as the entities they describe, queryable
cross-modally alongside graph and tabular data.

This is the **modality** tier — a thin wrapper over the low-level engine surface
(`engine_timeseries`), action-routed 1:1 over the `epistemic_graph` client's
`TimeSeriesClient`. The action set is discovered from the client (append, range,
window, asof, gapfill, …); call with an empty `action` to list the live set.

## How to reach it

**Via the multiplexer:**
1. `load_tools(tools=["engine_timeseries"])`.
2. `engine_timeseries(action="", params_json="{}")` — list actions.
3. `engine_timeseries(action="range", params_json="{...}", graph="")` — invoke.
4. `unload_tools(...)` when done.

**Direct MCP on graph-os:** `engine_timeseries` is a registered tool; per-method
verbose tools appear under `MCP_TOOL_MODE=verbose|both`.

**REST twin:** `POST /engine/timeseries` with body
`{"action": "<method>", "params_json": "{...}", "graph": ""}`.

## Example

```jsonc
// discover the TSDB actions the live engine supports
engine_timeseries(action="", params_json="{}")

// append a point and read a window (exact args come from the action list)
engine_timeseries(action="append",
  params_json="{\"series\": \"cpu.util\", \"ts\": 1700000000, \"value\": 0.42}")
engine_timeseries(action="window",
  params_json="{\"series\": \"cpu.util\", \"start\": 1699990000, \"end\": 1700000000}")
```
