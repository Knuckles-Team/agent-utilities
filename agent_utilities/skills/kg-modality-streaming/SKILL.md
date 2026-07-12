---
name: kg-modality-streaming
skill_type: skill
description: >-
  Subscribe to change-data-capture and continuous queries on the epistemic-graph
  engine — streams, watches, and triggers that fire as the graph changes. Use
  when you need push/reactive data ("watch for changes", "CDC feed", "continuous
  query", "fire a trigger when X is written", "tail the graph").
license: MIT
tags: [graph-os, engine, modality, streaming, cdc, triggers]
tier: modality
wraps: [engine_streaming]
metadata:
  author: Genius
  version: '0.1.0'
---

# KG Modality — Streaming (CDC / continuous queries)

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `engine_streaming` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["engine_streaming"])` once per session (as below), then proceed exactly as documented; or (2) call the `act` intent verb with the same natural-language request — the resolver routes to `engine_streaming` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


Fronts the epistemic-graph engine's **`streaming`** domain: change-data-capture
streams, continuous (standing) queries, watches, and triggers. Instead of
polling, you register interest and the engine pushes matching changes — turning
the graph into a reactive substrate for event-driven agents and pipelines.

This is the **modality** tier — a thin wrapper over the low-level engine surface
(`engine_streaming`), action-routed 1:1 over the `epistemic_graph` client's
`StreamingClient`. The action set is discovered from the client (register/watch
a continuous query, install a trigger, poll/consume a stream, …); call with an
empty `action` to list the live set.

## How to reach it

**Via the multiplexer:**
1. `load_tools(tools=["engine_streaming"])`.
2. `engine_streaming(action="", params_json="{}")` — list actions.
3. `engine_streaming(action="watch", params_json="{...}", graph="")` — invoke.
4. `unload_tools(...)` when done.

**Direct MCP on graph-os:** `engine_streaming` is a registered tool; per-method
verbose tools appear under `MCP_TOOL_MODE=verbose|both`.

**REST twin:** `POST /engine/streaming` with body
`{"action": "<method>", "params_json": "{...}", "graph": ""}`.

## Example

```jsonc
// discover the streaming actions the live engine supports
engine_streaming(action="", params_json="{}")

// register a continuous query / watch (exact args come from the action list)
engine_streaming(action="watch",
  params_json="{\"query\": \"MATCH (n:Alert) RETURN n\"}")
```
