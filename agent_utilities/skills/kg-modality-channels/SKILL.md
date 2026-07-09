---
name: kg-modality-channels
skill_type: skill
description: >-
  Create and use dynamic pub/sub communication channels in the epistemic-graph
  engine — publish, subscribe, and route messages between agents at the engine
  layer. Use when you need low-level pub/sub or agent messaging fabric ("open a
  channel", "publish/subscribe", "route a message between agents at the engine").
license: MIT
tags: [graph-os, engine, modality, channels, pubsub, messaging]
tier: modality
wraps: [engine_channels]
metadata:
  author: Genius
  version: '0.1.0'
---

# KG Modality — Channels (dynamic pub/sub)

Fronts the epistemic-graph engine's **`channels`** domain: dynamic
agent-communication channels — create channels, publish messages, and
subscribe/consume — at the engine layer. This is the raw pub/sub fabric beneath
higher-level messaging, useful when you want engine-native fan-out between
producers and consumers on the same substrate as the graph.

This is the **modality** tier — a thin wrapper over the low-level engine surface
(`engine_channels`), action-routed 1:1 over the `epistemic_graph` client's
`ChannelsClient`. The action set is discovered from the client (create,
publish, subscribe, list, …); call with an empty `action` to list the live set.

## How to reach it

**Via the multiplexer:**
1. `load_tools(tools=["engine_channels"])`.
2. `engine_channels(action="", params_json="{}")` — list actions.
3. `engine_channels(action="publish", params_json="{...}", graph="")` — invoke.
4. `unload_tools(...)` when done.

**Direct MCP on graph-os:** `engine_channels` is a registered tool; per-method
verbose tools appear under `MCP_TOOL_MODE=verbose|both`.

**REST twin:** `POST /engine/channels` with body
`{"action": "<method>", "params_json": "{...}", "graph": ""}`.

## Example

```jsonc
// discover the channel actions the live engine supports
engine_channels(action="", params_json="{}")

// publish to a channel (exact args come from the action list)
engine_channels(action="publish",
  params_json="{\"channel\": \"alerts\", \"message\": {\"kind\": \"ping\"}}")
```

For the federated, cross-host agent-to-agent bus (durable `:Agent`/`:Topic`),
use the core `kg-bus` skill; this domain is the engine-native channel primitive.
