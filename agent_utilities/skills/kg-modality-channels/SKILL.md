---
name: kg-modality-channels
skill_type: skill
description: >-
  Create and use dynamic pub/sub communication channels AND the native
  message-broker (exchanges/queues/streams) in the epistemic-graph engine —
  publish, subscribe, route messages between agents, and declare/bind/consume
  broker exchanges and queues at the engine layer. Use when you need low-level
  pub/sub or agent messaging fabric ("open a channel", "publish/subscribe",
  "route a message between agents at the engine", "declare a queue", "publish
  to an exchange").
license: MIT
tags: [graph-os, engine, modality, channels, pubsub, messaging, broker]
tier: modality
wraps: [engine_channels, engine_broker]
metadata:
  author: Genius
  version: '0.2.0'
---

# KG Modality — Channels & Broker (dynamic pub/sub + message broker)

> **Condensed intent-surface note (Seam 8).** Under the small/cheap-LLM profile (`MCP_TOOL_MODE=intent`), `engine_channels`, `engine_broker` are held back from the default tool list (nothing removed — REST + `_execute_tool` still reach them exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["engine_channels"])` once per session (as below), then proceed exactly as documented; or (2) call the `act` intent verb with the same natural-language request — the resolver routes to `engine_channels` for you and returns the result plus a routing justification. The default `MCP_TOOL_MODE=condensed` is completely unaffected.


Fronts two epistemic-graph engine communication domains:

- **`channels`** — dynamic agent-communication channels: create channels,
  publish messages, and subscribe/consume, at the engine layer. This is the
  raw pub/sub fabric beneath higher-level messaging, useful when you want
  engine-native fan-out between producers and consumers on the same substrate
  as the graph.
- **`broker`** (AU-P0-6) — the engine's native RabbitMQ/Kafka-class message
  broker: exchange/queue/stream admin, routed publish (incl. confirmed/
  idempotent variants), and consumer-group ack/nack.

This is the **modality** tier — thin wrappers over the low-level engine surface
(`engine_channels` / `engine_broker`), each action-routed 1:1 over the
`epistemic_graph` client's `ChannelsClient` / `BrokerClient`. The action set is
discovered from the client (create, publish, subscribe, list, declare_queue, …);
call with an empty `action` to list the live set.

## How to reach it

**Via the multiplexer:**
1. `load_tools(tools=["engine_channels", "engine_broker"])`.
2. `engine_channels(action="", params_json="{}")` — list actions.
3. `engine_channels(action="publish", params_json="{...}", graph="")` — invoke.
4. `unload_tools(...)` when done.

**Direct MCP on graph-os:** `engine_channels` / `engine_broker` are registered
tools; per-method verbose tools appear under `MCP_TOOL_MODE=verbose|both`.

**REST twins:** `POST /engine/channels` and `POST /engine/broker` with body
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
