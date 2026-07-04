---
name: kg-broker
description: >-
  Thin verb over the engine's AMQP-style message broker — declare exchanges/queues, bind,
  publish, consume, and inspect stats. Use for engine-level queue/stream messaging
  (distinct from the agent bus) — "publish to this exchange", "consume the queue",
  "declare a queue".
license: MIT
tags: [graph-os, engine, broker, messaging]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-broker

`graph_broker` (CONCEPT:AU-KG.coordination.engine-message-broker) is action-routed 1:1 over the engine broker surface (exchanges + queues + streams), distinct from the agent-to-agent `kg-bus`. Set `action` to the broker method: `declare_exchange` (+`exchange_type`), `declare_queue`, `bind` (`queue`+`exchange`[+`routing_key`]), `publish` (`exchange`+`routing_key`+`payload`), `consume` (`queue`[+max_messages,ack via `params_json`]), `stats`/`list_queues`/`list_exchanges`. Degrades cleanly when the engine build has no broker.

## Invoke
- **MCP:** `load_tools(tools=["graph_broker"])`, then `graph_broker(action="publish", exchange="ex", routing_key="k", payload="hi")`.
- **REST twin:** `POST /graph/broker` with `{"action": "consume", "queue": "q", "params_json": "{\"max_messages\":10,\"ack\":true}"}`.

## Example
```
graph_broker(action="declare_queue", queue="jobs")
```
