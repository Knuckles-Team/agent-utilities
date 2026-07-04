---
name: kg-bus
description: >-
  The federated agent-to-agent bus — let this session discover and message other
  Claude/LLM sessions (any provider/host) and dispatch objectives to the fleet through one
  graph-os hub. Use for cross-session/host coordination — "who's online", "message that
  agent", "dispatch this to the fleet", "subscribe to a topic".
license: MIT
tags: [graph-os, messaging, bus, a2a]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-bus

`graph_bus` (CONCEPT:AU-ECO.bus.agent-to-agent-bus) is the durable, cross-host agent bus (state lives in the KG). Actions: `register`/`heartbeat`/`leave`/`status`, `roster` (discover peers + presence), `send` (`sender`+`payload`+`to|topic`), `receive` (+`since` cursor), `subscribe`/`unsubscribe`, `ack`, `dispatch` (hand an objective to the fleet as a Loop). Mesh/federation: `register_hub`/`list_hubs`/`federate`/`federate_in`. Store-and-forward + auto-presence: any action keeps you online and rosterable.

## Invoke
- **MCP:** `load_tools(tools=["graph_bus"])`, then `graph_bus(action="roster", online_only=true)`.
- **REST twin:** `POST /graph/bus` with `{"action": "send", "sender": "me", "to": "peer", "payload": "..."}`.

## Example
```
graph_bus(action="dispatch", sender="me", objective="Reingest the fleet", kind="develop", priority="high")
```
