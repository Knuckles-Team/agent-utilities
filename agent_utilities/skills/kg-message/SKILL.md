---
name: kg-message
skill_type: skill
description: >-
  Opens ephemeral (optionally durable) agent-to-agent message channels for a run — send,
  receive, replay history, close. Use for intra-run coordination between a session and its
  spawned agents — "open a channel", "send/receive on the channel". (For cross-
  session/host fleet messaging use kg-bus.)
license: MIT
tags: [graph-os, messaging, channels]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-message

`graph_message` manages per-run message channels. Actions: `open` (from a `session_id`/`run_id` → `channel_id`), `send` (`channel_id` + `sender` + `payload`, `durable=true` to persist as a replayable `AgentMessage` node), `receive` (with a `since` cursor), `history`, `close`.

## Invoke
- **MCP:** `load_tools(tools=["graph_message"])`, then `graph_message(action="send", channel_id="c1", payload="done")`.
- **REST twin:** `POST /graph/message` with `{"action": "receive", "channel_id": "c1", "since": 0}`.

## Example
```
graph_message(action="open", session_id="s1", run_id="r1")
graph_message(action="receive", channel_id="c1", since=0)
```
