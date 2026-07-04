---
name: kg-reach
description: >-
  Reaches the human user over a messaging backend (Telegram/Slack/Discord…) — routed to
  their last-active channel or an explicit target, governed and mirrored into KG memory.
  Use to notify or message the user — "tell the user…", "message me on Telegram", "reach
  the operator".
license: MIT
tags: [graph-os, messaging, reach]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-reach

`graph_reach` (CONCEPT:AU-ECO.mcp.graph-reach-mcp-tool) sends outbound messages to the user. Actions: `reach_user` (`text` [+`user_id`] → the user's LAST-ACTIVE channel, else the configured default), `send` (explicit `platform`+`channel_id`+`text`), `list_channels` (`platform`), `last_channel` ([`user_id`]→resolved channel), `status`. Every send is governed by the ActionPolicy gate and mirrored into conversational memory.

## Invoke
- **MCP:** `load_tools(tools=["graph_reach"])`, then `graph_reach(action="reach_user", text="Deploy finished")`.
- **REST twin:** `POST /graph/reach` with `{"action": "send", "platform": "telegram", "channel_id": "123", "text": "..."}`.

## Example
```
graph_reach(action="reach_user", text="Ingestion drained: 1053 → 20", reason="status update")
```
