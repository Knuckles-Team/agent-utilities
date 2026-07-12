---
name: kg-reach
skill_type: skill
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

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `graph_reach` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_reach"])` once per session (as below), then proceed exactly as documented; or (2) call the `act` intent verb with the same natural-language request — the resolver routes to `graph_reach` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


`graph_reach` (CONCEPT:AU-ECO.mcp.graph-reach-mcp-tool) sends outbound messages to the user. Actions: `reach_user` (`text` [+`user_id`] → the user's LAST-ACTIVE channel, else the configured default), `send` (explicit `platform`+`channel_id`+`text`), `list_channels` (`platform`), `last_channel` ([`user_id`]→resolved channel), `status`. Every send is governed by the ActionPolicy gate and mirrored into conversational memory.

## Invoke
- **MCP:** `load_tools(tools=["graph_reach"])`, then `graph_reach(action="reach_user", text="Deploy finished")`.
- **REST twin:** `POST /graph/reach` with `{"action": "send", "platform": "telegram", "channel_id": "123", "text": "..."}`.

## Example
```
graph_reach(action="reach_user", text="Ingestion drained: 1053 → 20", reason="status update")
```
