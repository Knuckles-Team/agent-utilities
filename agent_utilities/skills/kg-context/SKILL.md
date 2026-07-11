---
name: kg-context
skill_type: skill
description: >-
  Stores and retrieves session-scoped context blobs in the KG so work survives across
  turns/agents. Use to persist or recall working context — "remember this for the
  session", "store/get context", "list session context".
license: MIT
tags: [graph-os, context, session]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-context

> **Condensed intent-surface note (Seam 8).** Under the small/cheap-LLM profile (`MCP_TOOL_MODE=intent`), `graph_context` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_context"])` once per session (as below), then proceed exactly as documented; or (2) call the `ask` intent verb with the same natural-language request — the resolver routes to `graph_context` for you and returns the result plus a routing justification. The default `MCP_TOOL_MODE=condensed` is completely unaffected.


`graph_context` is a small session-anchored key/value store backed by `ContextBlob` nodes. Actions: `put` (store `content` under a `session_id`/`key`, optional `ttl_s`), `get` (by `context_id`), `list` (by session). Stored blobs are linked to a `Session` node for reliable id-anchored retrieval.

## Invoke
- **MCP:** `load_tools(tools=["graph_context"])`, then `graph_context(action="put", content="...", session_id="s1")`.
- **REST twin:** `POST /graph/context` with `{"action": "put", "content": "...", "session_id": "s1"}`.

## Example
```
graph_context(action="put", content="user prefers efficient mode", session_id="s1", key="prefs")
graph_context(action="list", session_id="s1")
```
