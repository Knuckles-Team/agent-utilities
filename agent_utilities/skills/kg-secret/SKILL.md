---
name: kg-secret
skill_type: skill
description: >-
  Manages secrets in the durable, engine-encrypted secret store — set/get/list/delete,
  with values sealed at rest and mutations governed. Use for secret handling — "store this
  credential", "get the API key", "list secret names", "delete a secret".
license: MIT
tags: [graph-os, security, secret]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-secret

> **Condensed intent-surface note (Seam 8).** Under the small/cheap-LLM profile (`MCP_TOOL_MODE=intent`), `graph_secret` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_secret"])` once per session (as below), then proceed exactly as documented; or (2) call the `manage` intent verb with the same natural-language request — the resolver routes to `graph_secret` for you and returns the result plus a routing justification. The default `MCP_TOOL_MODE=condensed` is completely unaffected.


`graph_secret` (CONCEPT:AU-OS.identity.encrypted-secret-store) manages the `__secrets__` store (values sealed by encryption-at-rest; key names + metadata stay queryable). Actions: `set` (`key`+`value` [+`metadata`], GOVERNED by ActionPolicy `secret.set`), `get` (`key`→value or null), `list` (key names only, never values), `delete` (GOVERNED by `secret.delete`). The enterprise OpenBao/Vault backend is used transparently when configured.

## Invoke
- **MCP:** `load_tools(tools=["graph_secret"])`, then `graph_secret(action="get", key="openai/api_key")`.
- **REST twin:** `POST /graph/secret` with `{"action": "set", "key": "svc/token", "value": "...", "reason": "..."}`.

## Example
```
graph_secret(action="list")
```
