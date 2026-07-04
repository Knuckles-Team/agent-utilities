---
name: kg-secret
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

`graph_secret` (CONCEPT:OS-5.66) manages the `__secrets__` store (values sealed by encryption-at-rest; key names + metadata stay queryable). Actions: `set` (`key`+`value` [+`metadata`], GOVERNED by ActionPolicy `secret.set`), `get` (`key`→value or null), `list` (key names only, never values), `delete` (GOVERNED by `secret.delete`). The enterprise OpenBao/Vault backend is used transparently when configured.

## Invoke
- **MCP:** `load_tools(tools=["graph_secret"])`, then `graph_secret(action="get", key="openai/api_key")`.
- **REST twin:** `POST /graph/secret` with `{"action": "set", "key": "svc/token", "value": "...", "reason": "..."}`.

## Example
```
graph_secret(action="list")
```
