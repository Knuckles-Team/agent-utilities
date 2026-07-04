---
name: kg-configure
description: >-
  Manages graph-os backend configuration and credentials — register MCP servers,
  secrets/vault sync, named graph connections, schema packs, harness fences, doctors and
  DB provisioning. Use for platform config — "register this MCP", "add a graph
  connection", "run config doctor", "vault sync".
license: MIT
tags: [graph-os, configure, admin]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-configure

`graph_configure` manages backend configuration and abstract credentials. Actions include `register_mcp`, `set_secret`/`vault_sync`, `add_connection`/`remove_connection`/`list_connections`/`set_default_connection` (named external graph backends, KG-2.63), `schema_pack`/`schema_candidates`, `harness_fence` (Claude Code permission fence), `install_hooks`, `get_config`/`set_config`/`list_config`, `system_doctor`/`config_doctor`/`preflight`, `setup_databases`/`verify_databases`, `generate_config`.

## Invoke
- **MCP:** `load_tools(tools=["graph_configure"])`, then `graph_configure(action="list_connections")`.
- **REST twin:** `POST /graph/configure` with `{"action": "add_connection", "config_key": "neo4j-prod", "config_value": "{...}"}`.

## Example
```
graph_configure(action="system_doctor", config_value='{"fix":false}')
```
