---
name: kg-mux-use
skill_type: skill
description: >-
  Discover and mount fleet MCP tools on demand through the mcp-multiplexer
  meta-tools (find_tools / list_catalog / load_tools / unload_tools /
  multiplexer_status). Use when a capability you need is not in your current
  tool list ("I don't see a tool for X", "load the servicenow tools", "what
  tools exist", "mount/unmount tools", "the multiplexer is in dynamic mode").
license: MIT
tags: [graph-os, meta, multiplexer, tools, discovery]
tier: meta
metadata:
  author: Genius
  version: '0.1.0'
---

# KG Mux — Use (discover + mount fleet tools on demand)

The **mcp-multiplexer** aggregates the whole agent-package fleet — dozens of
child MCP servers, hundreds of tools — behind one server, tool names namespaced
by a short per-server prefix (e.g. `go__` for graph-os). In **dynamic mode**
(`MCP_MULTIPLEXER_MODE=dynamic`) only a small subset (a few meta-tools plus the
always-on servers) is loaded at any time; the rest exist but are **not** mounted
yet. So a tool not being in your current list does **not** mean it is
unavailable — discover and mount it.

This is a **meta** skill: it describes the workflow, not a single verb. It wraps
nothing and adds no verb coverage.

## The five meta-tools

- **`find_tools(query)`** — semantic search for the right tool by intent.
- **`list_catalog()`** — browse every child server and its tools.
- **`load_tools(tools=[...] | servers=[...])`** — mount specific tools or whole
  servers; they become directly callable immediately (the live tool list
  updates).
- **`unload_tools(...)`** — retract tools to reclaim context window.
- **`multiplexer_status`** — health of the mounted child servers.

## Workflow

1. **Don't assume absence.** If you need a capability you don't see, start with
   `find_tools("<what you want to do>")` (or `list_catalog()` to browse).
2. **Mount** the matches: `load_tools(tools=["engine_blob", "graph_query"])` —
   or a whole server: `load_tools(servers=["servicenow-mcp"])`.
3. **Call** the now-mounted tools directly, as if they were always present.
4. **Unmount** when finished with `unload_tools(...)` to keep the context lean.
5. If a mounted tool misbehaves, check `multiplexer_status` for child health.

## Example

```jsonc
find_tools(query="content-addressed blob storage")
load_tools(tools=["engine_blob"])
engine_blob(action="", params_json="{}")   // now callable
unload_tools(tools=["engine_blob"])
```

Related: use `kg-mux-extend` to add a brand-new child MCP server to the fleet so
its tools become discoverable here.
