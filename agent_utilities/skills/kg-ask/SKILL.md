---
name: kg-ask
skill_type: skill
description: >-
  Answers a natural-language question over the Knowledge Graph by generating and executing
  a query (Cypher/SQL/SPARQL) for you. Use when you want KG data but would rather ask in
  plain English — "ask the graph…", "how many…", "list the… " without writing Cypher.
license: MIT
tags: [graph-os, query, nl]
tier: core
wraps: [graph_ask, ask_data]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-ask

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `graph_ask`, `ask_data` are held back from the default tool list (nothing removed — REST + `_execute_tool` still reach them exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_ask"])` once per session (as below), then proceed exactly as documented; or (2) call the `ask` intent verb with the same natural-language request — the resolver routes to `graph_ask` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


`graph_ask` turns a plain-English `question` into a read-only query and runs it, returning the generated query (auditable) plus result rows. `dialect` is `auto` by default (the model picks) or force `cypher`/`sql`/`sparql`; set `execute=false` to preview the generated query without running it.

## Invoke
- **MCP:** `load_tools(tools=["graph_ask"])`, then `graph_ask(question="How many Concept nodes exist?")`.
- **REST twin:** `POST /graph/ask` with `{"question": "...", "dialect": "auto", "execute": true, "limit": 50}`.

## Example
```
graph_ask(question="Which agents depend on the mcp-multiplexer?", execute=true)
```
