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

`graph_ask` turns a plain-English `question` into a read-only query and runs it, returning the generated query (auditable) plus result rows. `dialect` is `auto` by default (the model picks) or force `cypher`/`sql`/`sparql`; set `execute=false` to preview the generated query without running it.

## Invoke
- **MCP:** `load_tools(tools=["graph_ask"])`, then `graph_ask(question="How many Concept nodes exist?")`.
- **REST twin:** `POST /graph/ask` with `{"question": "...", "dialect": "auto", "execute": true, "limit": 50}`.

## Example
```
graph_ask(question="Which agents depend on the mcp-multiplexer?", execute=true)
```
