---
name: kg-query
description: >-
  Runs a read-only Cypher query against the shared Knowledge Graph to fetch nodes,
  relationships and properties. Use when you need to read graph data directly — "query the
  KG", "run this Cypher", "fetch nodes/edges", "what does the graph say about X".
license: MIT
tags: [graph-os, query, cypher]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-query

`graph_query` executes a **read-only** Cypher statement against the unified KG and returns the result rows. It is the primary read path for graph structure — traversals, property reads, relationship exploration. (Mutations go through `kg-write`.)

## Invoke
- **MCP:** `load_tools(tools=["graph_query"])`, then call `graph_query(query="MATCH (n:Concept) RETURN n.id LIMIT 25")`.
- **REST twin:** `POST /graph/query` with `{"query": "MATCH ... RETURN ...", "limit": 50}`.

## Example
```
graph_query(query="MATCH (c:Concept {id:'KG-2.7'})-[:RELATES_TO]->(x) RETURN x.id, x.name")
```
Prefer natural-language reads via `kg-ask` when you don't want to hand-write Cypher.
