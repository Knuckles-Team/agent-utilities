---
name: kg-write
skill_type: skill
description: >-
  The primary KG mutation interface — add/delete nodes and edges, bulk-ingest, atomic
  compare-and-set, store/recall memory, log chat, register external graphs. Use to change
  graph data — "add a node/edge", "bulk ingest", "conditionally update", "store a memory".
license: MIT
tags: [graph-os, write, mutation]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-write

`graph_write` is the write path for the KG. Actions: `add_node`, `add_edge`, `delete_node`, `delete_edge`, `register_external_graph`, `bulk_ingest`, `compare_and_set` (atomic conditional update — applies `updates` only if every field in `conditions` still matches, for optimistic concurrency and safe concurrent graph-shaping), `store_memory`, `recall_memory`, `recall_media`, `log_chat`, `submit_sdd`, `register_execution`, `check_loop`.

## Invoke
- **MCP:** `load_tools(tools=["graph_write"])`, then `graph_write(action="add_node", node_id="x", node_type="Concept", properties='{"name":"X"}')`.
- **REST twin:** `POST /graph/write` with `{"action": "add_edge", "source_id": "a", "target_id": "b", ...}`.

## Example
```
graph_write(action="compare_and_set", node_id="task:1",
            properties='{"conditions":{"status":"open"},"updates":{"status":"claimed"}}')
```
