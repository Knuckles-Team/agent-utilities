---
name: kg-kvcache
skill_type: skill
description: >-
  Thin verb over the engine's shared, content-addressed KV-cache — get/put opaque blocks,
  check membership, and read occupancy/dedup stats. Use for the shared block cache —
  "cache this block", "is this key cached", "kvcache stats".
license: MIT
tags: [graph-os, engine, kvcache]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-kvcache

`graph_kvcache` (CONCEPT:AU-KG.coordination.engine-message-broker, via the KG-2.306 EpistemicGraphKVBackend) is the engine's content-addressed KV-cache. Actions: `get` (`key`→base64 block bytes or miss), `put` (`key`+`value_b64`→stored bool), `contains`/`exists` (`key`→bool), `stats` (occupancy + dedup counters). The connector degrades every transport error to a cache miss, so it never raises.

## Invoke
- **MCP:** `load_tools(tools=["graph_kvcache"])`, then `graph_kvcache(action="stats")`.
- **REST twin:** `POST /graph/kvcache` with `{"action": "get", "key": "..."}`.

## Example
```
graph_kvcache(action="put", key="blk:abc", value_b64="...")
```
