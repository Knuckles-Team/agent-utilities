---
name: kg-share
description: >-
  Explicitly promotes a private-by-default node's visibility — share with my org, promote
  a copy into the shared commons, attach a mandatory marking, or restrict it back to
  private. Use for tenancy/sharing — "share this node with my org", "promote to commons",
  "mark it", "make it private again".
license: MIT
tags: [graph-os, tenancy, sharing]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-share

`graph_share` (CONCEPT:KG-2.60) is the explicit promotion path for data that is private-to-its-owner by default. Actions: `org` (share with the owner's org in-place), `commons` (promote a copy into the shared cross-org commons graph), `mark` (attach a mandatory `marking`), `private` (restrict back). Actor/owner is the ambient identity — never caller-supplied.

## Invoke
- **MCP:** `load_tools(tools=["graph_share"])`, then `graph_share(action="org", node_id="doc:42")`.
- **REST twin:** `POST /graph/share` with `{"action": "commons", "node_id": "doc:42"}`.

## Example
```
graph_share(action="mark", node_id="doc:42", marking="confidential")
```
