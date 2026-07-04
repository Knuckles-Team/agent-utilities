---
name: kg-fork
description: >-
  Warm-fork fan-out — pay warm-up once for a parent context, then fork N copy-on-write
  branches to run per-branch computations concurrently and collect each result. Use for
  cheap concurrent fan-out — "fork N branches", "run this snippet across n copies", "warm-
  fork fan-out".
license: MIT
tags: [graph-os, engine, fork, fanout]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-fork

`graph_fork` (CONCEPT:AU-KG.coordination.warm-fork-fanout) fans out over the ORCH-1.86..93 warm-fork primitive (LMCache KV / copy-on-write sandboxes). Provide `branches_json` (a JSON list of per-branch snippets) OR `code`+`n` (same snippet across n branches); `vars_json` seeds the shared namespace forked into every branch; `sandbox` optionally pins a rung (`forkserver|container_fork|firecracker`), else the cheapest available warm-fork rung is used. Degrades cleanly (structured `unavailable`) when no warm-fork rung exists on this host.

## Invoke
- **MCP:** `load_tools(tools=["graph_fork"])`, then `graph_fork(code="result = x*2", n=4, vars_json='{"x":10}')`.
- **REST twin:** `POST /graph/fork` with `{"branches_json": "[\"a()\",\"b()\"]", "vars_json": "{}"}`.

## Example
```
graph_fork(branches_json='["score(m1)","score(m2)","score(m3)"]', vars_json='{"dataset":"..."}')
```
