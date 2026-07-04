---
name: kg-sandbox
description: >-
  Inspects and controls the native warm-fork sandbox runtime — status, reap idle warm
  parents, or pre-warm a rung so the next fan-out forks cheaply. Use for warm-fork
  lifecycle/visibility — "sandbox status", "warm the forkserver rung", "reap idle
  sandboxes".
license: MIT
tags: [graph-os, sandbox, warm-fork]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-sandbox

`graph_sandbox` (CONCEPT:ORCH-1.93) manages the RLM warm-fork tier (forkserver/os.fork, Wizer-warmed wasm, warm container pool, firecracker microVM). Actions: `status` (per-rung availability + pooled warm-parent count + per-rung reward EMA), `reap` (close idle warm parents + idle dev-workspaces), `warm` (pre-pay a `rung`'s startup so the next fan-out forks cheaply). Code execution itself stays inside the governed RLM loop; this surface is lifecycle + visibility.

## Invoke
- **MCP:** `load_tools(tools=["graph_sandbox"])`, then `graph_sandbox(action="status")`.
- **REST twin:** `POST /graph/sandbox` with `{"action": "warm", "rung": "forkserver"}`.

## Example
```
graph_sandbox(action="warm", rung="container_fork")
```
