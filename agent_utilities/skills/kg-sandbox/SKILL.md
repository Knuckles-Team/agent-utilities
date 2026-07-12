---
name: kg-sandbox
skill_type: skill
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

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `graph_sandbox` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_sandbox"])` once per session (as below), then proceed exactly as documented; or (2) call the `act` intent verb with the same natural-language request — the resolver routes to `graph_sandbox` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


`graph_sandbox` (CONCEPT:AU-ORCH.sandbox.graph-sandbox-surface) manages the RLM warm-fork tier (forkserver/os.fork, Wizer-warmed wasm, warm container pool, firecracker microVM). Actions: `status` (per-rung availability + pooled warm-parent count + per-rung reward EMA), `reap` (close idle warm parents + idle dev-workspaces), `warm` (pre-pay a `rung`'s startup so the next fan-out forks cheaply). Code execution itself stays inside the governed RLM loop; this surface is lifecycle + visibility.

## Invoke
- **MCP:** `load_tools(tools=["graph_sandbox"])`, then `graph_sandbox(action="status")`.
- **REST twin:** `POST /graph/sandbox` with `{"action": "warm", "rung": "forkserver"}`.

## Example
```
graph_sandbox(action="warm", rung="container_fork")
```
