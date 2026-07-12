---
name: kg-analyze
skill_type: skill
description: >-
  Runs residual ops/structural analysis across the KG — inspect, enrichment coverage,
  process write-back (Camunda/ARIS), workload placement plan, infra sweep, security scan.
  Use for structural/ops analysis — "inspect this subgraph", "placement plan", "security
  scan". (Code→kg-code, research→kg-research, eval→kg-evaluate, Q&A→kg-explain.)
license: MIT
tags: [graph-os, analyze]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-analyze

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `graph_analyze` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_analyze"])` once per session (as below), then proceed exactly as documented; or (2) call the `ask` intent verb with the same natural-language request — the resolver routes to `graph_analyze` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


`graph_analyze` handles the ops/structural analysis actions: `inspect`, `enrichment_coverage`, `process_writeback` (push KG intelligence into Camunda/ARIS; `target=camunda|aris|both`), `placement_plan` (workload placement, KG-2.9), `infra_sweep`, `security_scan`. Codebase, research, evaluation and Q&A intents are routed to their own verbs (`kg-code`, `kg-research`, `kg-evaluate`, `kg-explain`).

## Invoke
- **MCP:** `load_tools(tools=["graph_analyze"])`, then `graph_analyze(action="inspect", query="...")`.
- **REST twin:** `POST /graph/analyze` with `{"action": "placement_plan", "query": "..."}`.

## Example
```
graph_analyze(action="security_scan", query="services/")
```

## Delegation
If graph-os is reachable, offload composite multi-step work via `graph_orchestrate` (`execute_agent` / `execute_workflow`) instead of hand-running the steps — let the local LLM + Loop engine do it, and resolve only the exceptions.
