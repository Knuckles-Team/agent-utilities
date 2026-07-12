---
name: kg-research
skill_type: skill
description: >-
  Runs the research assimilation & knowledge-synthesis pipeline — synthesize, deep
  entity/relation extraction, background research, relevance sweeps, citation tracking,
  variant evolution. Use to ingest and synthesize research — "assimilate this paper",
  "deep-extract entities", "spawn background research".
license: MIT
tags: [graph-os, research]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-research

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `graph_research` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_research"])` once per session (as below), then proceed exactly as documented; or (2) call the `ask` intent verb with the same natural-language request — the resolver routes to `graph_research` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


`graph_research` drives the research/assimilation pipeline. Actions: `synthesize` (synthesize knowledge from a source), `deep_extract` (entity/relation extraction), `background_research`/`spawn_background` (background jobs → poll with `graph_ingest(action='status')`), `relevance_sweep`, `research_ingest`, `evolve_variants`, `track_citations`. `query` is the source/topic.

## Invoke
- **MCP:** `load_tools(tools=["graph_research"])`, then `graph_research(action="synthesize", query="https://arxiv.org/abs/…")`.
- **REST twin:** `POST /graph/research` with `{"action": "deep_extract", "query": "..."}`.

## Example
```
graph_research(action="background_research", query="warm-fork microVM isolation")
```

## Delegation
If graph-os is reachable, offload composite multi-step work via `graph_orchestrate` (`execute_agent` / `execute_workflow`) instead of hand-running the steps — let the local LLM + Loop engine do it, and resolve only the exceptions.
