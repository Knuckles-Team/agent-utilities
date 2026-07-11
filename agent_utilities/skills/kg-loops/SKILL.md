---
name: kg-loops
skill_type: skill
description: >-
  The single entrypoint for long-running objectives (research/develop/skill Loops) plus
  observing and steering the self-evolution flywheel. Use to run or steer Loops and
  evolution — "submit a Loop", "drive this Loop", "show evolution state", "review a spec
  proposal".
license: MIT
tags: [graph-os, loops, evolution]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-loops

> **Condensed intent-surface note (Seam 8).** Under the small/cheap-LLM profile (`MCP_TOOL_MODE=intent`), `graph_loops` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_loops"])` once per session (as below), then proceed exactly as documented; or (2) call the `act` intent verb with the same natural-language request — the resolver routes to `graph_loops` for you and returns the result plus a routing justification. The default `MCP_TOOL_MODE=condensed` is completely unaffected.


`graph_loops` (CONCEPT:AU-KG.research.these-properties-carry) is one entrypoint for Loops of kind `research|develop|skill`. Actions: `submit` (objective + kind [+ validation_cmd/end_state for develop, skill_ref for skill]), `list`, `run` (advance all active Loops one cycle), `drive` (run ONE Loop by id to completion, durably/resumable), `cancel`, `prioritize`. Transparency + steering: `state` (live EvolutionState — stage + why, saturation, open-gaps trend, velocity, backlog), `specs` (SpecProposal backlog), `review` (approve|edit|reject a distilled spec before it develops).

## Invoke
- **MCP:** `load_tools(tools=["graph_loops"])`, then `graph_loops(action="state")`.
- **REST twin:** `POST /graph/loops` with `{"action": "submit", "objective": "...", "kind": "develop", "validation_cmd": "pytest -q"}`.

## Example
```
graph_loops(action="review", loop_id="spec:123", ...)  # approve|edit|reject a distilled spec
```

## Delegation
If graph-os is reachable, offload composite multi-step work via `graph_orchestrate` (`execute_agent` / `execute_workflow`) instead of hand-running the steps — let the local LLM + Loop engine do it, and resolve only the exceptions.
