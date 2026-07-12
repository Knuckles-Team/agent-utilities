---
name: kg-goals
skill_type: skill
description: >-
  Creates and manages background/autonomous goal loops that iterate toward an objective.
  Use for long-running autonomous objectives — "kick off a goal", "list goals", "show
  iterations", "cancel this goal".
license: MIT
tags: [graph-os, goals]
tier: core
wraps: [graph_goals, spec_ticket]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-goals

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `graph_goals`, `spec_ticket` are held back from the default tool list (nothing removed — REST + `_execute_tool` still reach them exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_goals"])` once per session (as below), then proceed exactly as documented; or (2) call the `act` intent verb with the same natural-language request — the resolver routes to `graph_goals` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


`graph_goals` orchestrates autonomous goal loops. Actions: `create` (a `goal` description + `max_iterations`), `list`, `iterations` (progress for a `goal_id`), `cancel`.

## Invoke
- **MCP:** `load_tools(tools=["graph_goals"])`, then `graph_goals(action="create", goal="Keep the ingestion lane drained")`.
- **REST twin:** `POST /graph/goals` with `{"action": "list"}`.

## Example
```
graph_goals(action="create", goal="Monitor and heal maint-lane throughput", max_iterations=20)
```
