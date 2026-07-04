---
name: kg-goals
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

`graph_goals` orchestrates autonomous goal loops. Actions: `create` (a `goal` description + `max_iterations`), `list`, `iterations` (progress for a `goal_id`), `cancel`.

## Invoke
- **MCP:** `load_tools(tools=["graph_goals"])`, then `graph_goals(action="create", goal="Keep the ingestion lane drained")`.
- **REST twin:** `POST /graph/goals` with `{"action": "list"}`.

## Example
```
graph_goals(action="create", goal="Monitor and heal maint-lane throughput", max_iterations=20)
```
