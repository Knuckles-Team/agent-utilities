---
name: kg-orchestrate
description: >-
  Orchestrates multi-agent work — dispatch/swarm agents, execute agents & workflows,
  approvals, consensus/debate, computer-use, DSPy optimization, skill distillation, loop
  cycles. Use to delegate or coordinate execution — "dispatch an agent", "run this
  workflow", "swarm this goal".
license: MIT
tags: [graph-os, orchestrate]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-orchestrate

`graph_orchestrate` is the fleet execution entrypoint. Key actions: `dispatch`, `swarm` (goal→decompose→parallel-waves→verify→synthesize), `execute_agent`, `execute_workflow`/`compile_workflow`, `status`, `request_approval`/`grant_approval`, `consensus`/`start_debate`, `computer_use` (GUI agent on a gui-sandbox), `optimize_component` (DSPy pass), `distill_skills`, `loop_cycle`, `publish_proposal`, `failure_ingest`.

## Invoke
- **MCP:** `load_tools(tools=["graph_orchestrate"])`, then `graph_orchestrate(action="execute_agent", agent_name="agent-utilities-expert", task="...")`.
- **REST twin:** `POST /graph/orchestrate` with `{"action": "swarm", "task": "..."}`.

## Example
```
graph_orchestrate(action="swarm", task="Audit the arr-stack for VPN leaks", max_fan_out=5)
```

## Delegation
If graph-os is reachable, offload composite multi-step work via `graph_orchestrate` (`execute_agent` / `execute_workflow`) instead of hand-running the steps — let the local LLM + Loop engine do it, and resolve only the exceptions.
