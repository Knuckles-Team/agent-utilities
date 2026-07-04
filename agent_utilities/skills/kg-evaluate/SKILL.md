---
name: kg-evaluate
description: >-
  Evaluates agents/harnesses and reasons over learned world models — scoring, harness
  gates, world-model rollouts, forecasting, causal/invariant analysis, specialization
  cycles. Use for eval & simulation — "score these outputs", "run the harness gate",
  "forecast", "world-model rollout".
license: MIT
tags: [graph-os, evaluate]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-evaluate

`graph_evaluate` covers evaluation, gates and world-model reasoning. Actions: `evaluate`/`evaluate_alpha` (score outputs), `evaluate_harness`, `guard_corpus`, `harness_gate` (formal no-regression SHACL gate, AHE-3.53), `check_constraints`, `specialize` (SAI specialization cycle, AHE-3.29), `world_model_rollout` (KG-2.73b), `latent_efficiency_benchmark`, `evolve_model`, `forecast`, `causal`, `invariant`.

## Invoke
- **MCP:** `load_tools(tools=["graph_evaluate"])`, then `graph_evaluate(action="harness_gate", ...)`.
- **REST twin:** `POST /graph/evaluate` with `{"action": "forecast", ...}`.

## Example
```
graph_evaluate(action="world_model_rollout", query="ingestion lane under full-fleet reingest")
```

## Delegation
If graph-os is reachable, offload composite multi-step work via `graph_orchestrate` (`execute_agent` / `execute_workflow`) instead of hand-running the steps — let the local LLM + Loop engine do it, and resolve only the exceptions.
