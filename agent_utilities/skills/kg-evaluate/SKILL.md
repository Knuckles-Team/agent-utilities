---
name: kg-evaluate
skill_type: skill
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

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `graph_evaluate` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_evaluate"])` once per session (as below), then proceed exactly as documented; or (2) call the `why` intent verb with the same natural-language request — the resolver routes to `graph_evaluate` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tools eagerly instead.


`graph_evaluate` covers evaluation, gates and world-model reasoning. Actions: `evaluate`/`evaluate_alpha` (score outputs), `evaluate_harness`, `guard_corpus`, `harness_gate` (formal no-regression SHACL gate, AHE-3.53), `check_constraints`, `specialize` (SAI specialization cycle, AU-AHE.harness.sai-controller), `world_model_rollout` (KG-2.73b), `latent_efficiency_benchmark`, `evolve_model`, `forecast`, `causal`, `invariant`.

## Invoke
- **MCP:** `load_tools(tools=["graph_evaluate"])`, then `graph_evaluate(action="harness_gate", ...)`.
- **REST twin:** `POST /graph/evaluate` with `{"action": "forecast", ...}`.

## Example
```
graph_evaluate(action="world_model_rollout", query="ingestion lane under full-fleet reingest")
```

## Delegation
If graph-os is reachable, offload composite multi-step work via `graph_orchestrate` (`execute_agent` / `execute_workflow`) instead of hand-running the steps — let the local LLM + Loop engine do it, and resolve only the exceptions.
