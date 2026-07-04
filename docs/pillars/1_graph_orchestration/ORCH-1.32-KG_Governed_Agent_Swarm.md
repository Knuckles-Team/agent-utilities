# ORCH-1.32 — KG-Governed Agent Swarm

> Assimilated from **Kimi K2.6 Agent Swarm** (Moonshot AI) — the three swarm guides + PARL
> (arXiv 2602.02276) and Mooncake (arXiv 2407.00079). Extends **ORCH-1.8** (Parallel Engine),
> **ORCH-1.1** (Planner), **ORCH-1.27** (model-role routing).

## Premise

Kimi's swarm decomposes one goal into a dependency graph, runs up to 300 sub-agents in
dependency-ordered parallel waves, and synthesizes one deliverable — but it is a **black-box
trained behavior** (PARL-RL'd orchestrator, Mooncake KV-disaggregated serving). agent-utilities
already had the *engine* for this (`ParallelEngine` — `topological_generations` waves +
`asyncio.gather` + `CognitiveScheduler` + RLM synthesis + `UsageGuard` cost caps + HITL approval).
ORCH-1.32 adds the **governance and quality deltas** Kimi's opaque model gets for free, so the swarm
is **transparent, KG-grounded, and verifiable** — the way it surpasses a trained black box.

## The seven deltas (SWARM-1…7)

| | Delta | Where |
|---|---|---|
| **SWARM-1** | **One-shot `graph_orchestrate(action="swarm")`** — a one-line goal → `Planner.decompose` → `ExecutionManifest.from_graph_plan` → `ParallelEngine.execute` → verify → synthesize → single deliverable. Governance ON by default (`verify=True`, `max_retries=2`). | `mcp/kg_server.py` |
| **SWARM-2** | **Planner→execute→verify loop** — each leaf with `success_criteria` judged against it; failures get one bounded re-dispatch with the judge's feedback before assembly. "The loop only closes when something with real judgment signs off." | `parallel_engine._verify_and_redispatch` |
| **SWARM-3** | **Critical-path metric** — report the longest dependency chain (true wall-clock floor) + parallelism ratio, not raw wave count (the PARL insight: optimize critical steps, not total). | `parallel_engine._schedule_waves` |
| **SWARM-4** | **Per-agent structured-output contract** — `AgentSpec.output_schema` forces JSON; a violation is a soft failure (retried/quarantined) so prose never poisons synthesis (Kimi guardrail #3). | `parallel_engine.enforce_structured_output` |
| **SWARM-5** | **Retry-with-backoff** — per-agent (or manifest) `max_retries` with exponential backoff; recovers transient failures within a wave (distinct from the circuit breaker that disables chronic failures across waves). | `parallel_engine._run_one` |
| **SWARM-6** | **Heterogeneous-model swarm (Claw Groups)** — `AgentSpec.model_role` routes each agent to a model tier via ORCH-1.27 (reasoning vs bulk vs local) before the default fallback. | `parallel_engine.resolve_model_role` |
| **SWARM-7** | **Scale telemetry** — per-wave cost/latency/success + critical-path/parallelism surfaced on `ExecutionResult.telemetry`; concurrency ceiling tunable toward the 300/4,000 envelope under `UsageGuard`. | `parallel_engine.execute` |

SWARM-8 (native PDF/PPT/Excel/web rendering) is deferred to an AU-ECO.connector.plane-provisioning-auth ecosystem peripheral.

## What we deliberately did NOT adopt

- **PARL training / Mooncake serving** — model + infra layers; we orchestrate hosted pydantic-ai
  agents, not train K2.6 or run vLLM. Our decomposition is Planner/LLM-driven, not RL-optimized — a
  transparent + governed trade vs. trained + opaque.
- "300 agents" is a **ceiling, not a guarantee** — SWARM-2's verify loop is the value, not the count.

## Wiring (Wire-First)

`graph_orchestrate(action="swarm")` → `Planner.decompose` → `ExecutionManifest.from_graph_plan` →
`ParallelEngine.execute` (waves → verify → synthesize). All engine deltas live on the single
`execute()` path every caller (`workflows/runner`, `orchestration/engine`, `agent_runner`) already
uses, so they apply everywhere, not just the new action. Verified by
`tests/unit/graph/test_orch_1_32_kg_governed_swarm.py` (7 tests; engine integration with mocked LLM).

## Success metrics

- One prompt → synthesized, verified multi-part deliverable; KG-persisted run.
- `critical_path_length` reflects the longest chain (wall-clock floor); `parallelism_ratio > 1` when
  work is parallelizable.
- 100% of intermediate outputs schema-valid or quarantined; transient agent failures auto-recover.
- % leaves passing `success_criteria` before assembly surfaced in `result.verification`.
