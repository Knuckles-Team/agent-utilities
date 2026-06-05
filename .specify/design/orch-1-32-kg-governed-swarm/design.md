# Design — ORCH-1.32 KG-Governed Agent Swarm

**Concept:** ORCH-1.32 (extends ORCH-1.8 ParallelEngine, ORCH-1.1 Planner, ORCH-1.27 model roles).
**Source:** Kimi K2.6 Agent Swarm (Moonshot AI) + PARL/Mooncake papers. **Status:** Implemented (SWARM-1…7).

## Problem

The Kimi swarm decomposes a goal → parallel waves → synthesize, but as a black-box trained model.
agent-utilities already has the parallel DAG executor (`ParallelEngine`), decomposition, synthesis,
cost guard, and HITL — what it lacked was the **governance/quality loop**: a one-shot goal→swarm
entry point, per-leaf verification with re-dispatch, structured-output enforcement, retry/backoff,
heterogeneous model routing, critical-path metrics, and telemetry.

## Approach

Extend `ParallelEngine.execute` (the single path every caller uses) with additive features, and add
one `graph_orchestrate(action="swarm")` entry point that fuses decompose→execute→verify→synthesize.
All `AgentSpec`/`ExecutionResult` additions are default-valued (zero behavior change unless used).

## C4 (component)

- `graph_orchestrate(action="swarm")` (`mcp/kg_server.py`) — entry point.
- `Planner.decompose` (ORCH-1.1) → `ExecutionManifest.from_graph_plan` — goal → manifest.
- `ParallelEngine` (`graph/parallel_engine.py`) — waves (`_schedule_waves` + critical-path),
  per-agent exec (`_execute_agent`: model-role + schema), retry (`_run_one`), verify
  (`_verify_and_redispatch`), telemetry (`execute`).

## Data flow

`action="swarm"` → decompose(goal) → manifest(verify=True, max_retries=2, per-leaf success_criteria)
→ execute → [waves: retry/backoff per agent, schema-enforce output] → verify leaves vs criteria +
bounded re-dispatch → synthesize → ExecutionResult{deliverable, critical_path, parallelism,
verification, telemetry}.

## Honest boundary

No PARL training, no Mooncake serving (model/infra layers). Decomposition is Planner/LLM-driven, not
RL-optimized. "300 agents" is a tunable ceiling, not a guarantee; verification is the value.

## Wiring & metrics

See `docs/pillars/1_graph_orchestration/ORCH-1.32-KG_Governed_Agent_Swarm.md`. Tests:
`tests/unit/graph/test_orch_1_32_kg_governed_swarm.py` (7).
