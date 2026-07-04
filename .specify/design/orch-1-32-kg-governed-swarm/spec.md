# Spec — ORCH-1.32 KG-Governed Agent Swarm

## Requirements

1. **SWARM-1** `graph_orchestrate(action="swarm")` takes a one-line goal, decomposes it, runs the
   ParallelEngine, verifies, synthesizes, and returns a JSON deliverable + telemetry + verification.
   Governance ON by default (`verify=True`, `max_retries=2`).
2. **SWARM-2** Leaves declaring `success_criteria` are judged; failures get one bounded re-dispatch
   with feedback before synthesis. Gated by `metadata["verify"]`; degrades to pass when no judge LLM.
3. **SWARM-3** `ExecutionResult.critical_path_length` = longest dependency chain (not wave count);
   `parallelism_ratio` = agents / critical path.
4. **SWARM-4** `AgentSpec.output_schema` → JSON contract enforced; violation = soft failure
   (retried/quarantined). No schema = pass (back-compat).
5. **SWARM-5** `AgentSpec.max_retries` (or `metadata["max_retries"]`) retries transient failures with
   exponential backoff; distinct from the circuit breaker.
6. **SWARM-6** `AgentSpec.model_role` routes to a model via ORCH-1.27 before default fallback; never
   raises (returns "" when unresolvable).
7. **SWARM-7** `ExecutionResult.telemetry` carries per-wave agents/duration/success + critical-path,
   parallelism, total agents, max concurrency.

## Acceptance

- 7 unit/integration tests in `test_orch_1_32_kg_governed_swarm.py` green.
- Additive only: existing ParallelEngine callers unchanged when new fields unused.
- `check_wiring`/`check_concepts`/no-stub/sprawl gates pass.

## Non-goals

PARL training; Mooncake serving; reproducing the trained orchestrator; native multi-format rendering
(SWARM-8, deferred to AU-ECO.connector.plane-provisioning-auth).
