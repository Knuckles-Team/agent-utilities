# Tasks — ORCH-1.32 KG-Governed Agent Swarm

- [x] SWARM-1 `graph_orchestrate(action="swarm")` (decompose→execute→verify→synthesize); docstring updated.
- [x] SWARM-2 `_verify_and_redispatch` + `_judge_against_criteria`; gated by `metadata["verify"]`.
- [x] SWARM-3 critical-path + parallelism in `_schedule_waves`; surfaced on `ExecutionResult`.
- [x] SWARM-4 `enforce_structured_output` + `AgentSpec.output_schema`; soft-fail on violation.
- [x] SWARM-5 retry-with-backoff in `_run_one`; `AgentSpec.max_retries` + `metadata["max_retries"]`.
- [x] SWARM-6 `resolve_model_role` + `AgentSpec.model_role` in `_execute_agent`.
- [x] SWARM-7 per-wave telemetry on `ExecutionResult.telemetry`.
- [x] Tests `test_orch_1_32_kg_governed_swarm.py` (7); engine integration with mocked LLM.
- [x] Docs: deep-dive, `concept_map` row, `concepts.yaml` regen (86), CHANGELOG, AGENTS regen.
- [ ] Live-LLM validation: real goal→swarm run; verify acceptance-rate + critical-path on a true
  multi-agent task (deferred to the live-testing pass).
- [ ] SWARM-8 native multi-format deliverable rendering (deferred → ECO-4 peripheral).
