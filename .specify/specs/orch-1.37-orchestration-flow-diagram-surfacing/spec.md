# Spec: Orchestration Flow Diagram Surfacing (ORCH-1.37)

> References design: `.specify/design/orch-1.37-orchestration-flow-diagram-surfacing/design.md`

## Pre-Flight Checklist

- [x] Design document exists and KG-nearest-concepts table completed.
- [x] Extension target identified (ORCH-1.8 Parallel Engine Visualizer, similarity ≥ 0.70).
- [x] New CONCEPT:ORCH-1.37 justified in design as augmentation (sub-concept of ORCH-1.8).
- [x] Wire-First confirmed: ≤2 hops from `graph_orchestrate`.
- [x] No diagram generation added — surfaces an existing artifact only.

## Context

End-to-end testing of skill-workflow execution via the graph-os MCP (`graph_orchestrate`,
spawning a sub-agent that leverages ingested MCP tools) requires reviewing the Mermaid diagram
of the execution flow. The diagram is already generated (ORCH-1.8 `WorkflowVisualizer`) but no
`graph_orchestrate` action returns it, so it cannot be reviewed by an MCP client. This feature
closes that gap additively and is validated by a full ingest → execute → review test harness.

## User Stories

### US-1 — Get the flow diagram from a swarm run
**As** an MCP client calling `graph_orchestrate(action="swarm")`,
**I want** the response JSON to include the execution-flow Mermaid diagram,
**so that** I can render the wave/agent/dependency topology of the run.
- **AC1**: The `swarm` response JSON includes a `"mermaid"` key sourced from
  `ExecutionResult.mermaid` (null-safe — `null` when generation was skipped/failed).
- **AC2**: All pre-existing response keys (`deliverable`, `agent_count`, `wave_count`,
  `critical_path_length`, `parallelism_ratio`, `verification`, `telemetry`, `execution_id`,
  `success`) are unchanged.

### US-2 — Get the flow diagram from an executed agent
**As** an MCP client calling `graph_orchestrate(action="execute_agent")`,
**I want** the response to carry the graph Mermaid diagram when one was produced,
**so that** I can see the routed graph the sub-agent executed (including its MCP tool calls).
- **AC3**: `_execute_graph` passes `streamdown=True` so `GraphResponse.mermaid` is populated.
- **AC4**: `run_agent(return_mermaid=True)` returns a JSON string `{"output": ..., "mermaid": ...}`
  when a diagram is present, and the bare `output` string otherwise.
- **AC5**: Internal callers of `run_agent` (default `return_mermaid=False`) still receive a bare
  string — the dynamic-workflow fan-out (`engine.py:1316`, `isinstance(r, str)`) is unaffected.
- **AC6**: The `execute_agent` MCP handler opts in (`return_mermaid=True`) and returns the wrapper.

### US-3 — Get the flow diagram from workflow compile/execute
**As** an MCP client calling `compile_workflow` or `execute_workflow`,
**I want** the response to include the workflow's stored Mermaid diagram,
**so that** I can review the compiled topology and its execution.
- **AC7**: `compile_workflow` response includes `"mermaid"` from `WorkflowStore.get_mermaid(name)`.
- **AC8**: `execute_workflow` response is a JSON object `{"result": ..., "mermaid": ...}` with the
  diagram read via `WorkflowStore.get_mermaid(name)` (null-safe).

### US-4 — End-to-end validation harness
**As** a developer validating the graph-os MCP path,
**I want** a harness that ingests all MCP tools + universal-skills, executes an agent and a
workflow, and writes the captured diagrams to a report,
**so that** the full ingest → spawn → execute → review loop is reproducible.
- **AC9**: `test_workflow_e2e.py` ingests MCP tools (small config first, then full
  `mcp_config.json` tolerating per-server discovery errors) and universal-skills + workflows,
  then verifies non-zero `Server`/`CallableResource` counts via `graph_query`.
- **AC10**: It runs `execute_agent` (read-only `github-mcp` task) and asserts an MCP tool was
  actually invoked (result data and/or provenance node from `_record_execution_trace`).
- **AC11**: It `compile_workflow`s then `execute_workflow`s a read-only canary workflow
  (`full_ecosystem_health`), handling the CallableResource↔WorkflowDefinition gap.
- **AC12**: It captures the `mermaid` from each response and writes a Markdown report with fenced
  ```mermaid blocks and per-phase pass/fail.

## Non-Functional Requirements

- [ ] Unit test in `tests/` tagged `@pytest.mark.concept(id="ORCH-1.37")`, ≤60s, no network
      (assert the four handlers include/forward `mermaid`; mock the engine where needed).
- [ ] All existing tests continue to pass (zero regression).
- [ ] `pre-commit run --all-files` green (ruff/mypy/bandit + guardrail gates).
- [ ] Post-modification artifacts updated: `docs/` pillar page (ORCH-1.8/1.37), `AGENTS.md`
      (`graph_orchestrate` response shape), `CHANGELOG.md`, `README.md` if applicable,
      `docs/concepts.yaml` regen for ORCH-1.37, this `.specify/` set, KG dual-write via `kg_ingest`.
