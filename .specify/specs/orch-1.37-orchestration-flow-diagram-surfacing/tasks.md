# Tasks: Orchestration Flow Diagram Surfacing (ORCH-1.37)

> Spec: `.specify/specs/orch-1.37-orchestration-flow-diagram-surfacing/spec.md`
> Design: `.specify/design/orch-1.37-orchestration-flow-diagram-surfacing/design.md`

## T1 — swarm: include `mermaid` (AC1, AC2)
- `agent_utilities/mcp/kg_server.py:2784-2797` — add `"mermaid": pe_result.mermaid,` to the
  returned `json.dumps({...})` dict. `pe_result.mermaid` is `str | None` (null-safe).
- Tag the change `CONCEPT:AU-ORCH.execution.orchestration-flow-mermaid`.

## T2 — execute_agent: preserve + surface `mermaid` (AC3–AC6)
- `agent_utilities/orchestration/agent_runner.py:596-605` — pass `streamdown=True` into the
  `AgentOrchestrationEngine().execute_graph(...)` call so `GraphResponse.mermaid` populates.
- `agent_utilities/orchestration/agent_runner.py:224-234` — add `return_mermaid: bool = False`
  param to `run_agent`; when set and `result.get("mermaid")` is truthy, return
  `json.dumps({"output": <output>, "mermaid": <mermaid>})`; else return the bare output string.
- `agent_utilities/orchestration/manager.py:71-80` — thread `return_mermaid` through
  `Orchestrator.execute_agent` to `run_agent`.
- `agent_utilities/mcp/kg_server.py:2806-2815` — call with `return_mermaid=True`; return result as-is.

## T3 — compile_workflow: include `mermaid` (AC7)
- `agent_utilities/mcp/kg_server.py:2816-2828` — add
  `"mermaid": WorkflowStore(engine).get_mermaid(name)` to the returned dict (mirror the local
  `WorkflowStore` import used in the `list_workflows` branch at `kg_server.py:2831`).

## T4 — execute_workflow: include `mermaid` (AC8)
- `agent_utilities/mcp/kg_server.py:2844-2855` — return
  `json.dumps({"result": wf_result, "mermaid": WorkflowStore(engine).get_mermaid(name)}, default=str)`.

## T5 — Docstring (US-2 note)
- `agent_utilities/mcp/kg_server.py:~2715` — note `execute_agent` returns a JSON wrapper with
  `mermaid` when a diagram is available; other actions gain an additive `mermaid` key.

## T6 — Unit test (NFR)
- `tests/unit/test_orchestrate_mermaid_surfacing.py` (or nearest existing orchestration test
  module) — `@pytest.mark.concept(id="ORCH-1.37")`; assert each handler includes/forwards
  `mermaid` (mock engine/orchestrator/`WorkflowStore.get_mermaid`); assert `run_agent` default
  returns bare string and `return_mermaid=True` returns the wrapper. ≤60s, no network.

## T7 — E2E harness (US-4, AC9–AC12)
- `scratch/test_workflow_e2e.py` (in the worktree; `scratch/` is gitignored — all testing
  isolated in the worktree) — modeled on `test_delegation.py` using
  `_build_server` + `_execute_tool` + `_get_engine` (multiplexer-independent). Phases: bootstrap,
  vLLM preflight, ingest MCP tools (small→full), ingest universal-skills, verify topology,
  execute_agent (github-mcp, assert tool call), compile+execute_workflow (full_ecosystem_health),
  optional swarm, write Markdown report with fenced ```mermaid blocks.

## T8 — Post-modification artifacts (NFR)
- `docs/` ORCH pillar page; `AGENTS.md` `graph_orchestrate` response-shape note; `CHANGELOG.md`
  Unreleased entry; `README.md` if applicable; regenerate `docs/concepts.yaml` for ORCH-1.37;
  KG dual-write of this spec via `kg_ingest`.

## Validation Gate
- `python -m pytest tests/unit/test_orchestrate_mermaid_surfacing.py -q` green (system python per
  egeria/py312 pre-commit note).
- Import-check `kg_server`, `agent_runner`, `manager`.
- Run `python /home/apps/workspace/test_workflow_e2e.py`; confirm non-null `mermaid` in
  execute_agent / execute_workflow / swarm responses and a readable report.
- `pre-commit run --all-files` green.
