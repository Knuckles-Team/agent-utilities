# Implementation plan: harness evolution

1. Pin and golden-test public EG generated contracts. Implement capability attestation and capture-only mode first; all three controls default off.
2. Add immutable terminal capture assembly in the provider/harness path. Commit through typed client; add length, mask, sampler and blob-holder negative fixtures.
3. Extend the injected `SubstrateTrainer` job schema to reference capture digests and optional KLPO. Dispatch to an external trainer through graph-os resource admission. Keep train and promote disabled while measuring capture overhead.
4. Add independent held-out `PolicyEvaluation` and compare-and-swap promotion with rollback. Use new artifacts only; publish measured resource and task outcomes before enabling any nondefault setting.
5. Replace direct Gap writes and Python sorting with atomic `GapUpsert`, `WorkOfferPut`, `Decide`/`DecisionCommit` and native WorkItem claim. Do not stage an AU fallback selector.
6. Replace AU direct Git publishing with `ChangeProposal` → graph-os → repository-manager → validation/materialization receipts. Delete obsolete authority classes and their consumers in the same cutover.
7. Add the `prompt_evolution` target to the existing optimization sweep (`harness/program_optimization.py`). It calls `harness/run_outcome_prompt_evolution.py`, which reuses the trace ontology cursor, `run_program_optimization`, `EvolveAgent._compiled_prompt_candidate` and `PromptVersionNode`. The existing `KG_OPTIMIZATION_ENABLED` daemon tick schedules it, so the change needs no new schedule, flag or engine task (AU-HARNESS-R011).
8. Run synthetic, fault and served integration fixtures. Record exact merged-head evidence per requirement and leave incomplete rows open.

Dependencies: real EG Decide and generated clients precede work-market acceptance; repository-manager and graph-os receipt contracts precede Git materialization acceptance. A locally emitted job is not trained, evaluated or promoted.

## L4 harness port (AU-HARNESS-R007 to AU-HARNESS-R010)

Architecture, in call order:

```text
L3 AgentSpec(harness=...) -> ParallelEngine._execute_agent
  native     -> existing in-process pydantic-ai path (unchanged)
  other name -> HarnessRegistry.select -> HarnessPort.run(HarnessRequest)
             -> RunOutcome -> record_outcome -> RunTrace writer + UsageRecorder
```

Interfaces live in `agent_utilities/layers/`:

| Module | Role |
|---|---|
| `harness_port.py` | `HarnessPort`, `HarnessRequest`, `RunOutcome`, `UsageReport`, `DiffStat`, `refused` |
| `harness_native.py` | `NativeHarness` over `run_agent`, the registry default |
| `harness_cli.py` | `ClaudeCodeHarness`, the pinned argument vector, the graph-os MCP configuration helper |
| `harness_process.py` | one bounded child launch, the environment allowlist and the read-only diff stat |
| `harness_registry.py` | per-node selection by `AgentSpec.harness`; unknown names raise |
| `harness_record.py` | L5 recording through the existing ordered RunTrace writer and `UsageRecorder` |
| `harness_node.py` | `AgentSpec` to `HarnessRequest`, and `RunOutcome` to `AgentExecutionResult` |

Reuse decisions:

- The native adapter calls `run_agent` unchanged. Tool calls, structured output, usage, trace export and the RunTrace write stay on that path.
- L5 reuses `_record_execution_trace_ordered` and `UsageRecorder.record_run`. No new trace store exists.
- The run identifier comes from `run_identity.new_run_id`.
- `core.execution.adapters` stays the text-only multi-CLI dispatcher. It returns no exit code, so the typed port does not route through it.
- The earlier unmerged layer stack (`HarnessPort`, `SandboxPort`, five adapters, negotiation) informed the environment allowlist and the Claude Code flags. Negotiation and `SandboxPort` stay under `AU-CONTROL-001`.

Extension point: a LangGraph or other runtime adds one class with `name` and `run`, then calls `HarnessRegistry.register`. No stub adapter ships.

Live wiring: the default registry holds only `native`. An operator registers `ClaudeCodeHarness` with a written graph-os MCP configuration. A node then sets `harness="claude-code"` and `harness_workspace` to a worktree. The MCP configuration references the token through an environment variable, never as a literal value.
