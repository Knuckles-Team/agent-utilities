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

## Run-loop mechanisms (AU-HARNESS-R012 to AU-HARNESS-R018)

Data flow, in call order:

```text
engine Decide(HarnessConfig) -> decide_harness_config (AU-CONTROL-R036) -> AgentTurnEnvelope.harness_config
  (agent_dispatch.py:471, signed by DispatchCarrier :180) -> enqueue_agent_turn (:737)
  -> _run_agent_turn_kind (agent_dispatch_worker.py:1764) -> verify (agent_dispatch.py:350)
  -> run_agent (agent_runner.py:702) -> HarnessRuntime
  -> capabilities (composition.py:32): ContextManager, plan injection, guards
  -> _stream_agent_run (agent_runner.py:471) -> agent.run / run_stream
  -> _record_execution_trace_ordered (agent_runner.py:5453) -> engine run trace
```

Types and fields: `HarnessConfig` and `TerminationReason` are new. The field names follow the cross-repository brief: `context_policy`, `soft_threshold`, `hard_threshold`, `recent_window_fraction`, `plan_scaffold`, `action_space`, the allowed tool subset, and the trace names `terminationReason`, `peakContextTokens`, `contextBudgetTokens`, `elisionCount`, `summarizationCount`, `turnCount`, `toolCallCount`, `costUsd`.

Attach points:

- **Configuration.** `run_agent` builds one `HarnessRuntime` after `validate_tool_contract` at `agent_runner.py:783`. The native adapter (`harness_native.py:116`) passes the request config. The claude-code adapter ignores context and plan fields and records that in its outcome.
- **Context manager.** It is a pydantic-ai capability on `before_model_request`, the hook `MementoCompaction` uses at `memento.py:243`. The default composition at `composition.py:102` swaps one capability for the other.
- **Plan scaffold.** The same hook appends the plan to the outgoing request. The `update_plan` tool registers with the other tools.
- **Action space.** Tool filtering happens where `run_agent` hands tools to the agent. The execution-time check reuses the carrier rule at `agent_dispatch.py:445`.
- **Guards.** They wrap tool execution through `after_tool_execute`, the hook `ToolOutputEviction` uses at `eviction.py:41`.
- **Trace.** `_record_execution_trace` (`agent_runner.py:5305`) gains the counters. No second store exists.

Failure and refusal behaviour:

- Absent `harness_config` gives a typed `refused` outcome with reason `harness_config_absent`.
- A provider length error or a history above budget after both mechanisms gives `context_overflow` with partial output.
- A tool outside the allowed set returns a `tool_not_allowed` observation and never runs.
- Identical failing calls past the second limit give `stuck`.
- A usage limit breach gives `step_budget`. Any other exception gives `error`.

Reused code:

- `enforce_tool_pair_safety` (`capabilities/compaction/_shared.py:61`) keeps tool calls and returns together.
- `ToolOutputEviction` (`capabilities/eviction.py:28`) stays for single oversized results.
- The L5 writer and `UsageRecorder` stay the only trace and cost path (`harness_record.py`).
- `readOnlyHint` parsing (`mcp/connector_surface.py:73`) supplies the parallel-execution marker.

New code: `harness_config.py`, `harness_runtime.py`, `harness_plan.py`, the `ContextManager` capability, the guards module and `harness/ablation.py`. No McNemar or paired-ablation code exists; `harness/hote_tri_evolution.py:224` ablates skill evolution and is not reused.

Decisions:

- **Replace, not add, compaction.** Rejected: keep `MementoCompaction` beside the new manager. Two sets of thresholds conflict and break the one-implementation rule.
- **No recall tool.** Rejected: a tool that restores elided content. The source study measured a median of zero calls and no accuracy gain. A later measured need may add one through a new row.
- **Plan scaffold is run state.** Rejected: reuse the `AU-CONTROL-R027` plan. That plan is engine-built and composes agents. The scaffold is a list the model edits during one run, and it composes nothing.
- **Config only from the envelope.** Rejected: local defaults when the field is absent. A hidden default is a second policy table.
- **Paper thresholds are defaults.** The values 0.6, 0.85 and 0.3 are initial defaults a measured value may replace. The code must not treat them as facts.

Cross-repository rows: the `EG` row for the `HarnessConfig` decision kind supplies the committed answer. The `EG` run-trace vocabulary row owns the trace field names. The `EG` typed-pack rows own `contextWindowTokens`, `shellProficiency`, `stoppingBehaviour` and `shellCentric`, and the `EG` run-trace rows own the measured-fact write that R018 feeds. The `GRAPHOS-A2A` admission row supplies the context budget. The `SDK-CONNECTOR-CONTROL` row for the `read_only` tool contract supplies the parallel marker. Client rows in webui, terminal and geniusbot read the trace.

## Context quality and large outputs (AU-HARNESS-R013.10 to R013.19)

Data flow: `run_agent` (`agent_runner.py:702`) -> `ContextManager.before_model_request` -> judge score -> regenerate or keep raw block -> model request. A tool result above `large_output_chars` -> run variable store -> stub in request -> read-slice tool.

Types: `SummaryNode` (text, source message ids, parent) and a run variable (name, size, content). Attach point: the manager hook replacing `capabilities/memento.py:243`, and the store beside `ToolOutputEviction` (`capabilities/eviction.py:28`). Failure: a twice-failed summary keeps the raw block; a bad slice returns `slice_out_of_range`.

Reused: the manager of `AU-HARNESS-R013` and the truncation guard of `AU-HARNESS-R016.6`. New: the judge call, the node graph and the variable store. Decision: pointers to originals instead of discarding them, because compaction must stay reversible. Rejected: lossy summaries without pointers.

## Parallel workspace (AU-HARNESS-R020)

Data flow: `_dispatch_parallel_batch` (`graph/_router_impl.py:1677`) -> `parallel_batch_processor` (`_router_impl.py:1772`) -> `ParallelWorkspace` -> worker slots -> synthesizer -> trace.

Types: `ParallelWorkspace`, `WorkerSlot`, a frozen result schema. Failure: schema violations are refused and count toward the breaker. Reused: the existing batch dispatch. New: the workspace and the strategy choice. Decision: synthesis by reference over typed objects. Rejected: concatenating worker text into a prompt, which grows the context with worker count.

## Evaluation gates (AU-HARNESS-R021)

Data flow: evaluation round -> `ContinuousEvaluationEngine.evaluate` (`harness/continuous_evaluation_engine.py:1103`) -> `eval_gates` -> `EvalGateReport` -> graph evidence.

Types: `EvalGateReport` with accuracy, safety error rate, judge score, independent outcome, divergence and verdict. Failure: `refuse` stops promotion. Reused: the evaluation engine and the ablation report. New: the safety measure, divergence check and bounds check. Decision: report safety apart from accuracy. Rejected: one blended score, which hides a safety regression behind an accuracy gain.

References:
- arXiv:2606.04923 — Wang et al., "Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based Reinforcement Learning" (2026)
- arXiv:2605.04039 — Wind et al., "Safety and accuracy follow different scaling laws in clinical large language models" (2026)
- arXiv:2605.04003 — Hoang et al., "Physics-Grounded Multi-Agent Architecture for Traceable, Risk-Aware Human-AI Decision Support in Manufacturing" (2026)

## Run experience (AU-HARNESS-R022)

Data flow: `run_agent` -> `record_outcome` (`harness_record.py`) -> distiller -> `ExperienceRecord` and `SkillCandidate` -> `SkillGraphPipeline` (`knowledge_graph/distillation/skill_graph_pipeline.py:836`) -> retrieval in the next run.

Types: `ExperienceRecord`, `SkillCandidate`. Failure: `insufficient_evidence` refusal. Reused: the skill graph pipeline. New: the distiller call in `record_outcome`, which creates that call site. Decision: knowledge accumulates in the graph with no parameter change. Rejected: fine-tuning per run, which costs more and cannot be reverted.

References:
- arXiv:2603.12056 — Jiang et al., "XSkill: Continual Learning from Experience and Skills in Multimodal Agents" (2026)

## Design

### Existing AU wiring and migration boundary

`agent_utilities/harness/substrate_trainer.py` already builds GRPO samples and emits an injected `TrainingJobSpec`; its default records locally and does not run gradients. `knowledge_graph/research/gaps.py` currently uses direct graph mutation/Cypher, Python priority sorting and best-effort handling. `knowledge_graph/research/loops.py` already has a statechart and native WorkItem claim calls. The evolution publication path has local Git helpers. These are migration surfaces, not acceptance of typed durability. Keep the Loop controller as procedural orchestration, then cut over each durable record and claim to the public EG generated client; delete old persistence/selection and direct-Git fallbacks in the same accepted tree.

### Authority and contracts

```text
evidence signal → EG GapUpsert (Gap + WorkItem atomic) → EG WorkOfferPut
 → EG Decide/DecisionCommit → native WorkItem claim → AU Loop executes
 → EG terminal outcome + independent evaluation

AU synthesis → EG ChangeProposalCommit → graph-os authorization/A2A
 → repository-manager worktree/patch/tests/commit/queue → EG MaterializationReceiptCommit
 → GapResolve after receipt verification

AU provider capture → EG trajectory + held Blob CAS → PolicyCaptureCommit
 → external training job → TrainingRunCommit → independent PolicyEvaluationCommit
 → graph-os release-pointer CAS if separately approved
```

AU must consume generated `GapClient`, `WorkMarketClient`, `DecideClient`, `PolicyEvolutionClient` and `EvolutionMaterializationClient`; if any is unavailable, return a typed refusal. Contract records: `OpenWeightPolicyCapability/v1`, `PolicyCapture/v1`, `ModelPolicyVersion/v1`, `TrainingRun/v1`, `PolicyEvaluation/v1`, `WorkOffer/v1`, `ChangeProposal/v1`, `ValidationReport/v1`, `MaterializationReceipt/v1`. EG owns their graph state; artifact bytes live in an external artifact store, repository bytes/history in Git. Credentials are opaque references resolved at the service boundary. AU never invents a parallel dataclass/graph row that crosses a durable boundary.

### Capture and training details

A capture binds terminal trajectory ID, policy-token count, action-mask digest, exact sampler policy version, checkpoint/adapter/tokenizer/decode digests, token-id and frozen chosen-token `log_q` blob references, optional Top-K/auxiliary samples, reward, verifier identity, provenance and content digest. Prompt, padding and tool-output tokens are masked from loss. A row is training-eligible only if terminal, all array dimensions align, sampler identity is resolvable, reward/verifier is approved and every blob holder is live. `log_q` is immutable; no recomputation with a newer model.

The vLLM-compatible adapter requests only supported observations and fails capability checks otherwise. `SubstrateTrainer` emits `TrainingJobSpec` referencing immutable capture IDs/digests and a **new** artifact destination. KLPO is optional and its estimator is chosen by measured cost; an expensive Monte Carlo default is not assumed. An external gradient service performs the work under a resource/A2A lease that preserves inference capacity. `TrainingRun` captures code/image digest, hyperparameters, resource use, terminal status and output digest. Held-out evaluation includes task quality, cost, latency, unsupported replay mass, trace completeness and safety; loss movement is diagnostic only. Promotion is separate, manually gated by default, compare-and-swap, canary-capable and reversible.

### Offer and materialization details

`WorkOffer/v1` is derived from the single Gap's WorkItem and records expected utility, closure probability, estimated token/GPU/CI/time cost, uncertainty, blast radius, reversibility, cooldown, dependencies, scope, evidence digests and units. EG's legal-option stage excludes open dependencies, active cooldown, missing capability/tenant/repository scope, policy violations and inadmissible budget. Initial `WorkOfferUtilityRateV1` is checked fixed-point arithmetic **inside EG Decide**: `floor(utility_micros * closure_ppm / max(cost_microunits, cost_floor))`, then tie-break by native priority, age and ID. AU does not sort offers. A committed decision does not grant authority; only a native fenced claim admits work.

AU creates a proposal with repository, exact base commit, path allowlist, patch digest, evidence, required gates, approval policy, risk and idempotency key. graph-os authenticates/authorizes it; repository-manager alone checks base, creates the worktree, applies the bounded patch, runs gates and commits explicitly. Conflict, dirty base, path escape, failed gate, missing approval and uncertain remote result leave the proposal pending. Gap resolution waits for a verified materialization receipt. Never write the same transition to old and new records.

### Security and observability

Tenant and auth scopes are checked at the EG boundary (`policy:capture-write`, `work:offer-write`, `evolution:proposal-write` and receipt-specific actions), then at graph-os dispatch. The run trace records sampler/offer/decision/claim/proposal/validation/materialization IDs with digests and high-watermarks. An uncertain trainer or Git dispatch reconciles by idempotency key and external receipt before any retry. No secrets, raw prompt corpora or checkpoint bytes enter a spec or unprotected graph field.

## Study pipeline (AU-HARNESS-R019)

Data flow: source locator -> `study_intake.fetch_once` -> `study_compare.compare` -> `study_propose.propose` -> `study_admit.admit` -> engine verdict.

Attach point: the first child row locates and wires it beside `agent_utilities/harness/assimilation_benchmark.py` and the evolution sweep.

Reused: the assimilation benchmark suite, `citation_tracker.py`, the compute governor and the propose-only pattern of `AU-HARNESS-R011`.

Decision: the engine decides capability matches and admission, and AU only runs the steps. Rejected: an AU-side capability inventory, which adds a second store and a second decider.

## Verified run programs (AU-HARNESS-R023)

Data flow: `run_agent` (`agent_runner.py:702`) -> program lookup by task signature -> `RunProgram` replay -> deviation -> agent path -> `record_outcome` (`harness_record.py`) -> compiler -> independent replay -> engine store.

Types: `RunProgram`, `ProgramStep`, `ProgramStatus`. Attach point: `run_agent`, called by the dispatch worker and the graph nodes. Failure: a deviation returns control to the agent. Reused: `ReplayManifest` and `InteractionRecord` (`observability/replay_engine.py:20`, `:36`). New: preconditions, compiler, verifier.
Existing code: `agent_utilities/observability/replay_engine.py:48`, `agent_utilities/orchestration/agent_runner.py:702`. Reachable today: no.
Search terms used: replay, compiled, macro, record_trajectory, workflow_cache.
Decision: the engine stores programs and the verifier is independent of the compiler. Rejected: self-verification by the compiler, which admits its own mistakes.

References:
- arXiv:2606.17929 — Li, "PreAct: Computer-Using Agents that Get Faster on Repeated Tasks" (2026)

## End-to-end tracing (AU-HARNESS-R024)

Data flow: served graph run -> `_dispatch_parallel_batch` (`graph/_router_impl.py:1677`) -> step span -> tool span -> `default_runtime_capabilities` instrumentation (`capabilities/composition.py:160`) -> exporter -> run trace `traceId`.

Types: `RunSpanContract` with closed span kinds. Attach point: graph steps and the tool path. Failure: no collector gives one typed observation, and the run continues. Reused: `build_fleet_instrumentation` (`capabilities/telemetry_instrumentation.py:40`), `setup_otel` (`observability/custom_observability.py:850`). New: span kinds, step and tool spans, replay comparison.
Existing code: `agent_utilities/capabilities/telemetry_instrumentation.py:40`, `agent_utilities/capabilities/composition.py:160`, `agent_utilities/observability/custom_observability.py:850`, `agent_utilities/observability/replay_engine.py:81`. Reachable today: yes for agent-call spans, no for graph-step spans and replay.
Decision: one connected trace per run with a fixed contract. Rejected: separate agent and graph traces, which cannot show a regression across layers.

## Revertible harness edits (AU-HARNESS-R025)

Data flow: evolution round -> `AegisLoop` (`harness/aegis_loop.py:91`) -> `HarnessEditRecord` opened -> `apply_edits` (`harness/edit_engine.py:457`) -> next round outcomes -> verdict -> revert of recorded files -> graph record.

Types: `HarnessEditRecord`, `DeclaredPrediction`. Attach point: `AegisLoop`, reached from the evolution sweep. Failure: a missing prediction refuses the edit. Reused: `HarnessGate` (`harness/harness_gate.py:114`). New: prediction, verdict, file-level revert.
Existing code: `agent_utilities/harness/harness_gate.py:127`, `agent_utilities/harness/aegis_loop.py:274`, `agent_utilities/harness/edit_engine.py:457`, `agent_utilities/core/registry/kg_adapter.py:660`. Reachable today: yes for the gate, no for predictions.
Decision: revert on unverified as well as falsified. Rejected: reverting on falsified only, which keeps edits that carry no evidence.

References:
- arXiv:2604.25850 — Lin et al., "Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses" (2026)

## Bounded map and decomposition (AU-HARNESS-R026)

Data flow: planning step -> `_router_impl.py:811` -> `run_full_rlm` (`rlm/repl.py:972`) -> `MapPrimitive` -> `run_parallel_sub_calls` (`rlm/repl.py:470`) -> result slots -> combined answer -> run trace.

Types: `MapPrimitive`, `DecompositionBudget`. Attach point: the sandbox namespace in `run_full_rlm`. Failure: a typed `budget_exhausted` result. Reused: `map_concurrent` (`core/model_concurrency.py:296`), `RLMConfig.max_depth` (`rlm/config.py:60`). New: budget, primitives, loop refusal.
Existing code: `agent_utilities/rlm/repl.py:470`, `agent_utilities/rlm/repl.py:972`, `agent_utilities/core/model_concurrency.py:296`, `agent_utilities/rlm/config.py:60`. Reachable today: `run_full_rlm` yes, `run_parallel_sub_calls` no production caller.
This rollup merges two requirements that state one capability: bounded work over inputs larger than the window.
Decision: the engine supplies the loop. Rejected: model-written loops, which cannot guarantee termination.

References:
- arXiv:submit/7269166 — Ehrlich et al., "LCM: Lossless Context Management" (2026)
- arXiv:2512.24601 — Zhang et al., "Recursive Language Models" (2026)
