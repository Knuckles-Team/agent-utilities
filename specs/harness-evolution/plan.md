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
