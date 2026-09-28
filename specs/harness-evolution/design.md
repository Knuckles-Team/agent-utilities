# Design: harness evolution and governed work

## Existing AU wiring and migration boundary

`agent_utilities/harness/substrate_trainer.py` already builds GRPO samples and emits an injected `TrainingJobSpec`; its default records locally and does not run gradients. `knowledge_graph/research/gaps.py` currently uses direct graph mutation/Cypher, Python priority sorting and best-effort handling. `knowledge_graph/research/loops.py` already has a statechart and native WorkItem claim calls. The evolution publication path has local Git helpers. These are migration surfaces, not acceptance of typed durability. Keep the Loop controller as procedural orchestration, then cut over each durable record and claim to the public EG generated client; delete old persistence/selection and direct-Git fallbacks in the same accepted tree.

## Authority and contracts

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

## Capture and training details

A capture binds terminal trajectory ID, policy-token count, action-mask digest, exact sampler policy version, checkpoint/adapter/tokenizer/decode digests, token-id and frozen chosen-token `log_q` blob references, optional Top-K/auxiliary samples, reward, verifier identity, provenance and content digest. Prompt, padding and tool-output tokens are masked from loss. A row is training-eligible only if terminal, all array dimensions align, sampler identity is resolvable, reward/verifier is approved and every blob holder is live. `log_q` is immutable; no recomputation with a newer model.

The vLLM-compatible adapter requests only supported observations and fails capability checks otherwise. `SubstrateTrainer` emits `TrainingJobSpec` referencing immutable capture IDs/digests and a **new** artifact destination. KLPO is optional and its estimator is chosen by measured cost; an expensive Monte Carlo default is not assumed. An external gradient service performs the work under a resource/A2A lease that preserves inference capacity. `TrainingRun` captures code/image digest, hyperparameters, resource use, terminal status and output digest. Held-out evaluation includes task quality, cost, latency, unsupported replay mass, trace completeness and safety; loss movement is diagnostic only. Promotion is separate, manually gated by default, compare-and-swap, canary-capable and reversible.

## Offer and materialization details

`WorkOffer/v1` is derived from the single Gap's WorkItem and records expected utility, closure probability, estimated token/GPU/CI/time cost, uncertainty, blast radius, reversibility, cooldown, dependencies, scope, evidence digests and units. EG's legal-option stage excludes open dependencies, active cooldown, missing capability/tenant/repository scope, policy violations and inadmissible budget. Initial `WorkOfferUtilityRateV1` is checked fixed-point arithmetic **inside EG Decide**: `floor(utility_micros * closure_ppm / max(cost_microunits, cost_floor))`, then tie-break by native priority, age and ID. AU does not sort offers. A committed decision does not grant authority; only a native fenced claim admits work.

AU creates a proposal with repository, exact base commit, path allowlist, patch digest, evidence, required gates, approval policy, risk and idempotency key. graph-os authenticates/authorizes it; repository-manager alone checks base, creates the worktree, applies the bounded patch, runs gates and commits explicitly. Conflict, dirty base, path escape, failed gate, missing approval and uncertain remote result leave the proposal pending. Gap resolution waits for a verified materialization receipt. Never write the same transition to old and new records.

## Security and observability

Tenant and auth scopes are checked at the EG boundary (`policy:capture-write`, `work:offer-write`, `evolution:proposal-write` and receipt-specific actions), then at graph-os dispatch. The run trace records sampler/offer/decision/claim/proposal/validation/materialization IDs with digests and high-watermarks. An uncertain trainer or Git dispatch reconciles by idempotency key and external receipt before any retry. No secrets, raw prompt corpora or checkpoint bytes enter a spec or unprotected graph field.
