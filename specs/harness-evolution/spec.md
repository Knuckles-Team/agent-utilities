# Harness evolution and governed work

**Owner:** agent-utilities (AU) · **Stable ID:** `AU-HARNESS-001`
**Delivery:** PARTIAL · **Acceptance:** NOT VERIFIED · **Scope IDs:** AU-HARNESS-R001–AU-HARNESS-R011 (AU portions). See [requirements.md](requirements.md) for the definition of every requirement ID and [status.json](status.json) for its delivery state and evidence.

## Outcome

An operator sees one evidence-backed queue of gaps, can capture eligible open-weight policy trajectories, order safe work using the canonical decision and WorkItem contracts, and follow an approved change from proposal to Git receipt. AU contributes capture, synthesis and execution; it does not own model weights, a competing work scheduler, graph durability or repository mutation.

## Stories and acceptance

1. **P0: captured evidence.** A supported provider exposes attested chosen-token log probabilities, immutable behavior-policy version and tokenizer/decode identity. With `capture=true`, AU records terminal reward, masks and held data references. Unsupported/closed/unversioned providers refuse capture; train and promote remain independently off.
2. **P0: one work market.** A signal upserts one canonical Gap and one native WorkItem atomically. AU derives a versioned WorkOffer from evidence, asks EG `Decide` to select legal work, commits the record and executes only after the native fenced claim. Missing Decide never triggers a local sort or fallback queue.
3. **P0: governed code correction.** AU produces a bounded `ChangeProposal` tied to a Gap and a complete local spec. graph-os checks policy and sends the approved task to repository-manager; its Git/validation receipt is re-ingested before resolution. AU never runs Git as a materializer.
4. **P0: one harness port.** An L3 node names its runtime in one spec field. The default is the native pydantic-ai path. A node that names `claude-code` runs headless Claude Code in a provided worktree. Its MCP tool calls go through graph-os only. Every run returns one typed `RunOutcome`. The existing RunTrace and usage writers record it.
5. **P1: policy experiment.** AU emits an external LoRA/new-artifact training job with immutable capture digests, receives a terminal run receipt and requests independent held-out evaluation. A separate approved compare-and-swap promotion may move the serving pointer; failure or cancellation cannot.
6. **P1: prompt evolution from run outcomes.** An agent accumulates failed and successful runs. The scheduled optimization sweep reads the new outcomes for that agent and runs the program optimizer. A reviewer finds one `PromptVersion` candidate with status `proposal` and links to the source traces. The live prompt file stays unchanged. A second candidate waits until the first one leaves review.

## Requirements

| ID | Requirement | Scope IDs | Acceptance proof |
|---|---|---|---|
| HE-01 | Attest provider capability; bind tokenizer, decode, sampler and base/adapter digests. Three independent `capture/train/promote` controls default false. | AU-HARNESS-R001 | capability and default-off negatives |
| HE-02 | Capture terminal trajectory, frozen `log_q`, token ids, policy-token mask, verifier/reward and live blob references. Reject incomplete or mismatched arrays as training input. | AU-HARNESS-R001 | replay/golden/fault tests |
| HE-03 | Extend `SubstrateTrainer` job emission to optional KLPO beside GRPO/DPO/SFT; gradients and checkpoint bytes remain external; first output is a new adapter artifact. | AU-HARNESS-R002 | injected dispatcher and receipt tests |
| HE-04 | Use EG Gap/WorkOffer/Decide/DecisionCommit/WorkItem contracts for selection, claim, budget, cooldown and terminal outcome. | AU-HARNESS-R003 | served one-Gap/one-claim fixture |
| HE-05 | Route code proposals through graph-os authorization and repository-manager materialization; remove AU direct Git, local publication/report authority and dual writes. | AU-HARNESS-R004 | source gate and end-to-end receipt |
| HE-06 | Train/promote only with independent held-out evaluation, bounded resource lease, immutable artifact, compare-and-swap pointer and rollback receipt. | AU-HARNESS-R002 | negative and canary tests |
| HE-09 | Define one typed L4 `HarnessPort`: a `name` and `run(HarnessRequest) -> RunOutcome`. Frozen, strict models. A new runtime is one registered class. | AU-HARNESS-R007 | protocol conformance kit |
| HE-10 | Serve the `native` harness through `run_agent` unchanged. Keep typed tool calls, structured output, token accounting, trace export and the RunTrace write. | AU-HARNESS-R008 | envelope, timeout and no-double-write tests |
| HE-11 | Serve `claude-code` through `claude -p --output-format json`. Pin flags, require a worktree, bound time, allowlist the child environment, and return a typed outcome. | AU-HARNESS-R009 | fake-executable tests; no real CLI call |
| HE-12 | Select the harness per L3 node through `AgentSpec.harness`, default `native`. Record each non-native outcome through the existing L5 writers. | AU-HARNESS-R010 | selection, parallel-engine dispatch and L5 tests |
| HE-07 | Keep in-engine generative/autograd implementation deferred until a separate evidence-backed Gap proves held-out benefit, cost and safety. | AU-HARNESS-R005 | absence and decision record |
| HE-16 | Read new attributed `RunTrace` outcomes past a durable cursor. Optimize through a pluggable optimizer (native `eg-program` by default). Record a `PromptVersion` proposal with `was_derived_from` trace edges. Never write or promote the prompt. | AU-HARNESS-R011 | graph-double, fake-optimizer and native-path tests |

## Success criteria

The same event cannot create two schedulable WorkItems; a repeated offer cannot bypass cooldown; dispatch without approval/client/service availability has zero Git or graph side effects; missing log probabilities never become invented values. Exact merged-head CI, receipt and manual-promotion evidence are needed before marking any requirement accepted. The [agent control plane](../agent-control-plane/spec.md) supplies RunSpec, conformance and L5 trace guarantees consumed here.

Requirement IDs are defined in [requirements.md](requirements.md); delivery state per ID is in `status.json`.
