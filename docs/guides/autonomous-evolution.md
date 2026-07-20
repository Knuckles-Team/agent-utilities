# Enabling Autonomous Evolution

The platform's self-evolution arcs are fully wired but **off by default**: the
daemon ticks for the golden loop (`KG-2.7`) and failure-driven evolution
(`AU-AHE.harness.failure-evolution`) are registered in the engine's maintenance scheduler, yet their
flags default to `False` in code. That is deliberate — turning a fleet
autonomous is an explicit XDG AgentConfig deployment decision, never a library
default or repository-local setting.

This guide describes the safety chain you get when you turn the loops on, and
exactly which flags do what.

## The safety chain

Every autonomous change passes a chain of independent stages, each of which
can stop it:

```mermaid
flowchart LR
    A["Propose-only loops<br/>(golden loop, failure ingest,<br/>anomaly consumer, fleet events)"] --> B["Governed validation<br/>PromotionGovernanceValidator<br/>(AU-AHE.harness.promotion-governance-validator)"]
    B --> C["Regression gate<br/>recorded RegressionGateResult<br/>(AU-AHE.harness.failure-evolution)"]
    C --> D["Merge<br/>human by default;<br/>auto only with KG_GOLDEN_AUTO_MERGE"]
    D --> E["Promotion policy gate<br/>ActionPolicy merge_promotion<br/>(OS-5.24) — deny blocks the flip"]
    E --> F["Publication<br/>same merge_promotion approval<br/>(AHE-3.21, approval by default)"]
    F --> G["Reviewable branch<br/>change synthesis + RLM sandbox<br/>(AHE-3.21) — never pushed"]
    G --> H["Human merges<br/>normal release flow"]
```

1. **Propose-only loops.** The golden loop, the failure-evolution sweep, the
   PerformanceAnomaly consumer (`AU-AHE.optimization.performance-anomaly-consumer`) and fleet-event triage
   (`AU-OS.config.fleet-event-ingress`) only ever *write proposals*: `failure_gap` Concept topics, spec
   drafts under `.specify/`, and `TeamSpec`/`AgentSpec` proposal nodes. No
   code executes, nothing is promoted.
2. **Governed validation.** `GovernedAutoMerger` now constructs the
   *production* `PromotionGovernanceValidator` by default
   (`knowledge_graph/research/promotion_governance.py`). A promotion candidate
   must clear all four rules: MergePolicy quality thresholds, the bundled
   SHACL governance shapes (`shapes/governance.shapes.ttl`), the recorded
   regression-gate verdict, and active constitution `forbid` rules in the KG.
3. **Regression gate.** Failure remediations carry a live regression check
   bound to the failures they address; every verdict is also persisted as a
   `RegressionGateResult` node, and a recorded `hold` blocks promotion until a
   later gate run records a `pass`.
4. **Human merge.** With `KG_GOLDEN_AUTO_MERGE` unset (the default), even a
   proposal that passes every gate stays proposal-only; promotion is a human
   act. Flipping it on delegates only the *final* step — to the governed,
   audited path above, with every decision logged through the
   `golden_loop.auto_merge` audit trail.
5. **Operational promotion gate (`OS-5.24` adoption).** Even with auto-merge
   on, the merger's own promotion decision consults the operational
   `ActionPolicy` under the reserved `merge_promotion` kind *before* the
   lifecycle flip (`research/auto_merge.py`). A `deny` — a `forbidden` tier,
   a rate-limit breach, or a policy-engine failure (the gate fails closed) —
   blocks the promotion and is recorded on the evaluation and the audit
   trail. The shipped `approval_required` tier queues the **same**
   `ActionApproval` the publication step consumes (deduped per kind+target),
   so the KG-internal lifecycle flip proceeds while the real-world change
   stays human-gated; `auto`/`auto_notify` tiers proceed (with the policy's
   own notification).
6. **Publication as a reviewable branch (`AHE-3.21`).** Promotion used to end
   at a KG lifecycle flip. The evolution→branch bridge closes that gap: a
   *merged* proposal is materialized into a concrete change set and published
   as a **local git branch** — gated by the operational `ActionPolicy`'s
   reserved `merge_promotion` action kind (`OS-5.24`), which ships
   `approval_required`. Nothing is ever pushed or merged to `main`; a human
   reviews the branch and takes it through the normal release flow.

## The evolution→branch bridge (AHE-3.21)

Two modules under `knowledge_graph/research/` implement the bridge:

* **`change_synthesis.py`** — materializes a promoted proposal into a
  `ChangeSet`, with **no LLM calls** (generation belongs to the golden loop's
  synthesize/distill stages):
  * a proposal that embeds explicit file artifacts (`files` /
    `files_json` = `[{"path", "content"}, ...]`) becomes a `kind="code"`
    change set, validated through the tiered RLM sandbox (`ORCH-1.38`):
    per-file syntax compile + best-effort import. Proposed test targets
    (`tests` / `tests_json`) are treated as data and may only run through an
    injected governed sandbox runner. The publisher never executes
    proposal-selected host commands. Sandbox-invalid change sets are never
    published.
  * a prose-only proposal (most SpecDrafts/TeamSpecs) becomes a
    `kind="sdd_plan"` change set: an SDD skeleton under
    `.specify/specs/<topic>/` (`spec.md` + `tasks.md`). For prose, that
    skeleton **is** the reviewable artifact.
* **`change_publisher.py`** — the publication seam. `ChangePublisher` is a
  protocol (`publish(change_set, metadata) -> PublishResult`); the default
  `LocalBranchPublisher` uses plain `git`: it adds a **fresh worktree** off
  the target repo's default branch under `EVOLUTION_WORKTREE_ROOT` (default
  `data_dir()/evolution_worktrees` — never a checkout's working tree),
  applies the change set, optionally delegates bounded targets to an injected
  governed sandbox runner, runs the regression gate (`make_regression_check`,
  `AU-AHE.harness.failure-evolution`), and commits with opaque proposal
  attribution. Persisted graph and audit records contain opaque references,
  verdicts, and counts only—never repository paths, worktree paths, branch
  names, commit hashes, proposal content, or test output.

### The human workflow (approve → publish → merge)

1. A proposal merges through the governed chain (or you decide to publish a
   promoted one). With the shipped policy the merger's own promotion consult
   has already **queued the `merge_promotion` approval**; the bridge consults
   the ActionPolicy again and dedups to that same approval — visible in
   `GET /api/fleet/approvals`.
2. A human grants it: `POST /api/fleet/approvals/grant` with the approval's
   `job_id`. (Granted `merge_promotion` approvals are deliberately *not*
   drained by the fleet reconciler — they belong to the bridge.)
3. The human (or any agent surface) triggers the one-shot publication:
   `graph_evolution(action="publish_proposal", target="<proposal node id>")`
   over MCP, or REST `POST /api/graph/evolution` with
   `{"action":"publish_proposal","target":"..."}`. The granted approval is consumed; the change set
   is synthesized, sandbox-validated, and published as a local branch.
4. Resolve the returned opaque publication references through authorized
   deployment tooling, review the branch, then merge and release through the
   normal governed flow. The bridge never pushes and does not persist local
   filesystem locations.

A deployment that wants zero manual steps can relax the tier with a KG
override — `governance_rule {scope: 'action_policy', kind: 'merge_promotion',
tier: 'auto'}` — at which point a merged proposal publishes its branch
immediately (still local, still human-merged).

### Wiring an MCP-backed publisher

agent-utilities takes no hard dependency on repository-manager. A deployment
that wants publication to flow through its repo tooling (e.g. the
`rm_git`/`rm_worktree` MCP tools, which can also open a hosted PR) registers
its own publisher at startup — same seam pattern as `set_fleet_actuator`:

```python
from agent_utilities.knowledge_graph.research.change_publisher import (
    PublishResult, set_change_publisher,
)

class RepositoryManagerPublisher:
    name = "repository_manager"

    def __init__(self, mcp_call):  # e.g. a bound multiplexer client
        self._call = mcp_call

    def publish(self, change_set, metadata=None):
        # rm_worktree: create a worktree + branch; rm_git: apply/commit (and
        # optionally push + open a PR — that policy lives in the deployment,
        # not in agent-utilities).
        ...
        # Raw repository coordinates remain ephemeral inside the publisher;
        # PublishResult.to_dict() and graph/audit records expose opaque refs.
        return PublishResult(ok=True, branch=..., commit_sha=..., repo_path=...)

set_change_publisher(RepositoryManagerPublisher(mcp_call))
```

## Flags

All flags are typed `AgentConfig` fields (see
[Configuration](configuration.md)); set their aliases in XDG `config.json` or
inject an explicit process override. Concrete secret values remain behind
runtime references.

| Flag | Default | Effect |
| --- | --- | --- |
| `KG_LOOP` | `false` | Hourly propose-only self-evolution cycle (intake → acquire → resolve → distill/synthesize proposals). |
| `KG_LOOP_INTERVAL` / `KG_LOOP_TOPICS` | `3600` / `5` | Tick cadence and per-cycle topic budget. |
| `KG_FAILURE_EVOLUTION` | `auto` | Pull Langfuse failures → `failure_gap` topics → regression-gated remediation when both Langfuse credential refs are configured; explicit `false` opts out. |
| `KG_FAILURE_EVOLUTION_INTERVAL` / `KG_FAILURE_EVOLUTION_WINDOW` | `3600` / `86400` | Tick cadence and telemetry look-back. |
| `KG_ANOMALY_CONSUMER` | `true` | Consume unconsumed `PerformanceAnomaly` nodes into `failure_gap` topics (cheap, LLM-free, propose-only — on by default). |
| `KG_GOLDEN_AUTO_MERGE` | `false` | Allow governed proposal→active promotion. Keep `false` until you trust the proposal stream. |
| `KG_GOLDEN_MERGE_THRESHOLD` | `0.85` | Minimum proposal quality score for auto-merge eligibility. |
| `EVOLUTION_WORKTREE_ROOT` | `data_dir()/evolution_worktrees` | Where the `AHE-3.21` bridge creates fresh git worktrees when publishing a promoted proposal as a local branch. |
| `FLEET_EVENTS_TOKEN_REF` | unset | Secret-provider reference for the `POST /api/fleet/events` monitoring-webhook ingress (`AU-OS.config.fleet-event-ingress`). |
| `FLEET_RECONCILER` | `false` | Desired-state fleet reconciler tick — registry vs observed, converged through the `OS-5.24` ActionPolicy gate (see [Fleet Autonomy](../architecture/fleet_autonomy.md)). |
| `ACTION_POLICY_PATH` | shipped default | Operational action policy (tiers / rate limits / maintenance windows / blast-radius caps); the shipped default keeps every mutating action approval-required (`OS-5.24`). |

## Recommended rollout

1. Configure both Langfuse credential refs, leave content capture and auto-merge
   off, and enable `KG_LOOP=true`. Failure evolution engages automatically unless
   explicitly disabled. Watch the proposal stream (`EvolutionCycle` nodes,
   `failure_gap` Concepts, audit log) for a few cycles. Nothing merges.
2. Point Alertmanager / Uptime Kuma at `POST /api/fleet/events` (set
   `FLEET_EVENTS_TOKEN_REF`) so production incidents also feed the loop. Critical
   events now dispatch the `AU-OS.host.remediation-playbooks` remediation playbooks — with the shipped
   action policy every mutating step lands in `GET /api/fleet/approvals`
   instead of executing.
3. Only once the proposals are consistently sane, consider
   `KG_GOLDEN_AUTO_MERGE=true`. Every promotion remains gated by the
   `AU-AHE.harness.promotion-governance-validator` validator + regression gate and is fully audited; rejected
   proposals stay proposal-only for human review.
4. With auto-merge on, merged proposals additionally queue a
   `merge_promotion` approval (`AHE-3.21`). Work the approve → publish →
   merge loop above; only relax the tier to `auto` once you trust the
   published branches — even then nothing is pushed without a human.

## Closing the loop: generate, verify, ratchet (AHE-3.22 / AHE-3.23 / AU-AHE.evaluation.capability-benchmark-regression-ratchet)

Through `AHE-3.21` the loop could *branch* a code change, but nothing on the live
path ever **generated** the diff — every real proposal fell back to the prose SDD
skeleton and a human wrote the code. These three concepts close that gap; together
they turn "branch a change" into "branch a **verified, capability-ratcheted**
change". All three sit inside the existing `governed_publish` flow, so the OS-5.24
`merge_promotion` ActionPolicy gate (default: human approval queue) still fronts
everything — nothing here can auto-merge or push.

- **AHE-3.22 — autonomous code-synthesis** (`research/code_synthesis.py`). Before
  synthesis, for a proposal that names a resolvable, existing, repo-relative `.py`
  target and carries no embedded files, a single-file generator reads that file and
  emits a `{path, content}` edit, fed into the **unchanged**
  `synthesize_change_set → validate_in_sandbox → publisher` pipeline via the new
  `extra_files` seam. Safety envelope: single attributed `.py` file only;
  un-attributed proposals fall through to the prose skeleton exactly as before; the
  generated file is sandbox-validated (a broken diff is never branched); the default
  generator self-degrades to "no edit" when no model is reachable. The LLM call lives
  in `code_synthesis.py` — `change_synthesis.py` stays generation-free.

- **AU-AHE.evaluation.capability-benchmark-regression-ratchet — capability ratchet** (`research/capability_ratchet.py`). After a
  branch is published, a standing capability suite is run **in that worktree**,
  producing a per-capability score vector compared against a persisted
  `CapabilityScoreVector` baseline node. Every tracked capability must stay
  at-or-above baseline (monotone ratchet); a passing run advances the baseline, the
  first run bootstraps it. A worktree with no probes present is *not measured* and
  never blocks. The recorded `CapabilityRatchetResult` is consulted by the `AU-AHE.harness.promotion-governance-validator`
  promotion-governance gate as an additional predicate.

- **AHE-3.23 — verified apply→verify→rollback**. The keep/abandon decision is the
  authoritative recommendation from the existing `ManifestVerifier`
  (`confirm` / `partial_revert` / `full_revert`, derived from the measured benchmark
  delta), fed the ratchet's before/after scores. On a `*_revert` recommendation — or
  any per-capability regression — `governed_publish` **abandons the branch**
  (`git worktree remove` + `branch -D`); since the branch was never pushed, the
  publication is fully undone. The probe set (`DEFAULT_CAPABILITY_TARGETS`) is tunable.
