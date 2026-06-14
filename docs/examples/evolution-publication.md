# Worked Example: Publishing an Evolution Proposal as a Reviewable Branch

**What this demonstrates.** CONCEPT:AHE-3.21, the evolution-to-branch bridge:
a golden-loop proposal that cleared promotion governance (CONCEPT:AHE-3.20) is
published as a **local, never-pushed git branch** through the
`graph_orchestrate action=publish_proposal` surface — gated by the OS-5.24
ActionPolicy's `merge_promotion` tier, which by default queues a human
approval. You see the enable flags, the approval round-trip, the exact publish
report, and what the produced branch contains. Deep dive:
[Autonomous evolution](../guides/autonomous-evolution.md).

**Prerequisites (ladder rung).** Single-host rung of
[Deployment configurations](../guides/deployment-configurations.md) with the
KG host running (the REST gateway `python -m agent_utilities`, the headless
`graph-os-daemon`, or `graph-os`) and the target repository
being a git checkout. The evolution daemon ticks are opt-in flags; the one-shot
`publish_proposal` action works without any daemon.

---

## 1. Enable the evolution flags

All typed on `AgentConfig` (`agent_utilities/core/config.py`), all off by
default — the platform is propose-only until you opt in:

```bash
# .env
KG_GOLDEN_LOOP=1                  # 60-min golden-loop daemon tick (KG-2.7)
KG_GOLDEN_LOOP_INTERVAL=3600
KG_GOLDEN_LOOP_TOPICS=5
KG_GOLDEN_DISTILL=1               # distil acquired papers into Insight nodes
KG_GOLDEN_AUTO_MERGE=1            # AHE-3.14 governed promotion (proposal -> active)
KG_GOLDEN_MERGE_THRESHOLD=0.85    # conservative default quality bar

KG_FAILURE_EVOLUTION=1            # AHE-3.18 Langfuse failures -> failure_gap topics
KG_FAILURE_EVOLUTION_WINDOW=86400
KG_FAILURE_REGRESSION_DATASET=1   # failures double as the regression gate dataset

# AHE-3.21 publication:
EVOLUTION_WORKTREE_ROOT=/var/lib/agent-utilities/evolution_worktrees
# Empty default resolves to data_dir()/evolution_worktrees — the publisher
# always creates a FRESH worktree there, never writing into a checkout's tree.

# OS-5.24 policy file (empty default = the shipped conservative policy):
# ACTION_POLICY_PATH=/etc/agent-utilities/action-policy.yml
```

## 2. A proposal gets promoted (AHE-3.14 + AHE-3.20)

The golden loop synthesizes proposals as KG nodes. With `KG_GOLDEN_AUTO_MERGE`
on, each cycle runs the governed auto-merge: a proposal must clear the quality
threshold **and** the production promotion-governance validator
(`agent_utilities/knowledge_graph/research/promotion_governance.py` — SHACL/
governance validity, regression must not lower a tracked metric, a recorded
`hold` blocks). The merger's promotion decision also consults the OS-5.24
ActionPolicy under `kind=merge_promotion`: a `deny` blocks the lifecycle flip
outright, while the shipped `approval_required` tier queues the same
`ActionApproval` step 3 below dedups to. Passing proposals are promoted and
audited; failing ones stay proposal-only. Promotion is a prerequisite here —
`publish_proposal` only materializes and publishes **what was already
promoted**.

For this walkthrough the promoted proposal node is:

```text
proposal:retry_backoff:demo1   (a SpecDraft: title/problem/approach, status: promoted)
```

A proposal may also embed explicit file artifacts (`files_json` property:
`[{"path": ..., "content": ...}, ...]`, plus optional `tests_json` pytest
targets) — that becomes a `kind="code"` change set whose Python files are
validated in the tiered RLM sandbox (CONCEPT:ORCH-1.38) before publication.
Prose-only proposals (the common case) become a `kind="sdd_plan"` change set:
an SDD plan skeleton under `.specify/specs/<topic>/`.

## 3. First publication attempt queues an approval

```json
{
  "tool": "graph_orchestrate",
  "arguments": {
    "action": "publish_proposal",
    "task": "proposal:retry_backoff:demo1"
  }
}
```

REST twin: `POST /api/graph/orchestrate/publish-proposal` with body
`{"proposal_id": "proposal:retry_backoff:demo1"}`.

The governed entry point (`governed_publish` in
`agent_utilities/knowledge_graph/research/change_publisher.py`) asks the
ActionPolicy to decide `kind=merge_promotion`. The shipped default policy
(`deploy/action-policy.default.yml`) pins that kind to `approval_required`:

```yaml
- {kind: merge_promotion, target: "*", tier: approval_required}
```

**Expected output** (no prior grant — an `ActionApproval` node is queued):

```json
{
  "proposal_id": "proposal:retry_backoff:demo1",
  "source": "publish_proposal",
  "decision": "queue_approval",
  "approval_id": "action_approval:9f31c2ab44d0",
  "status": "approval_queued",
  "detail": "tier requires human approval"
}
```

If the policy engine itself is unreachable the report is
`{"status": "denied", "detail": "action policy unavailable (fail closed): ..."}`
— the gate fails closed.

## 4. A human grants the approval

The pending queue and the grant both live on the fleet supervisory plane
(`agent_utilities/gateway/fleet.py`):

```bash
curl -s http://localhost:8000/api/fleet/approvals          # list pending
curl -s -X POST http://localhost:8000/api/fleet/approvals/grant \
  -H 'content-type: application/json' \
  -d '{"job_id": "action_approval:9f31c2ab44d0", "decision": "approved"}'
```

**Expected output:**

```json
{
  "status": "success",
  "result": {"approval_id": "action_approval:9f31c2ab44d0", "decision": "approved"}
}
```

## 5. Re-run `publish_proposal` — the granted approval is consumed

Repeat the exact call from step 3. `governed_publish` finds the granted
`merge_promotion` approval for this proposal, consumes it (stamping it
`executed`), synthesizes the change set, refuses sandbox-invalid code, and
publishes through the resolved `ChangePublisher`.

**Expected output** — this report was captured by actually running
`publish_proposal` in this tree (fresh demo proposal, default
`LocalBranchPublisher`, pre-granted approval):

```json
{
  "proposal_id": "proposal:retry_backoff:demo1",
  "source": "publish_proposal",
  "decision": "approved",
  "approval_id": "action_approval:demo-grant-1",
  "change_kind": "sdd_plan",
  "publish": {
    "ok": true,
    "proposal_id": "proposal:retry_backoff:demo1",
    "branch": "evolution/add-jittered-retry-backoff-to-the-harvest-client-2945a9fb",
    "commit_sha": "73ce4e056803dfd73d92e05e214350730d47dfe9",
    "repo_path": "/home/apps/worktrees/au-docs-refresh",
    "worktree_path": ".../evolution_worktrees/evolution--add-jittered-retry-backoff-to-the-harvest-client-2945a9fb",
    "gate_result": "not_run",
    "tests_passed": null,
    "test_report": null,
    "detail": "published local branch evolution/add-jittered-retry-backoff-to-the-harvest-client-2945a9fb (NOT pushed — review, then merge through the normal release flow)"
  },
  "status": "published",
  "execution_id": "action_execution:959bb38bc5f2"
}
```

Field semantics:

- `gate_result`: `pass` | `hold` | `not_run` — the injected regression gate
  (e.g. the AHE-3.18 failure analyzer's `make_regression_check`). A `hold` does
  **not** delete the branch; the branch is the review artifact either way.
- `tests_passed`: `null` when the proposal named no pytest targets; otherwise
  the verdict of running them inside the fresh worktree (`test_report` carries
  return code, targets and output tail).
- A missing proposal returns
  `{"status": "not_found", "detail": "no proposal node with id ..."}`; a
  sandbox-rejected code change set returns `{"status": "validation_failed", ...}`.

## 6. Inspect what the publisher created

The default publisher (`LocalBranchPublisher`) is plain argv-only `git`. It
creates a **fresh worktree** off the target repo's default branch under
`EVOLUTION_WORKTREE_ROOT`, writes the change set, and commits — citing the
proposal and concept ids. It never pushes and never merges:

```bash
git -C <repo_path> log --oneline -1 evolution/add-jittered-retry-backoff-to-the-harvest-client-2945a9fb
git -C <repo_path> show --stat --format= evolution/add-jittered-retry-backoff-to-the-harvest-client-2945a9fb
```

**Expected output** (captured from the same run):

```text
73ce4e0 evolution: sdd_plan change for Add jittered retry backoff to the harvest client

 .specify/specs/add-jittered-retry-backoff-to-the-harvest-client/spec.md  | 29 +++++++++++++++
 .specify/specs/add-jittered-retry-backoff-to-the-harvest-client/tasks.md |  8 ++++++
 2 files changed, 37 insertions(+)
```

The commit message body records the proposal id, concept ids, regression-gate
verdict and targeted-test verdict.

Targeting: by default the publisher targets *the agent-utilities checkout the
package runs from* (`default_target_repo()` walks up to the nearest `.git`).
To publish into a different repository, construct
`LocalBranchPublisher(repo_path=...)` — or register a deployment publisher
(e.g. one backed by repository-manager MCP tools) via
`set_change_publisher(...)`; the `ChangePublisher` protocol is the seam.

## What landed in the KG

```text
(:ActionApproval {id, kind: "merge_promotion", target: <proposal id>, status: "executed"})
(<proposal>)-[:PUBLISHED_AS]->(:ProposalPublication {branch, commit_sha, gate_result, ...})
(<proposal> {publish_branch, publish_commit, publish_gate})       # stamped in place
(:ActionExecution {kind: "merge_promotion", actuator: "local_branch", ok: true, ...})
AuditLogger entry: action "golden_loop.publish_proposal"
```

---

*Verification: smoke-run against this tree (2026-06-11). Executed:
`python3 -m pytest tests/test_evolution_pr_bridge.py -q` (passed, as part of a
38-test run with the identity suite), plus a live one-off invocation of
`publish_proposal(engine, "proposal:retry_backoff:demo1")` with an in-memory
engine double carrying a pre-granted approval — the step-5 report and step-6
branch log/stat were captured verbatim from that run (worktree under a temp
`EVOLUTION_WORKTREE_ROOT`; the demo branch produced by that run is a throwaway
artifact, not part of this change).
The step-3 approval-queued and step-4 grant outputs were reviewed against code
(`governed_publish`, `ActionPolicy.queue_approval`, `fleet_grant_approval`)
rather than captured live.*
