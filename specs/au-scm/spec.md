# AU-SCM — Development loop on changesets

Owner: agent-utilities. Cross-repo IDs: EG-SCM-R002, EG-SCM-R003, ARBORA-OPS-R848, ARBORA-OPS-R849.

## Purpose and user stories

The coding harness develops a repository on changesets. The harness
edits files in a virtual workspace. It binds the pending edits to a
changeset base. It runs the fast gates in its sandbox. It submits
through the repository-manager connector tools `source.commit` and
`source.land`. On `SOURCE_CONFLICT{paths}` it rebases on the new head
and retries. Local git is not required.

A developer runs a coding task through arbora. The developer reads the
landed changeset from the graph. The operator sees which changeset
landed. The operator sees why a conflict retried. The operator sees the
gate verdicts behind the landing.

## Requirements and acceptance

- R001: The harness runs the edit, gate, submit and rebase loop per
  run. It edits in a virtual workspace. It runs the fast gates in its
  sandbox. It submits through the connector tools. It rebases on
  `SOURCE_CONFLICT{paths}` and retries. The loop holds no store and
  serves no operation. Acceptance: the loop proof of `AU-SCM-R001`.

## Why a new spec

The cross-repo contracts are prerequisites, not evidence of delivery.
EG-SCM-R002 specifies the changeset record. EG-SCM-R003 specifies
changeset acceptance and refusal with
`SOURCE_CONFLICT{paths}`. ARBORA-OPS-R848 specifies `source.commit` and
`source.land` through the repository-manager connector tools.
ARBORA-OPS-R849 specifies the landing policy. No spec owns the in-run loop that drives those operations. This
spec owns that loop. The `agent_utilities/harness/` package today holds
the edit engine and the quality gates only.

## Scope and interfaces

Owned: the changeset development loop and its models
(`agent_utilities/harness/changeset_loop.py`). Edits apply through
`apply_edits` (`agent_utilities/harness/edit_engine.py:457`). The fast
gates must execute in the `DevWorkspace` sandbox
(`agent_utilities/runtime/workspace.py:103`). The existing
`PreflightGate` (`agent_utilities/harness/quality_gates.py:107`) evaluates
output against a checklist and is not a sandbox command runner. The fast-gate
contract therefore still needs an owner-defined binding. Submission runs
through a new `ChangesetSubmitter` host port. The port fails closed
without a registered provider. arbora registers the provider over the
repository-manager connector tools. The conflict code is
`SOURCE_CONFLICT{paths}` from EG-SCM-R003. The loop re-reads the new
head, re-applies the pending edits and resubmits. Non-goals: the blob
store and the changeset log (epistemic-graph). The served operations
and the landing policy (arbora). The connector tools
(repository-manager). Any local git repository, packfile or remote.

## Traceability

| Requirement | Design section | Test ID | Evidence |
|---|---|---|---|
| R001 | plan.md#architecture | T-001 |  |

## Open questions

The head-read and virtual-workspace materialization interfaces, sandbox fast-gate
execution contract, submission outcome mapping, and run-loop trigger need to be
defined against the owner interfaces before the implementation can be verified.
The served proof also depends on the engine changeset substrate and arbora source
write operations being available.

The local draft's command exit codes are run observations, not authority to land.
Its local-workspace tests run host subprocesses and do not prove sandbox
isolation. The host must bind and verify an isolated workspace before the
served loop executes its fast gates.
`ARBORA-OPS-R849` requires the server to run its defined format, hygiene, secrets,
types, and fast-test gates on the changeset tree before landing. The eventual host
binding must carry the exact tree and verifiable gate evidence to that owner; a
caller-selected command or an in-process event alone cannot satisfy that policy.

At the reviewed owner revisions, [EG-SCM-R002 and R003 are specified](https://github.com/Knuckles-Team/epistemic-graph/blob/3897c07a46d2919299a7fa27fd7d2dc49f7aa8a1/specs/source-changesets/status.json)
and [ARBORA-OPS-R848.2 is pending the engine substrate](https://github.com/Knuckles-Team/arbora/blob/4aaa7bf23b1464f4fb86b9cdb7868aab6531fc4d/specs/hosted-api-operations/tasks.md).
The repository-manager development tools are also required: its
[RM-SCM-R003 tasks](https://github.com/Knuckles-Team/repository-manager/blob/7e3810346c8feb912bb8d82b8a9372c6b9c6deb6/specs/source-changesets/tasks.md)
remain open. That contract names the served tools `rm_source_commit` and
`rm_source_land`, and describes a `CONFLICT` refusal with paths. The mapping
between that refusal and the engine's `SOURCE_CONFLICT` must be established
through the real SDK and served tools; the local draft's `SourceConflict` type
does not establish the mapping.
The Scion SDK's [SDK-SCM source client and virtual-workspace tasks](https://github.com/Knuckles-Team/scion-sdk/blob/0f7c71778e3d5fd3d30ee9faa2da7a7e1b5279fa/specs/SDK-SCM/tasks.md)
are also open. They own the typed client, bounded digest-verified cache, sparse
mount, conflict preservation, and served receipts needed by this path.

The local symbol-edit draft carries an engine request, reads one fresh anchor
after a stale-base refusal, and returns the round-trip refusal. It does not
apply an engine response to the workspace, register a host port, or establish
that a submitted change was landed. The served symbol operation and its
authenticated response remain prerequisites for R002.
