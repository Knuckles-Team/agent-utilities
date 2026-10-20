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

The cross-repo rows landed first. EG-SCM-R002 stores the changeset
record. EG-SCM-R003 accepts the changeset and refuses with
`SOURCE_CONFLICT{paths}`. ARBORA-OPS-R848 serves `source.commit` and
`source.land`. The development surface runs through the
repository-manager connector tools. ARBORA-OPS-R849 applies the landing
policy. No spec owns the in-run loop that drives those operations. This
spec owns that loop. The `agent_utilities/harness/` package today holds
the edit engine and the quality gates only.

## Scope and interfaces

Owned: the changeset development loop and its models
(`agent_utilities/harness/changeset_loop.py`). Edits apply through
`apply_edits` (`agent_utilities/harness/edit_engine.py:457`). The fast
gates run through `PreflightGate`
(`agent_utilities/harness/quality_gates.py:107`) in the `DevWorkspace`
sandbox (`agent_utilities/runtime/workspace.py:103`). Submission runs
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

None.
