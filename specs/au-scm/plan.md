# AU-SCM — Design and implementation plan

Governing spec: [spec.md](spec.md).

## Existing system and reuse

The harness package keeps the in-run mechanism. `apply_edits` at
`agent_utilities/harness/edit_engine.py:457` applies search/replace and
unified-diff edits to file text. `PreflightGate` at
`agent_utilities/harness/quality_gates.py:107` runs the pre-emit
checklist against output text. `DevWorkspace` at
`agent_utilities/runtime/workspace.py:103` drives a sandbox backend
with typed actions. The backends are `local_workspace.py` and
`docker_workspace.py`. The run loop enters through `execute_graph` at
`agent_utilities/graph/runner.py:80`. The host ports live in
`agent_utilities/host_ports.py`. Every port fails closed without a
registered provider. The loop reuses these pieces and adds one submit
port beside them.

## Architecture

One module owns the loop: `agent_utilities/harness/changeset_loop.py`.
The loop is one cycle: edit, gate, submit, conflict. Each step is one
function with one refusal path. The module holds no store and serves no
operation. The edit step applies edits through `apply_edits` in the
virtual workspace. The gate step runs the fast gates through
`PreflightGate` in the sandbox. The submit step sends the pending
changeset through the `ChangesetSubmitter` port. arbora registers the
provider over the repository-manager connector tools (`source.commit`,
`source.land`; ARBORA-OPS-R848). The conflict step reacts to
`SOURCE_CONFLICT{paths}` (EG-SCM-R003). It re-reads the new head,
re-applies the pending edits and resubmits. The retry bound is a
received value with a default and maximum of 3 rebases. Exhaustion refuses with
`conflict_retry_exhausted`.

The local draft requires a host-bound `SandboxGateRunner` before it
materializes a workspace. The host must verify the isolated execution
boundary and bind it to the same workspace. It applies the edit set and
reads the resulting files through that boundary as well as running gates.
The test double uses a local backend only to check control flow and does not
prove isolation or race-safe file handling.
The draft reports only `submitted` after its host port accepts a candidate.
It does not emit a landed event or claim that the landing policy has run.

## Operator refinement (2026-10-19)

Graph-driven development goes through the tools the
repository-manager connector exposes. The connector tools are the
development surface inside arbora MCP. AU owns only the in-run loop
behavior. The loop edits in a virtual workspace. It runs fast gates in
the sandbox. It submits through the connector tools. It rebases and
retries on `SOURCE_CONFLICT{paths}`. AU does no storage or serving.
Local git is not required.

## Interfaces and data model

`ChangesetSubmitter` is the host port beside the run ports. Its
`submit` method takes a `PendingChangeset{base_head, edits, run_ref}`
and returns one outcome. The outcome is `landed{changeset_id}` or
`SOURCE_CONFLICT{paths}`. `PendingChangeset` is a frozen dataclass. It
carries references and edit text, not a store. The loop keeps the
pending changeset in run state. It re-reads the head through the same
port before each rebase.

## Live integration path

arbora registers `ChangesetSubmitter` at startup. The provider maps
`submit` to the connector tools `source.commit` and `source.land`. The
landing policy of ARBORA-OPS-R849 runs on the server. The loop reports
the landed changeset id through the existing `RunEventSink`. The run
then reads the landed changeset from the graph.

## Quality and release gates

Focused tests: `uv run --frozen python -m pytest
tests/harness/test_changeset_loop.py -q`. Language-native gates: ruff,
mypy, CCCC, jscpd, dupehound and KISS on the changed code through the
repository gates in `.config/pre-commit.yaml` and `release.yml`.

## Risks, alternatives, and decisions

The loop and the server risk drift on conflict semantics. The decision:
the server owns acceptance and the rebase judgment (EG-SCM-R003). The
loop owns only the re-application of its own edits. The rejected
alternative kept a local merge inside the loop. It fails the
no-duplicate-authority check. The retry bound stops an unbounded loop.
The bound is a received value with a default and maximum of 3 rebases. Exhaustion
refuses with `conflict_retry_exhausted`. The loop then hands the run
back to the caller.

## Symbol-level edits (R002)

The harness sends a symbol edit instead of a file write when the file has a parser. The engine
projects the edit to bytes and confirms it by a re-parse (EG-SCM-R013). The harness only applies
the answer: it retries a stale base once and returns a round-trip refusal to the model.
