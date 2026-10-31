# MERISTEM-SCM — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

The harness drives a repository on changesets in one loop: edit, gate, submit, rebase. It applies received decisions and runs one task run per received task. It reuses `apply_edits`, `DevWorkspace`, the host ports and the existing coding toolset.

## Technical Context

- **Language/Version**: Python 3.14 (floor 3.12)
- **Primary Dependencies**: The meristem harness package and the host ports.
- **Storage**: N/A. The loop holds no store.
- **Testing**: pytest. Each row names its test and proof level.
- **Target Platform**: Linux runners and developer machines
- **Project Type**: Library
- **Performance Goals**: N/A. A row states its own budget.
- **Constraints**: The rebase bound is 3. Every port fails closed without a provider.
- **Scale/Scope**: The rows of requirements.md.

## Constitution Check

- **Article one. The harness applies decisions and never makes them**: Pass. Rows R003 to R011 receive values and decide nothing.
- **Article two. Dependencies stay minimal**: Pass. The change adds no dependency.
- **Article three. A boundary owner exists for every directory**: Pass. `meristem/harness/` keeps its owner.

## Project Structure

```text
meristem/harness/changeset_loop.py      existing
meristem/harness/coding_toolset.py      existing
meristem/harness/task_run.py            new
tests/harness/test_task_run.py          new
tests/harness/test_task_run_served.py   new
```

## Phase 0: Research

No unknowns; no research.md.

## Phase 1: Design

### Existing system and reuse

The harness package keeps the in-run mechanism. `apply_edits` at
`meristem/harness/edit_engine.py:457` applies search/replace and
unified-diff edits to file text. `PreflightGate` at
`meristem/harness/quality_gates.py:107` runs the pre-emit
checklist against output text. `DevWorkspace` at
`meristem/runtime/workspace.py:103` drives a sandbox backend
with typed actions. The backends are `local_workspace.py` and
`docker_workspace.py`. The run loop enters through `execute_graph` at
`meristem/graph/runner.py:80`. The host ports live in
`meristem/host_ports.py`. Every port fails closed without a
registered provider. The loop reuses these pieces and adds one submit
port beside them.

### Architecture

One module owns the loop: `meristem/harness/changeset_loop.py`.
The loop is one cycle: edit, gate, submit, conflict. Each step is one
function with one refusal path. The module holds no store and serves no
operation. The edit step applies edits through `apply_edits` in the
virtual workspace. The gate step runs the fast gates through
`PreflightGate` in the sandbox. The submit step sends the pending
changeset through the `ChangesetSubmitter` port. arbora registers the
provider over the repository-manager connector tools (`source.commit`,
`source.land`; ARBORA-OPS-R848). The conflict step reacts to
`SOURCE_CONFLICT{paths}` (MEDULLA-SCM-R003). It re-reads the new head,
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

### Operator refinement (2026-10-19)

Graph-driven development goes through the tools the
repository-manager connector exposes. The connector tools are the
development surface inside arbora MCP. AU owns only the in-run loop
behavior. The loop edits in a virtual workspace. It runs fast gates in
the sandbox. It submits through the connector tools. It rebases and
retries on `SOURCE_CONFLICT{paths}`. AU does no storage or serving.
Local git is not required.

### Interfaces and data model

`ChangesetSubmitter` is the host port beside the run ports. Its
`submit` method takes a `PendingChangeset{base_head, edits, run_ref}`
and returns one outcome. The outcome is `landed{changeset_id}` or
`SOURCE_CONFLICT{paths}`. `PendingChangeset` is a frozen dataclass. It
carries references and edit text, not a store. The loop keeps the
pending changeset in run state. It re-reads the head through the same
port before each rebase.

### Live integration path

arbora registers `ChangesetSubmitter` at startup. The provider maps
`submit` to the connector tools `source.commit` and `source.land`. The
landing policy of ARBORA-OPS-R849 runs on the server. The loop reports
the landed changeset id through the existing `RunEventSink`. The run
then reads the landed changeset from the graph.

### Quality and release gates

Focused tests: `uv run --frozen python -m pytest
tests/harness/test_changeset_loop.py -q`. Language-native gates: ruff,
mypy, CCCC, jscpd, dupehound and KISS on the changed code through the
repository gates in `.config/pre-commit.yaml` and `release.yml`.

### Risks, alternatives, and decisions

The loop and the server risk drift on conflict semantics. The decision:
the server owns acceptance and the rebase judgment (MEDULLA-SCM-R003). The
loop owns only the re-application of its own edits. The rejected
alternative kept a local merge inside the loop. It fails the
no-duplicate-authority check. The retry bound stops an unbounded loop.
The bound is a received value with a default and maximum of 3 rebases. Exhaustion
refuses with `conflict_retry_exhausted`. The loop then hands the run
back to the caller.

### Symbol-level edits (R002)

The harness sends a symbol edit instead of a file write when the file has a parser. The engine
projects the edit to bytes and confirms it by a re-parse (MEDULLA-SCM-R013). The harness only applies
the answer: it retries a stale base once and returns a round-trip refusal to the model.

### Consumer rows for medulla decisions (2026-10-30)

Rows MERISTEM-SCM-R003 to MERISTEM-SCM-R007 file the in-run harness obligations of the medulla decision engine. The rows cover in-run coding over changesets. Each rollup row names one medulla row. Each child row names the meristem file, the planned test and the received decision. The harness decides nothing in these rows. Any threshold is a received value with a fallback default. Medulla issue 1029 tracks the source decisions.

### Task run

`MEDULLA-SDLC-R015.3` hands one task with its context to an agent run. The run enters through `execute_graph`. `CodingToolset.run_task` builds a `TaskRunContext` from the run input and calls `run_task_run`. That function calls `run_changeset_cycle` once and returns a `TaskRunResult` with evidence. A failure returns a typed code, a stage and the evidence for `MEDULLA-SDLC-R015.5`. The harness holds no planner, never picks the next task and never edits a spec.

## Phase 2: Task approach

Tasks follow the rows. The model row precedes the behavior row. The wiring row follows both. The hot-path proof closes the list. Rows R011.1 and R011.4 may run in parallel.

## Complexity Tracking

None. The constitution check passed.

## Build contract

Focused tests: `uv run --frozen python -m pytest tests/harness/test_changeset_loop.py tests/harness/test_task_run.py -q`. Language-native gates: ruff, mypy, CCCC, jscpd, dupehound and KISS run through `.config/pre-commit.yaml` and `release.yml`. The new module holds no scanner suppression and no copy of the loop.

## Production caller chain

Arbora run start (`ARBORA-OPS-R896.3`) calls `execute_graph` (`meristem/graph/runner.py:96`). That calls `build_agent` (`meristem/graph/executor.py:70`). That appends `CodingToolset`, whose `run_task` calls `run_task_run` and then `run_changeset_cycle`. Arbora registers `ChangesetSubmitter` at startup. The provider maps `submit` to `source.commit` and `source.land`. No obsolete path exists to delete.

## Cross-repo seams

- `MEDULLA-SDLC-R015` (medulla): the decision that assembles the context and accepts the result.
- `ARBORA-OPS-R896.3` and `ARBORA-OPS-R896.4` (arbora): run start and run observe.
- `MEDULLA-SCM-R003` and `ARBORA-OPS-R849`: acceptance, refusal and landing policy.
