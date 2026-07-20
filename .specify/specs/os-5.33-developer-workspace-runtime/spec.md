# Spec: Developer-Workspace Runtime (AU-OS.scaling.bridge-developer-workspace-mutating / ORCH-1.46 / KG-2.64)

## Pre-Flight Checklist (Mandatory — DSTDD)

- [x] **KG search completed** — `.specify/design/os-5.33-developer-workspace-runtime/design.md` exists
- [x] **Extension point identified** — ORCH-1.38 transport reused; new concepts justified
- [x] **C4 diagram created** — in design doc
- [x] **No new CONCEPT: tag** without pillar reference — OS/ORCH/KG referenced
- [ ] **`code-enhancer` audit** run against proposed changes
- [ ] **Design validation passes**

## Design Reference

→ [`../../design/os-5.33-developer-workspace-runtime/design.md`](../../design/os-5.33-developer-workspace-runtime/design.md)

## User Stories

### US-1: Stateful workspace
**As an** SWE agent, **I want** a long-lived workspace where my `cd`, env exports, and file
edits persist across steps, **so that** I can run a real edit→build→test loop instead of
re-establishing state on every snippet.

**Acceptance Criteria:**
- [ ] `DevWorkspace.start()` provisions a backend; `act(CmdRunAction("cd /repo && export X=1"))`
      then `act(CmdRunAction("echo $X && pwd"))` reflects persisted cwd/env.
- [ ] `FileWriteAction` then `FileReadAction` returns the written content.
- [ ] `TestRunAction` returns a `TestResultObservation` with passed/failed counts parsed from junit.
- [ ] `stop()` tears the backend down; a reaper reclaims leaked containers by `run_id`/idle.

### US-2: Typed action/observation protocol
**As the** orchestration engine, **I want** every step expressed as a typed Action that yields a
typed Observation over a single bridge, **so that** the loop is uniform, streamable, and persistable.

**Acceptance Criteria:**
- [ ] `events.py` defines frozen discriminated unions; every event carries `run_id/step/ts/actor`.
- [ ] The bridge round-trips actions/observations; outputs >64KB stream in chunks without truncation.
- [ ] Unknown action kinds are rejected with a typed error (not a crash).

### US-3: KG provenance grounded to symbols
**As the** golden loop, **I want** each action/observation mirrored into the KG and edits grounded
to the `Code` symbols they touched, **so that** failures are attributable and runs are replayable.

**Acceptance Criteria:**
- [ ] After a `FileEditAction`, `graph_query` finds `(:WorkspaceAction)-[:MUTATED]->(:Code)` for
      the touched symbol(s).
- [ ] `(:RunTrace)-[:HAS_ACTION]->(:WorkspaceAction)-[:PRODUCED]->(:WorkspaceObservation)` and
      `[:NEXT]` replay ordering exist.
- [ ] Provenance mirroring degrades gracefully (logs, never raises) when the KG is cold/unavailable.

### US-4: Policy-gated mutation
**As the** platform operator, **I want** shell/file mutations gated by the existing fail-closed
ActionPolicy, **so that** the runtime inherits the same guardrails as the rest of the fleet.

**Acceptance Criteria:**
- [ ] `workspace.cmd|write|edit` are registered mutating kinds; a denied policy decision blocks
      the action and returns an error observation.

## Non-Functional Requirements

- [ ] All existing tests continue to pass (zero regression) — `SandboxCapabilities.workspace`
      defaults `False`, snippet router unaffected.
- [ ] Pre-commit hooks pass cleanly.
- [ ] `docs/pillars/5_agent_os_infrastructure.md` + `1_graph_orchestration.md` +
      `2_epistemic_knowledge_graph.md` updated; CHANGELOG + AGENTS.md + README synced.
- [ ] `CONCEPT:AU-OS.scaling.bridge-developer-workspace-mutating`, `CONCEPT:AU-ORCH.reactive.action-dispatcher`, `CONCEPT:AU-KG.enrichment.atomic-triple-extraction` markers added; `concepts.yaml`
      regenerated; `check_concepts.py` + `check_wiring.py` (≤3-hop) green.
- [ ] `local` backend is the zero-infra floor (no Docker required) so the runtime works out-of-the-box.
- [ ] Live-path tests: `tests/runtime/test_workspace_lifecycle.py`, `test_event_protocol.py`,
      `test_provenance_mirror_live_path.py`.
