# Spec: EPIC 2 — Interactive Execution Loop (ORCH-1.35 + sidecar isolation)

> Design: `.specify/design/orch-1.35-midturn-tool-result-injection/design.md`. Depends on EPIC 1.
> Item 15 (OS-5.11 run-scoped token) lands with this epic.

## Pre-Flight Checklist
- [x] Design exists; KG-nearest table (ORCH-1.35 max 0.64 vs ORCH-1.3) <0.70 (provisional — confirm live).
- [x] Extension points: `graph/hsm.py` (new `WAITING_HOST`), `server/routers/human.py`, `core/execution/engine.py`.
- [x] Wire-First: ≤1 hop from `/api/runs/{id}/tool-result`.
- [ ] Live `kg_search`: if ORCH-1.35 ≥0.70 vs ORCH-1.3, downgrade to augmentation.

## User Stories
### US-1 — Hold a turn open on tool_use
**As** the engine, **I want** to pause a step on `tool_use`, record `pending_tool_use_ids`, and transition to `WAITING_HOST`, **so that** interactive tools work headless.
- **AC1**: A step emitting `tool_use` enters `WAITING_HOST` without erroring; stdin stays open for `stream-json` adapters.
- **AC2**: `prompt_delivery` ∈ {`args`,`stdin-text`,`stdin-jsonl`} honored per adapter (from EPIC 1).

### US-2 — Resume via tool-result
**As** a human or agent, **I want** `POST /api/runs/{id}/tool-result` to inject the result and resume, **so that** the same turn completes.
- **AC3**: POST writes a `user/tool_result` JSONL line and resumes; run reaches a final result (integration).
- **AC4**: An A2A skill can answer the same resume entry (no UI required).
- **AC5**: A held turn past `host_answer_timeout` auto-fails with a clear status (config).

### US-3 — Per-run sidecar isolation
**As** an operator running concurrent runs, **I want** each run's subprocess isolated by a typed process stamp + UDS path, **so that** runs can't cross-talk.
- **AC6**: `ProcessStamp{app,mode,namespace,ipc,source}` resolves a unique socket under `.tmp/agent-utilities/<namespace>/`.
- **AC7**: Two concurrent runs in distinct namespaces use distinct sockets; neither can read the other's IPC (unit). *(POSIX UDS; Windows named pipes deferred.)*

### US-4 — Run-scoped token (OS-5.11)
**As** a security owner, **I want** each run minted a scoped token validated by `tool_guard`, **so that** tool access is least-privilege.
- **AC8**: `mint_token(actor, run)` produces a token bound to runId/project/endpoints/ops/expiry; `tool_guard.validate` rejects expired/out-of-scope, passes in-scope (unit).

## Non-Functional Requirements
- `@pytest.mark.concept(id="ORCH-1.35")` / `("OS-5.11")`; no network; ≤60s.
- Non-interactive runs never enter `WAITING_HOST` (zero regression).
- Security tests: scope-escape, expiry, revocation for OS-5.11.
- Docs: `docs/pillars/1_graph_orchestration/ORCH-1.35.md`, `docs/pillars/5_agent_os_infrastructure/OS-5.11.md`; concepts.yaml regen.

## Tasks
- [ ] T1 `graph/hsm.py`: add `WAITING_HOST` state + transitions. *(unit)*
- [ ] T2 `core/execution/engine.py`: held-turn state (`pending_tool_use_ids`, `resume()`), stdin-open for `stdin-jsonl`. *(integration)*
- [ ] T3 `server/routers/human.py`: `/api/runs/{id}/tool-result` route + timeout policy. *(integration)*
- [ ] T4 `security/sidecar_runtime.py`: `ProcessStamp` + UDS isolation; `SidecarRuntime.spawn/invoke`. *(unit: namespace isolation)*
- [ ] T5 `graph/executor.py`: route `step.runtime=='sidecar'` → `SidecarRuntime`. *(integration)*
- [ ] T6 (OS-5.11) `security/brain_context.py` `mint_token`; `security/tool_guard.py` `validate`; inject at dispatch. *(unit: scope/expiry/revoke)*
- [ ] T7 Docs/concepts/wiring-audit/CHANGELOG.
