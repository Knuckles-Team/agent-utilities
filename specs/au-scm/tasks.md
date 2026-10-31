# MERISTEM-SCM — Implementation tasks

Governing spec: [spec.md](spec.md). Design: [plan.md](plan.md). Rows: [requirements.md](requirements.md).

Each task line reads `- [ ] T001 [P] [US1] <ID>-R012.4 description`. The task number and the row ID are required.
`[P]` marks a task that may run in parallel. `[US1]` names the user story. Order tasks producer first.

- [ ] T001 [US1] MERISTEM-SCM-R001 Development loop on changesets.
- [ ] T002 [US1] MERISTEM-SCM-R002 Symbol-level edits in the run loop.
- [ ] T003 [US2] MERISTEM-SCM-R003 Consumer of MEDULLA-DECISION-ENGINE-R036 (rollup).
- [ ] T004 [P] [US2] MERISTEM-SCM-R003.1 Emit verified agent facts and apply the plan.
- [ ] T005 [US2] MERISTEM-SCM-R004 Consumer of MEDULLA-DECISION-ENGINE-R135 (rollup).
- [ ] T006 [P] [US2] MERISTEM-SCM-R004.1 Run pre-plan tests and attach their receipt to the program commit.
- [ ] T007 [US2] MERISTEM-SCM-R005 Consumer of MEDULLA-DECISION-ENGINE-R136 (rollup).
- [ ] T008 [P] [US2] MERISTEM-SCM-R005.1 Declare a prediction per edit.
- [ ] T009 [US2] MERISTEM-SCM-R006 Consumer of MEDULLA-DECISION-ENGINE-R138 (rollup).
- [ ] T010 [P] [US2] MERISTEM-SCM-R006.1 Adversarial verifier receipts.
- [ ] T011 [US2] MERISTEM-SCM-R007 Consumer of MEDULLA-DECISION-ENGINE-R161 (rollup).
- [ ] T012 [P] [US2] MERISTEM-SCM-R007.1 Regression test on both revisions.
- [ ] T013 [US2] MERISTEM-SCM-R008 Engine decision port for the coding loop.
- [ ] T014 [US2] MERISTEM-SCM-R010 Coding toolset is the caller of the changeset cycle.
- [ ] T015 [US3] MERISTEM-SCM-R011 One task run applies one received task (rollup).
- [ ] T016 [US3] MERISTEM-SCM-R011.1 Model: task context and task result.
- [ ] T017 [P] [US3] MERISTEM-SCM-R011.2 Behavior: run exactly one task.
- [ ] T018 [P] [US3] MERISTEM-SCM-R011.3 Wiring: the coding toolset starts the task run.
- [ ] T019 [P] [US3] MERISTEM-SCM-R011.4 Evidence and typed failure.
- [ ] T020 [P] [US3] MERISTEM-SCM-R011.5 Hot-path proof: served task run.
