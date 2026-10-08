# AU-BASELINE-001 — Implementation tasks

Status: BUILDING. Governing spec: [spec.md](spec.md). Design: [plan.md](plan.md).

- [x] Inventory the daemon start, the durable queue, the skill, prompt and code ingest paths, and the SDK content helper.
- [x] Add the stage-to-class table and task routing (`AU-BASELINE-R004`).
- [x] Add the baseline planner, enqueuer and background start; hook it into `start_background_daemons` (`AU-BASELINE-R001`, `AU-BASELINE-R002`, `AU-BASELINE-R003`).
- [x] Resolve `skill-provider:<name>` targets in the skill workflow handler; add the `baseline_prompts` maintenance tick.
- [x] Register ontology providers on factory-built servers through the SDK helper (`AU-BASELINE-R005`).
- [x] Add the configuration fields and regenerate the runtime configuration catalog.
- [x] Run focused tests, ruff, complexity, env-sprawl, swallowed-error and event-loop gates.
- [ ] Merge the SDK helper and release it; raise the AU SDK floor to that release.
- [ ] Deploy a daemon-role process on an empty store; record the WorkItem list and the sparse-index ratio.
- [ ] Route the queue class through the EG decision once EG-DECISION-ENGINE-R030 ships.
