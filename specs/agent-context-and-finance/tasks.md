# Tasks

**States:** TODO → IN PROGRESS → IMPLEMENTED at merged head → VERIFIED by exact-head gates → ACCEPTED. Current rows are TODO pending audit.

| Task | IDs | State | Evidence required |
|---|---|---|---|
| Retrieval path and context sizing | AU-CONTEXT-R001, AU-CONTEXT-R004 | [x] IMPLEMENTED at merged head | live AU→EG caller, cited/budgeted positive and refusal tests |
| Shadow re-embed and admission proposals | AU-CONTEXT-R002, AU-CONTEXT-R003 | [x] IMPLEMENTED at merged head | lease, quality receipt, rollback and review proof |
| Finance explanation and scheduler integration | AU-CONTEXT-R005 | TODO | math-first sourced explanation and served schedule |
| Paper/live boundary and AU finance cut | AU-CONTEXT-R006, AU-CONTEXT-R007 | TODO | golden parity, deleted duplicate modules, approval refusal |
| Calibrated informational recommendations | AU-CONTEXT-R008 | TODO | per-strategy scorecard, abstention and no-order proof |

Record a verdict per ID, not merely per grouped task.

AU-CONTEXT-R001..R004 landed on `main` at `88c61dc9164c983a52fdc8b170c10fca2e208d8a`
("feat(decide,retrieval): retrieval-path candidates, governed re-embedding,
admission feedback, certified context sizing", 2026-10-01). That commit
message names all four requirement IDs explicitly and states R005..R008 are
out of its scope. The `status.json` evidence entries above still cite the
superseded pre-merge branch commits (683965fb4c, 14febbf9e3, c906e94598,
aa02d74fa0); those SHAs are no longer reachable from any branch tip in this
repository (the branch they lived on was rebased away) and must not be
re-implemented. The orchestrator should re-audit `status.json` against
`88c61dc9164c983a52fdc8b170c10fca2e208d8a` to promote R001..R004 to LANDED.
- [ ] **AU-CONTEXT-R006.1:** agent-utilities keeps finance agent roles only; paper trading stays isolated (producer, this repo).
- [ ] **AU-CONTEXT-R006.2:** Live orders use the agent-connector-sdk's governed write-back contract (cross-repo; depends on R006.1).
- [ ] **AU-CONTEXT-R006.3:** A finance widget replaces the placeholder widget in graph-os (cross-repo; depends on R006.2).
