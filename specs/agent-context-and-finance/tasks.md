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
- [x] **AU-CONTEXT-R006.1:** agent-utilities keeps finance agent roles only; paper trading stays isolated (producer, this repo). IMPLEMENTED at merged head: `tests/unit/finance/test_quant_live_order_refusal.py` locks the registered `quant` tool's `execute` domain refusing every submit/cancel/status call in both paper and live mode.
- [x] **AU-CONTEXT-R006.2:** Live orders use the agent-connector-sdk's governed write-back contract (cross-repo; depends on R006.1). Not built in this repo — owner is `agent-connector-sdk`; tracked here only for the dependency edge.
- [x] **AU-CONTEXT-R006.3:** A finance widget replaces the placeholder widget in graph-os (cross-repo; depends on R006.2). Not built in this repo — owner is `graph-os`; tracked here only for the dependency edge.
- [x] **AU-CONTEXT-R007.1:** boundary test (`tests/unit/finance/test_r007_finance_module_boundary.py`) pinning the current `agent_utilities/domains/finance/*.py` module set so it can only shrink, plus a typed delegation seam (`kelly_size_via_eg_or_local`/`resolve_finance_primitives_client` in `agent_utilities/api/finance_delegation.py`) that calls EG finance primitives when available and otherwise falls back to the caller-supplied local calculation — net-new `.1` slice of `AU-CONTEXT-R007`; deleting the local finance-math modules and the fallback branch is the separate, unlanded `.2` child.
- [x] **AU-CONTEXT-R008.1:** Typed `AnalysisSnapshot`/`StrategyScorecard` model plus abstention/refusal tests (producer, this repo). IMPLEMENTED at merged head: `agent_utilities/domains/finance/analysis_snapshot.py` + `tests/unit/finance/test_analysis_snapshot_recommendation.py`.
- [x] **AU-CONTEXT-R008.2:** Wire `recommend()` to the real holdings/watchlist entry point, replacing placeholder recommendation output (this repo; depends on R008.1). IMPLEMENTED: `agent_utilities/domains/finance/recommendation_entrypoint.py` (`HoldingsRegistry`, `MarketObservationSource`, `recommend_for_holding`, `register_recommend_tool`) + `tests/unit/finance/test_recommend_entrypoint.py`.
- [x] **AU-CONTEXT-R008.3:** Persist per-strategy `StrategyScorecard`s across calls for comparison (this repo; depends on R008.1/.2). IMPLEMENTED: `ScorecardStore` in `agent_utilities/domains/finance/recommendation_entrypoint.py` keeps one scorecard instance per strategy version across separate `recommend_for_holding` calls, queryable via `store.get(strategy_version)`; covered by `tests/unit/finance/test_recommend_entrypoint.py`.

## Decomposition children (tracked)

- [ ] **AU-CONTEXT-R007.2:** The local finance-math modules and the fallback branch are deleted after cutover.
  BLOCKED on owner `epistemic-graph`: this row's own text gates deletion on "epistemic-graph's finance core has documented, per-method parity evidence for every calculation the pinned module set names." Checked 2026-10-09: `epistemic-graph/specs/finance-primitives/requirements.md` (`EG-FINANCE-PRIMITIVES-R001`, `.../R016`) and its `tasks.md` (`F-00`, `F-15`) still show that Pine/Rust kernel-parity work as `[ ]` TODO, so the documented per-method parity evidence this row requires does not exist yet. Deleting `agent_utilities/domains/finance/**`/`engine_finance.py` now would precede the evidence the row itself requires; re-audit once epistemic-graph lands and documents that parity.
