# Tasks: harness evolution

- [ ] HE-01: Capability probe, immutable policy identity and independent default-off controls.
- [ ] HE-02: Terminal capture schema, masked arrays, frozen sampler probabilities, Blob CAS references and typed rejection.
- [ ] HE-03: Extend `SubstrateTrainer` with digest-bound external jobs and optional KLPO; test record-only and dispatch modes.
- [ ] HE-06: Resource lease, independent evaluation, canary/CAS promotion and rollback receipts.
- [ ] HE-04: Atomic Gap/WorkItem upsert, derived offers, EG decision/commit, fenced claim and cooldown.
- [ ] HE-05: Proposal/validation/materialization client cutover; delete direct Git, local publication and Gap Cypher paths.
- [ ] HE-07: Keep generative/autograd path deferred pending a separate approved evidence gate.
- [ ] HE-08: Define the feature schema and slate-crediting method, fit and evaluate the work-market offer scoring head against a synthetic width-quality gold set, and promote it only through the governed promotion gate. Closes AU-HARNESS-R006. Feature schema, slate-crediting (`credit_topology_outcome`), the gold set and the equal-budget benchmark landed under AU-CONTROL-R020 (`agent_utilities/decide/consumers/topology_learning.py`, `agent_utilities/decide/topology/benchmark.py`). The remaining gap -- the scorer's `plan_expected_cost`/`claim_cost_drift` had zero callers on main -- is closed: `agent_utilities/knowledge_graph/research/work_market.py` now prices a topology-carrying Gap's offer from the plan's own declared cost and refuses+re-prices a claim whose committed plan drifted past tolerance (`tests/unit/knowledge_graph/test_work_market.py::test_a_topology_plans_own_cost_changes_the_priced_offer`, `::test_claim_refused_and_reprised_on_committed_plan_cost_drift`). Not yet closed: no production path writes a `topology_plan` evidence item onto a Gap (today only a test fixture does), and the governed promotion-gate run for the scorer itself is unverified in this pass.
- [ ] HE-09: Typed `HarnessPort`, request and outcome models, typed refusal and the documented extension point. Closes AU-HARNESS-R007.
- [ ] HE-10: `NativeHarness` over `run_agent` with envelope mapping and timeout. Closes AU-HARNESS-R008.
- [ ] HE-11: `ClaudeCodeHarness` with pinned flags, worktree, timeout, environment allowlist, typed outcome and diff stat. Closes AU-HARNESS-R009.
- [ ] HE-12: `AgentSpec.harness`, the registry, parallel-engine dispatch and L5 recording. Closes AU-HARNESS-R010.
- [ ] HE-13: Run the `claude-code` harness once against served graph-os and record the receipt in `evidence.md`.
- [ ] HE-14: Propose a `harness` field on the EG `AgentGraphNode` contract so published L3 graphs carry the selection.
- [ ] HE-15: Update the `agent_utilities/layers/` row in the boundary coverage inventory for the new harness modules.
- [ ] HE-16: Schedule run-outcome prompt evolution through the existing optimization sweep and record propose-only `PromptVersion` candidates with trace provenance. Closes AU-HARNESS-R011.
- [ ] Run all fixtures and quality checks in `test-spec.md`; record exact merged-head evidence in `evidence.md`.

Checkboxes close only after acceptance evidence. A design, branch, source file or local green test alone is not a completed deliverable.
