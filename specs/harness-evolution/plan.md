# Implementation plan: harness evolution

1. Pin and golden-test public EG generated contracts. Implement capability attestation and capture-only mode first; all three controls default off.
2. Add immutable terminal capture assembly in the provider/harness path. Commit through typed client; add length, mask, sampler and blob-holder negative fixtures.
3. Extend the injected `SubstrateTrainer` job schema to reference capture digests and optional KLPO. Dispatch to an external trainer through graph-os resource admission. Keep train and promote disabled while measuring capture overhead.
4. Add independent held-out `PolicyEvaluation` and compare-and-swap promotion with rollback. Use new artifacts only; publish measured resource and task outcomes before enabling any nondefault setting.
5. Replace direct Gap writes and Python sorting with atomic `GapUpsert`, `WorkOfferPut`, `Decide`/`DecisionCommit` and native WorkItem claim. Do not stage an AU fallback selector.
6. Replace AU direct Git publishing with `ChangeProposal` → graph-os → repository-manager → validation/materialization receipts. Delete obsolete authority classes and their consumers in the same cutover.
7. Run synthetic, fault and served integration fixtures. Record exact merged-head evidence per requirement and leave incomplete rows open.

Dependencies: real EG Decide and generated clients precede work-market acceptance; repository-manager and graph-os receipt contracts precede Git materialization acceptance. A locally emitted job is not trained, evaluated or promoted.
