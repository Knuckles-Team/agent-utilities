# AU-HARNESS-CYCLE — Implementation tasks

Governing spec: [spec.md](spec.md). Design: [plan.md](plan.md). Rows: [requirements.md](requirements.md).

- [ ] Inventory the harness package, the run loop entry and the host ports; record the reuse map.
- [ ] Add `HarnessCandidate` and `CycleOutcome` with the closed refusal codes.
- [ ] Implement the propose step against the EG-ORCH-VOCAB-R802.3 query.
- [ ] Implement host-admitted, materialized probes under the shared actual conditions.
- [ ] Implement the compose step as one served `Decide` call.
- [ ] Implement the persist step through the generated-client envelopes and `apply_edits`.
- [ ] Implement the reflect step and the typed refusals.
- [ ] Register `HarnessCycleProvider` in `host_ports.py` beside `RunDecisionProvider`.
- [ ] AU-HARNESS-CYCLE-R001.10: Replace the fake persist port with the released generated EG client.
- [ ] AU-HARNESS-CYCLE-R001.11: Prove a host-registered provider drives one tick on the served path.
- [ ] Add the served double test and the focused unit tests; run the gates.
- [ ] Update `status.json` only after the required gates pass on the merged revision.

- [ ] AU-HARNESS-CYCLE-R001.12: Refuse unverified contrast measurements before Decide or persistence.
- [ ] Bind owner-authenticated baseline and candidate measurements before enabling contrast decisions.

- [ ] AU-HARNESS-CYCLE-R001.13: Refuse probe execution without a host admission executor.
- [ ] Implement and verify host-owned admission and component materialization before claiming actual probe conditions.
