# AU-HARNESS-CYCLE — Implementation tasks

Governing spec: [spec.md](spec.md). Design: [plan.md](plan.md). Rows: [requirements.md](requirements.md).

- [ ] Inventory the harness package, the run loop entry and the host ports; record the reuse map.
- [ ] Add `HarnessCandidate` and `CycleOutcome` with the closed refusal codes.
- [ ] Implement the propose step against the EG-ORCH-VOCAB-R802.3 query.
- [ ] Implement the probe step through `execute_graph` with the shared conditions.
- [ ] Implement the compose step as one served `Decide` call.
- [ ] Implement the persist step through the generated-client envelopes and `apply_edits`.
- [ ] Implement the reflect step and the four typed refusals.
- [ ] Register `HarnessCycleProvider` in `host_ports.py` beside `RunDecisionProvider`.
- [ ] Add the served double test and the focused unit tests; run the gates.
- [ ] Update `status.json` only after the required gates pass on the merged revision.
