# AU-SCM — Implementation tasks

Governing spec: [spec.md](spec.md). Design: [plan.md](plan.md). Rows: [requirements.md](requirements.md).

- [ ] Inventory the harness package, the run loop entry and the host ports; record the reuse map.
- [ ] Add `ChangesetSubmitter` to `host_ports.py` beside the run ports.
- [ ] Implement the edit step over the virtual workspace through `apply_edits`.
- [ ] Implement the fast-gate step in the sandbox through `PreflightGate`.
- [ ] Implement the submit step through the `ChangesetSubmitter` port.
- [ ] Implement the rebase-and-retry step on `SOURCE_CONFLICT{paths}` with the retry bound.
- [ ] Add the served proof and the focused unit tests; run the gates.
- [ ] Update `status.json` only after the required gates pass on the merged revision.
