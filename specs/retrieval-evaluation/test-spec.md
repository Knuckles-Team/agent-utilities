# EH-652 — Test contract

| ID | Requirements | Setup and action | Expected result |
|---|---|---|---|
| E652-1 | FR-1 | Fresh checkout loads checked-in synthetic corpus twice. | Identical manifest/hash, query split and gold labels; no external network required. |
| E652-2 | FR-1 | Alter one document or move a section without updating manifest. | Digest or gold-span validation fails before scoring. |
| E652-3 | FR-2 | Query each baseline with same actor, snapshot, candidates and budget. | Query-level ranks, Recall@1/3, nDCG, latency and update cost emitted with explicit method labels. |
| E652-4 | FR-2 | Make served EG unavailable while offline lexical remains usable. | Served result is UNAVAILABLE, never reported as a hybrid pass or copied from lexical score. |
| E652-5 | FR-3 | Use moved, stale, wrong-version and hidden evidence spans. | Citation tracker reports invalid/stale/denied; no unauthorized span reaches answer. |
| E652-6 | FR-4 | Evaluate held-out data with positive, no-gain and regression fixtures. | Only predeclared gain with confidence and no guardrail regression permits adoption recommendation. |
| E652-7 | FR-4 | Request an untrained selector or unavailable external dataset. | Explicit NOT_EVALUATED, no trained-model or production gain claim. |

Store run manifests, exact merged SHA, fixture digest and command output as public receipts. Focused tests and the repository CCCC, jscpd, Dupehound and full Python gates remain pending until implementation.
