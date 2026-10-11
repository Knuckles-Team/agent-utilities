# AU-RETRIEVAL — Test contract

| ID | Requirements | Setup and action | Expected result |
|---|---|---|---|
| RETRIEVAL-1 | FR-1 | Fresh checkout loads checked-in synthetic corpus twice. | Identical manifest/hash, query split and gold labels; no external network required. |
| RETRIEVAL-2 | FR-1 | Alter one document or move a section without updating manifest. | Digest or gold-span validation fails before scoring. |
| RETRIEVAL-3 | FR-2 | Query each baseline with same actor, snapshot, candidates and budget. | Query-level ranks, Recall@1/3, nDCG, latency and update cost emitted with explicit method labels. |
| RETRIEVAL-4 | FR-2 | Make served EG unavailable while offline lexical remains usable. | Served result is UNAVAILABLE, never reported as a hybrid pass or copied from lexical score. |
| RETRIEVAL-5 | FR-3 | Use moved, stale, wrong-version and hidden evidence spans. | Citation tracker reports invalid/stale/denied; no unauthorized span reaches answer. |
| RETRIEVAL-6 | FR-4 | Evaluate held-out data with positive, no-gain and regression fixtures. | Only predeclared gain with confidence and no guardrail regression permits adoption recommendation. |
| RETRIEVAL-7 | FR-4 | Request an untrained selector or unavailable external dataset. | Explicit NOT_EVALUATED, no trained-model or production gain claim. |

Store run manifests, exact merged SHA, fixture digest and command output as public receipts. Focused tests and the repository CCCC, jscpd, Dupehound and full Python gates remain pending until implementation.
| RETRIEVAL-8 | FR-5 | Submit a studied technique with no paired benchmark result. | Admission refused; the missing result is named. |
| RETRIEVAL-9 | R004 | Run the served `executable_rag` action with a scripted `insufficient` verdict, under `strict`, then under `lenient`. | `strict` abstains with the evidence gathered. `lenient` returns the extractive answer. The trace shows each round. |
| RETRIEVAL-10 | R005 | Script a `stop` verdict, then a `continue` verdict, for the iterative loop. | The first ends the loop early. The second keeps it running. The trace records the received reason and margin. |
| RETRIEVAL-11 | R006 | Script a choose verdict, then an abstain verdict, at the served answer step. | The first returns the chosen candidate. The second abstains with both candidates and their classes. |

## Producer contract coverage

AU-RETRIEVAL-R004, R005 and R006 consume EG-FEDERATED-QUERY-R081, R082 and R083 respectively.

Harness tests assert receipt and application of these contracts. They do not prove producer decisions, storage or admission.
