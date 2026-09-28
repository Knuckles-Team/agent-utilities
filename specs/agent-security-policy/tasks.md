# Tasks — AU-SEC-01

| Task | Requirement | Deliverable and completion proof |
|---|---|---|
| T1 | SEC-01/02 | Validate actor/session projection and local-only authority; gateway and profile positive/negative tests. |
| T2 | SEC-03 | Preserve dedicated control view and typed unavailable error; injected outage test. |
| T3 | SEC-04 | Generate classed AU allowlist from canonical registry; exact parity and stale-generation checks. |
| T4 | SEC-05/06 | Wire request/approve/revoke to the real public and tool paths with independent approver, expiry, idempotency, and final engine check. |
| T5 | SEC-07 | Wire bounded profile decisions to existing orchestration and approval flow; replay and revision conflict tests. |
| T6 | SEC-08 | Connect invalidation events to caches; test event loss and max TTL. |
| T7 | SEC-09 | Connect drift result to orchestration and SDK checkpoint contract; quarantine/recovery tests. |
| T8 | all | Run test matrix, CCCC/jscpd/Dupehound and privacy checks; capture exact commit, hosted result, and cross-owner acceptance. |

Tasks may be split across PRs. Mark a row **LANDED** only after its code reaches default branch; mark **ACCEPTED** only after required evidence and owner contract checks are linked.
