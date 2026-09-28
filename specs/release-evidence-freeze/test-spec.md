# RF-001 — Test contract

| ID | Requirement | Setup and action | Expected result |
|---|---|---|---|
| RF1-1 | FR-1 | Generate twice from one clean synthetic checkout. | Canonical source digest identical; declared observation timestamp is separate. |
| RF1-2 | FR-1 | Dirty checkout or modified lock after freeze. | Gate refuses and names changed input. |
| RF1-3 | FR-2 | Bind passing CI scanners, tests and wheel digest to exact commit. | Artifact and each run verify against one source digest and immutable receipt. |
| RF1-4 | FR-2 | Replace artifact, omit scanner, use another SHA or duplicate JSON key. | Qualification fails; no partial green aggregate. |
| RF1-5 | FR-3 | Run source-only PR profile without live services. | Source/contract checks execute; live qualification records NOT_RUN and remains pending. |
| RF1-6 | FR-3 | Present expired/revoked or wrong-artifact runtime snapshot. | Release qualification fails with explicit reason. |
| RF1-7 | FR-4 | Compare matching and mismatched AU-generated-client and EG-published contract digests. | Only exact compatible pair qualifies; no fallback digest is synthesized. |

Run focused source-freeze, release compatibility, wheel and assembly tests from a disposable checkout, then full AU tests and configured CCCC, jscpd and Dupehound. Record exact commit, command, exit, artifact digest and public CI receipt; tests written but not run are not acceptance evidence.
