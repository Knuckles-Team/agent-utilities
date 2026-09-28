# Evidence and state — AU-SEC-01

## State legend

`WAITING` = specified without verified implementation; `BUILDING` = work in an unmerged branch; `LANDED` = exact source commit on default branch; `ACCEPTED` = landed source plus passing test, contract, quality and hosted evidence on that revision; `BLOCKED` = named dependency prevents the next step. These are separate from specification review. A source line or historic work item is not acceptance evidence.

| Requirement | Delivery | Acceptance | Exact commit / run |
|---|---|---|---|
| SEC-01 | BUILDING | pending | Current `security/request_identity.py` provides projection; audit exact default branch and tests. |
| SEC-02 | BUILDING | pending | Current local mint has 120-second proof; network isolation and renewal need exact evidence. |
| SEC-03 | BUILDING | pending | Current control-view code present; failure test and exact merge evidence needed. |
| SEC-04 | BUILDING | pending | Registry projection and version parity need exact evidence. |
| SEC-05 | WAITING | pending | Cross-owner contract and served tests needed. |
| SEC-06 | BUILDING | pending | Current tool guard preserves hard denials; lease integration not established. |
| SEC-07 | WAITING | pending | Profile decision and approval proof needed. |
| SEC-08 | BUILDING | pending | Cache invalidation helpers present; source event and max TTL proof needed. |
| SEC-09 | BUILDING | pending | Drift metadata appears in AU; SDK checkpoint contract proof needed. |

Spec creation does not change delivery or acceptance. Replace placeholders only with a URL to an exact default-branch commit, CI run, and relevant test output.
