# Evidence and delivery state — AU-QUAL-01

**Legend:** `WAITING` is specified only; `BUILDING` is source or tests in progress; `LANDED` is an exact default-branch commit; `ACCEPTED` adds passing exact-revision tests, quality and hosted contract evidence; `BLOCKED` names an external dependency. A spec document is never proof of implementation.

| Requirement | Delivery | Acceptance | Proof still required |
|---|---|---|---|
| QUAL-01 | BUILDING | pending | Fresh clone dependency and fixture CI run. |
| QUAL-02 | BUILDING | pending | Exact served allow/deny and bootstrap isolation run. |
| QUAL-03 | WAITING | pending | Composed schema contract and removed AU shape authority. |
| QUAL-04 | BUILDING | pending | Full named failure inventory and deterministic results. |
| QUAL-05 | WAITING | pending | Complete mypy coverage including tests. |
| QUAL-06 | BUILDING | pending | Clock-frozen liveness and scheduled workflow run. |
| QUAL-07 | WAITING | pending | Forked PR check without private environment; gate diff. |

Replace these evidence requirements with exact default branch commit and hosted run links as work lands. No historical status assertion is inferred from the presence of a code file.
