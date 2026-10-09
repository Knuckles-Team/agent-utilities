# Tasks — AU-QUAL-01

| Task | Requirement | Done when |
|---|---|---|
| Q1 | 01/07 | Locked setup and fresh clone workflow install all tools and test dependencies without private network. |
| Q2 | 02 | `tiny` gateway/identity status and bootstrap test patch targets are corrected; served deny/success tests pass. |
| Q3 | 03 | [x] AU validation consumes engine `GraphSchema`; old shapes authority removed; drift contracts pass. |
| Q4 | 04 | Every named failure group has a causal fix and deterministic passing test. |
| Q5 | 05 | All test directories pass mypy; exclusion removed; no compensating blanket suppressions. |
| Q6 | 06/07 | Liveness checks execute on PR and daily schedule; docs deploy blocker removed; Pages publication independently observable. |
| Q7 | all | CCCC, jscpd, Dupehound, privacy, liveness, full tests and hosted checks pass on exact commit; evidence linked. |
| Q8 | AU-QUAL-R006 | Run a privacy check across the complete working tree and every commit message as a required gate before code reaches a public remote; fails the build on a disallowed pattern. |
| Q9 | AU-QUAL-R008 | `scripts/liveness_reconciler.py` rescues exact effective service-registry targets; `tests/gates/test_liveness_reconciler.py` covers last-row-wins, nonliteral and dispatch parity. |

Mark `LANDED` by default-branch commit and `ACCEPTED` only when exact-run evidence satisfies the row. An interim source fix may leave another row `BUILDING`.
