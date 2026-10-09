# Tasks — AU-QUAL-01

| Task | Requirement | Done when |
|---|---|---|
| Q1 | 01/07 | Locked setup and fresh clone workflow install all tools and test dependencies without private network. |
| Q2 | 02 | `tiny` gateway/identity status and bootstrap test patch targets are corrected; served deny/success tests pass. |
| Q3 | 03 | AU validation consumes engine `GraphSchema`; old shapes authority removed; drift contracts pass. |
| Q4 | 04 | Every named failure group has a causal fix and deterministic passing test. |
| Q5 | 05 | All test directories pass mypy; exclusion removed; no compensating blanket suppressions. |
| Q6 | 06/07 | Liveness checks execute on PR and daily schedule; docs deploy blocker removed; Pages publication independently observable. |
| Q7 | all | CCCC, jscpd, Dupehound, privacy, liveness, full tests and hosted checks pass on exact commit; evidence linked. |
| Q8 | AU-QUAL-R006 | [x] Run a privacy check across the complete working tree and every commit message as a required gate before code reaches a public remote; fails the build on a disallowed pattern. Commit-message scan (`_commit_message_violations`) is bounded to commits not yet reachable from `origin/main`; a pre-existing, unrelated 2-finding tracked-tree failure (`test_full_corpus_scan_is_clean`) remains open, tracked separately under AU-QUAL-R004's remaining failure groups. |
| Q9 | AU-QUAL-R008 | `scripts/liveness_reconciler.py` rescues exact effective service-registry targets; `tests/gates/test_liveness_reconciler.py` covers last-row-wins, nonliteral and dispatch parity. |

Mark `LANDED` by default-branch commit and `ACCEPTED` only when exact-run evidence satisfies the row. An interim source fix may leave another row `BUILDING`.
