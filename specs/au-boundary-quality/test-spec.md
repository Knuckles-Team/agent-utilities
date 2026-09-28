# Test contract — AU-QUAL-01

| Test | Requirement | Setup and action | Expected result |
|---|---|---|---|
| FRESH-1 | 01/07 | Clone only AU on ephemeral hosted runner; run documented bootstrap/doctor and unit gate | No undeclared sibling checkout, private endpoint or credential needed. |
| GATE-1 | 02 | Serve `tiny` gateway; issue permitted and forbidden graph operations | Correct success and denied HTTP/MCP status; no false 200. |
| GATE-2 | 02 | Remove an internal bootstrap helper in fixture while public seam stays | Isolation test targets maintained public seam; no deleted function patch. |
| SCHEMA-1 | 03 | Supply composed engine schema and connector payload | AU passes digest and typed result; no local shape authority. |
| SCHEMA-2 | 03 | Supply incompatible or unavailable schema | Typed drift/unavailable; checkpoint and operation remain unchanged. |
| BASE-1 | 04 | Run placement, trace, claim, ingest, observability, widget and benchmark groups in CI fixtures | All pass deterministically; negative assertions retained. |
| TYPE-1 | 05 | Run mypy over source and tests with no test exclusion | Zero errors on exact revision; no added Any casts or ignores. |
| LIVE-1 | 06 | Freeze time before/after deferral review date and run liveness check | Due item reported with owner/date; expired item fails hosted check. |
| CLOUD-1 | 07 | PR from fork with no secrets and no services outside job | Required CI reaches tests; setup failure names dependency; no live-environment demand. |
| DOC-1 | 07 | Change code with no docs change; run CI; separately simulate Pages publish failure | Code result follows code tests; publication job reports its own status. |

Run configured Ruff/mypy, `scripts/check_complexity_staged.py` (CCCC), `scripts/check_dupehound.py`, configured jscpd differential hook, relevant privacy checks and full `pytest` after focused tests. Respect current configured thresholds; do not claim success for a missing binary. Release-only live tests require an explicitly provisioned environment and are recorded separately. Each requirement needs an exact commit and CI run URL before acceptance.
