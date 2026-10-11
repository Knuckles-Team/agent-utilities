# Test contract — AU-QUAL-01

| Test | Requirement | Setup and action | Expected result |
|---|---|---|---|
| FRESH-1 | 01/07 | Clone only AU on ephemeral hosted runner. Run documented bootstrap/doctor and unit gate | No undeclared sibling checkout, private endpoint or credential needed. |
| GATE-1 | 02 | Serve `tiny` gateway; issue permitted and forbidden graph operations | Correct success and denied HTTP/MCP status; no false 200. |
| GATE-2 | 02 | Remove an internal bootstrap helper in fixture while public seam stays | Isolation test targets maintained public seam; no deleted function patch. |
| SCHEMA-1 | 03 | Supply composed engine schema and connector payload | AU passes digest and typed result; no local shape authority. |
| SCHEMA-2 | 03 | Supply incompatible or unavailable schema | Typed drift/unavailable; checkpoint and operation remain unchanged. |
| BASE-1 | 04 | Run placement, trace, claim, ingest, observability, widget and benchmark groups in CI fixtures | All pass deterministically; negative assertions retained. |
| TYPE-1 | 05 | Run mypy over source and tests with no test exclusion | Zero errors on exact revision. No added Any casts or ignores. |
| LIVE-1 | 06 | Freeze time before/after deferral review date and run liveness check | Due item reported with owner/date. Expired item fails hosted check. |
| LIVE-2 | AU-QUAL-R008 | Reconcile dead definitions against fixture and real `_SERVICE_DEFINITIONS` registries | Only last-row-wins targets are rescued. A nonliteral registry rescues nothing. Static targets equal real dispatch. |
| CLOUD-1 | 07 | PR from fork with no secrets and no services outside job | Required CI reaches tests. Setup failure names dependency. No live-environment demand. |
| DOC-1 | 07 | Change code with no docs change. Run CI. Separately simulate Pages publish failure | Code result follows code tests. Publication job reports its own status. |

Run configured Ruff/mypy, `scripts/check_complexity_staged.py` (CCCC), `scripts/check_dupehound.py`, configured jscpd differential hook, relevant privacy checks and full `pytest` after focused tests. Respect current configured thresholds; do not claim success for a missing binary. Release-only live tests require an explicitly provisioned environment and are recorded separately. Each requirement needs an exact commit and CI run URL before acceptance.

- **AU-QUAL-R009.1:** `tests/unit/harness/test_red_team_evidence.py::test_probe_result_verdict_fields` preserves all three engine verdicts and separates execution status. No local verdict includes `abstained` or `defended`.
- **AU-QUAL-R009.2:** `tests/unit/harness/test_red_team_evidence.py::test_no_scorer_abstains`.
- **AU-QUAL-R009.3:** `tests/unit/harness/test_red_team_evidence.py::test_assess_writes_trace_evidence`.
- **AU-QUAL-R009.4:** `tests/unit/harness/test_red_team_evidence.py::test_all_abstained_is_not_pass`.
- **AU-QUAL-R009.5:** `tests/unit/harness/test_red_team_evidence.py::test_evidence_reproduces`.
- **AU-QUAL-R009.6:** `tests/integration/test_red_team_assess_served.py::test_served_assess_records_evidence` covers every engine verdict, missing verdicts and target errors. Errors carry execution reasons without fabricated verdicts. A report without conclusive verdicts has no defense rate.

## Producer contract coverage

AU-QUAL-R009 consumes EG-DECISION-ENGINE-R138. Its served security-evaluation operation requires a GRAPHOS-OPS producer requirement before consumer acceptance.

Harness tests assert receipt and application of these contracts. They do not prove producer decisions, storage or admission.
