# Test contract

| Level | Positive proof | Negative proof |
|---|---|---|
| Unit | Generated EG request/result is converted once at AU API boundary; model adapter receives immutable scoped context | Unknown contract digest, missing graph/tenant, malformed result and unsupported enum fail before execution |
| Wiring | Real graph-os route invokes `agent_utilities.api` and EG/SDK owner method once; all old operations are mapped or approved drops | A forbidden AU internal import, duplicate console script or orphaned route fails the boundary check |
| Contract | Pinned generated client and served method agree on schema digest, error code, retryability and receipt | An older incompatible generation and changed idempotency payload are rejected with no side effect |
| Migration | Valid tenant-bound usage/memory record is replayed once and remains visible after process restart | Spoofed tenant, missing source attestation, corrupt row and cross-tenant read remain quarantined |
| Live path | Public endpoint returns scoped result from the new owner and leaves a durable trace/receipt | Denied principal, expired token, timeout and owner outage do not invoke fallback authority |

Run focused tests for each cut, then `python3 scripts/uv_workspace.py run --all-extras pytest -q`, `python3 scripts/check_current_only_contract.py`, `python3 scripts/check_tracked_privacy.py`, `python3 scripts/check_version_consistency.py`, and `agent-utilities lane lease --resource precommit-all-files --operation gate -- python3 scripts/safe_precommit_all_files.py`. Contract tests that need EG use a disposable public container/fixture with a declared image and pinned contract version; pure static gates run on a fresh checkout without credentials. Capture command, exit code, exact commit, date and artifact URL in `evidence.md`; tests written or passed on an unmerged branch are not landing evidence.

Quality acceptance includes zero newly duplicated code under jscpd and dupehound, no unexplained CCCC complexity regression, KISS review of every added layer, type/lint clean, and no bypassed normal hook. Scanner findings must be compared against the exact base and final head rather than waived because the repository already has debt.

## AU-BOUNDARY-R049 governed write-back

| Proof | Test |
|---|---|
| A dry run routes through the SDK, returns change-set evidence and writes no durable ledger | `test_dry_run_routes_through_sdk_without_a_durable_ledger` |
| A live write records the change set, the audit reservation and one `applied` receipt | `test_live_write_records_change_set_audit_and_applied_receipt` |
| A live write without the enable flag is refused before the SDK and the sink | `test_live_write_stays_refused_without_the_enable_flag` |
| A sink failure records `outcome_uncertain` and returns the original error | `test_sink_failure_is_recorded_as_uncertain_and_reported` |
| The SDK refuses a denied authorization before the sink runs | `test_sdk_refuses_an_unauthorized_live_call_before_the_sink` |
| The SDK refuses a stale source version before the sink runs | `test_sdk_refuses_a_stale_source_version_before_the_sink` |
| An approved replay uses the `proposal_approval` mode | `test_approved_replay_uses_proposal_approval_mode` |
| The change set is canonical and excludes private keys and clients | `test_change_set_is_canonical_and_drops_private_ops` |

All tests live in `tests/unit/knowledge_graph/enrichment/test_writeback_governed.py`. The existing write-back, approval, preflight, bitemporal and ETL suites run unchanged against the new path.
