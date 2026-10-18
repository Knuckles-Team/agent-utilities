# Test contract

| Level | Positive proof | Negative proof |
|---|---|---|
| Unit | Generated EG request/result is converted once at AU API boundary. Model adapter receives immutable scoped context | Unknown contract digest, missing graph/tenant, malformed result and unsupported enum fail before execution |
| Wiring | Real arbora route invokes `agent_utilities.api` and EG/SDK owner method once; all old operations are mapped or approved drops | A forbidden AU internal import, duplicate console script or orphaned route fails the boundary check |
| Contract | Pinned generated client and served method agree on schema digest, error code, retryability and receipt | An older incompatible generation and changed idempotency payload are rejected with no side effect |
| Migration | Valid tenant-bound usage/memory record is replayed once and remains visible after process restart | Spoofed tenant, missing source attestation, corrupt row and cross-tenant read remain quarantined |
| Live path | Public endpoint returns scoped result from the new owner and leaves a durable trace/receipt | Denied principal, expired token, timeout and owner outage do not invoke fallback authority |

Run focused tests for each cut, then `python3 scripts/uv_workspace.py run --all-extras pytest -q`, `python3 scripts/check_current_only_contract.py`, `python3 scripts/check_tracked_privacy.py`, `python3 scripts/check_version_consistency.py`, and `agent-utilities lane lease --resource precommit-all-files --operation gate -- python3 scripts/safe_precommit_all_files.py`. Contract tests that need EG use a disposable public container/fixture with a declared image and pinned contract version. Pure static gates run on a fresh checkout without credentials. Capture command, exit code, exact commit, date and artifact URL in `evidence.md`. Tests written or passed on an unmerged branch are not landing evidence.

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

## AU-BOUNDARY-R043 agent run loop

| Row | Proof | Test |
|---|---|---|
| `AU-BOUNDARY-R043.1` | The host plan drives the run. Steps share one history. Budgets and stop rules stop the run. | `tests/test_run_loop.py` |
| `AU-BOUNDARY-R043.2` | The fallback plan reads only the request and configuration. Missing providers fail closed. | `tests/test_run_loop.py` |
| `AU-BOUNDARY-R043.3` | The recursive loop caps slices and depth. It classifies each failure. | `tests/test_rlm.py` |

The tests use fakes for the host ports and pydantic-ai function models. They run under `pytest -n auto` in random order.

## AU-BOUNDARY-R043.4 scaffold migration acceptance

R043.4.1–R043.4.5 retain the historical R014.1–R014.5 acceptance. Use deterministic
model fixtures through the existing run loop: replace ordered items twice,
refuse malformed input atomically, gate the tool on the received decision,
inspect every outgoing request, and assert no scaffold content in stored history
or tool records. Check isolation across two runs and the absence of planner
imports or selection fields. The bound test files below provide the scaffold implementation evidence.

Snapshot follow-up must prove revision zero, repeated-content increments, read
stability, atomic item/revision capture, UTC expiry and lease capping, caller
isolation by tenant and principal, and identical unknown/foreign unavailability.
Test disabled, ended and restarted runs, absent ports, malformed or regressing
snapshots, expired leases, denied refresh and run switches. A served test must
reach the same state used by injection, without a model call or history rebuild.
The SDK, EG and Arbora boundaries remain separate acceptance obligations.

R043.4.1 is exercised by `tests/test_harness_plan.py`: full ordered replacement,
empty reset, stable old views, immutable items, two-run isolation, strict field
validation and invalid-later-item atomicity. The loop/tool/injection evidence for
R043.4.2–R043.4.5 is recorded below.

R043.4.2–R043.4.5 are exercised by `tests/test_plan_runtime.py` through
`graph.executor.run`: received on/off gating, two updates and a malformed update,
per-request injection across execution steps, separate-run isolation, retained
history inspection, budget-exit cleanup and the no-selection tool/import surface.
This evidence covers the scaffold loop, not a snapshot reader or served inspector.

## AU-BOUNDARY-R043.5.1 atomic revision acceptance

Read revision zero with empty items. Accept identical replacements and empty
resets with one increment each. Reject invalid items without an increment.
Preserve old immutable views. Exercise competing writers and interleaved reads
with bounded barriers, no sleeps or network, and prove every observed revision
corresponds to its complete item tuple. This proves no caller or lease authority.
