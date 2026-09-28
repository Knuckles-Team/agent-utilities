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
