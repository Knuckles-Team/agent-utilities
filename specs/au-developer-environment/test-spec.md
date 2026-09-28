# Tests

| Case | Expected success | Expected refusal |
|---|---|---|
| Fresh clone | Unit/spec profile installs from public sources and runs without ambient services | Missing toolchain names the dependency; no hidden private URL is contacted |
| Skill refresh | Source hash and installed hash match after install; second run is no-op | Renamed/deleted provider is flagged; stale installed skill cannot claim current |
| Composed fixture | Synthetic tenant, GraphOS route, AU API and EG client exchange typed receipt | Wrong digest, denied scope and owner outage fail closed |
| Cleanup | Disposable containers and temp state are removed after pass/fail | A failed setup never leaves a running shared service or leaked secret |
| Gate separation | Behavioral PR tests run in CI; release profile records explicit not-run without credentials | A skipped required test cannot turn CI green; live environment absence cannot block unrelated docs/source change |

Use a clean public checkout in CI and run `python3 scripts/uv_workspace.py doctor`, focused tests, `python3 scripts/uv_workspace.py run --all-extras pytest -q`, `python3 scripts/check_tracked_privacy.py`, `python3 scripts/check_version_consistency.py`, and `agent-utilities lane lease --resource precommit-all-files --operation gate -- python3 scripts/safe_precommit_all_files.py`. Capture exact command and commit. Compare CCCC, jscpd, dupehound and KISS findings with the base revision.
