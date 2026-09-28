# Test contract

| Area | Positive proof | Negative proof |
|---|---|---|
| Control graph | First work item sees a dedicated control graph; messaging config reaches backend | Missing control graph fails typed; no content graph fallback or unscoped write |
| Dependency/contract | Fresh clone installs declared EG client and pins generated digest | Missing sibling checkout does not fail; incompatible digest refuses runtime |
| Baseline/liveness | Named regressions and frozen-time deferral tests pass | Expired unresolved deferral is visible and cannot silently renew |
| Privacy/docs | Exact allowed automation authors pass; public Pages links and moved markers resolve | Credential, internal hostname and near-match author are rejected; unrelated docs deployment failure does not block code PR |
| Cleanup | No live entry point imports legacy publisher or retired modules | Static gate catches a reintroduced legacy route or duplicate authority |
| Pruning | Candidate diff/commits are reachable and clean before deletion | Dirty or unique branch state prevents pruning |
| AI benchmark | Tenant-scoped typed query and model-serving baseline is measured reproducibly | Query shortcut bypassing purpose scope or second data authority is rejected |

Run `python3 scripts/uv_workspace.py doctor`, focused Pytest, full `python3 scripts/uv_workspace.py run --all-extras pytest -q`, `python3 scripts/check_tracked_privacy.py`, `python3 scripts/check_version_consistency.py`, and `agent-utilities lane lease --resource precommit-all-files --operation gate -- python3 scripts/safe_precommit_all_files.py`. PR tests use public deterministic fixtures; live certification names its independently provisioned prerequisites. Record CCCC, jscpd, dupehound and KISS results on the exact commit.
