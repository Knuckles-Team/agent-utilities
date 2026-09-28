# Architecture and quality design — AU-QUAL-01

## Reuse map

`scripts/uv_workspace.py` is the workspace launcher and dependency doctor; adapt it to a fresh clone when sibling packages are needed. `.config/pre-commit.yaml` and `.github/workflows/` define local and hosted gates. `tests/integration/profiles/test_profile_tiny_zero_dep.py` drives the packaged local gateway; `tests/unit/mcp/test_graphos_bootstrap_isolation.py` tests bootstrap isolation. `security/request_identity.py`, `security/http_boundary.py`, and gateway route handlers own status translation. `knowledge_graph/core/engine.py` and the engine client own graph calls. `scripts/check_liveness.py` plus `scripts/liveness_deferred.tsv` own the liveness ratchet. `pyproject.toml` owns mypy and scanner configuration. Extend those seams; avoid a second test harness or status adapter.

## Fresh checkout test architecture

`git clone` → Python/uv bootstrap → locked dependency resolution → disposable in-process/mock or container fixture where a genuine protocol is required → unit/contract tests → source quality. A test declares optional external adapters in package extras. If a transport test needs a service, CI starts it in the job with a health wait and fixed test data, or the test constructs a local fake that implements the exact wire contract. Production endpoints and credentials are never a fallback. A release-only workflow may run a real multi-service deployment and publish its evidence separately.

Gate output distinguishes `PASS`, `FAIL`, and `SETUP ERROR` while keeping CI red for either failure. It prints the missing package/service and setup command. Use job artifacts for reports, avoiding committed generated `/docs` output. GitHub Pages is the publication target for human documentation, generated from the canonical source designated by the repository. A documentation deployment failure does not invalidate a code correctness run.

## Boundary contracts

AU sends connector validation through the engine client with a versioned `GraphSchema` digest and receives a typed validation result. SDK owns connector payload and checkpoint, engine owns composed schema and validation rules, AU owns orchestration and presentation of the outcome. A mismatched digest returns conflict/drift; no AU local shapes pack is consulted for authority. The `tiny` gateway maps denied engine operations to a non-success status and preserves error code; a wrapper may not turn a returned failure payload into HTTP 200.

## Quality design

Mypy expansion is staged by test directory with one baseline report per batch and a final config change that includes all tests. Existing typed fixtures and protocols should replace broad monkeypatch casts. CCCC uses the configured complexity hook (no new cyclomatic >10 or cognitive >15, no regression). Dupehound 0.1.2 uses threshold 0.85 and 40 tokens; jscpd 5.0.16 uses 50 tokens/5 lines in differential mode. KISS means fixing shared status translation and fixture setup once, deleting obsolete patch targets, and refusing per-test workaround copies. Scanner versions and thresholds come from `pyproject.toml` and must be checked against workflow/tool provisioning.

## Migration and risks

Moving a time-sensitive liveness check out of local pre-commit requires a hosted/scheduled job that cannot miss expiry on a no-change day. If GitHub scheduled workflows are unreliable, combine a daily schedule with the normal PR check. Record the time and frozen clock in tests. Keep security and privacy gates in the PR path; replace ambient private-service assumptions with provisioned fixtures. When a test cannot be made hermetic, move it into an explicit live certification stage and state why the remaining contract tests still cover the PR boundary.
