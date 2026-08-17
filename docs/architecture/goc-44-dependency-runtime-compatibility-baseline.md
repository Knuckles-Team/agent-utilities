# GOC-44 dependency/runtime compatibility — baseline revalidation (2026-08-16)

CONCEPT:AU-ECO.mcp.protocol-compat-bridge

Lane: `plans/graph-os-completion-program/lanes/GOC-44-dependency-runtime-compatibility.md`
(`OWNER-DEPENDENCY-GREEN`). This is the GOC-44-W01 output: a timestamped revalidation
of the lane's 13 imported AU–EG IDs against `main` as it stood on 2026-08-16
(`agent-utilities` HEAD `b60e5b439`, worktree
`/home/genius/.local/state/repository-worktrees/agent-utilities/goc-44`,
branch `goc/goc-44-dependency-runtime-compat`). Every row below records the command
or artifact used to revalidate — no disposition here is inferred from the imported
ledger text alone. Environment: interpreter `cpython-3.14.4-linux-x86_64-gnu`
(uv-managed, `python-build-standalone`), host `RW710`
(`Linux 7.0.0-28-generic`), venv used for live checks:
`/home/genius/.local/state/repository-worktrees/agent-utilities/goc-87/.venv`
(this worktree, `goc-44`, has no installed venv of its own — see "What is
unverified" below).

**Denominator for every count in this document:** the 13 IDs imported into
GOC-44's queue (`plans/graph-os-completion-program/archive/AU-EG-DEFERRED-IMPORT-LEDGER.md`
§"GOC-44 — `OWNER-DEPENDENCY-GREEN`"), unless a different denominator is stated inline.

## Disposition summary

| Disposition | Count | IDs |
|---|---:|---|
| STALE-PREMISE — already resolved by prior work, evidence attached | 6 | D-HX-1, D-2.1b-2, D-2.1b-3, D-VS-1, D-VS-2, D-W5AGB-1 |
| STALE-PREMISE — self-described transient state, not reproducible today | 1 | D-W2P-7 |
| CONFIRMED-LIVE — already mitigated adequately (fail-closed + tested); no unsafe rush-fix applied | 2 | D-CIP-17, D-CIP-3 |
| CONFIRMED-LIVE — real gap, root cause outside this lane's authority, coordinate with named owner | 2 | D-W6TG-1 (→ GOC-42), D-MQR-4 (partially; → operator) |
| CONFIRMED-LIVE — real gap in this lane's own scope, too large/risky to safely land in one pass; filed as owned blocker | 1 | D-VI-2 |
| DECISION-REQUIRED — operator consolidation call, not a code defect | 1 | D-HX-2 |

No code fix was made in this pass. Every item that looked fixable turned out to
already be fixed (dated 2026-07-30 through 2026-08-14, all before this lane's
2026-08-16 revalidation), and every item that is still genuinely open either has an
adequate existing mitigation or needs authority/effort this lane cannot safely
supply in one pass without full-gate access (which this lane does not run itself —
see the program's standing rule 3). This is treated as a legitimate outcome, not a
gap: a well-evidenced revalidation is the deliverable this pass, not a forced patch.

## Per-ID revalidation

### D-HX-1 — STALE-PREMISE (resolved)

> "httpx.AsyncClient handed to mcp.client.streamable_http.streamable_http_client
> (http_client=...) where httpx2.AsyncClient is typed/expected"

The type mismatch is real and current: `mcp==2.0.0`'s
`streamable_http_client(*, http_client: httpx2.AsyncClient | None = None, ...)`
is typed against `httpx2`, and `agent_utilities/mcp/multiplexer.py`'s non-SSE
remote-child branch (around line 2055-2057) builds `http_client` via
`agent_utilities.core.http_client.create_async_http_client`, which is `httpx`-based,
then hands it to `streamable_http_client(url, http_client=http_client)`.

**But the runtime claim — that this crossing is broken — does not reproduce.**
Two independent proofs:

1. A dedicated live regression test already exists and already passes:
   `tests/integration/mcp/test_multiplexer_remote_transport_live.py::test_hardened_client_round_trips_sdk_v2_streamable_http`,
   landed **2026-07-30** (`git log`, commit `5be9443420102b294146fdeb7cebb9c6cfc70151`,
   `fix(mcp): repair the MCP SDK v2 breakage the fastmcp-4 default exposed`) — i.e.
   before the AU-EG import even recorded this as `OPEN`. It calls
   `create_async_http_client(..., pin_egress=True, ...)` — the exact hardened
   construction multiplexer.py uses — and hands it to `streamable_http_client`,
   completing `initialize` + `list_tools` + `call_tool` against a live fastmcp-4
   server. Re-run today: **PASSED** (`goc-87/.venv`, `pytest
   tests/integration/mcp/test_multiplexer_remote_transport_live.py -v` →
   `1 passed in 33.22s`).
2. An independent probe built for this revalidation
   (`/tmp/.../scratchpad/goc44_dhx1_probe.py`, not committed — throwaway) drove
   the same call shape with a bare `httpx.AsyncClient` (no au hardening) against
   an in-process FastMCP/uvicorn server and got the same result:
   `PROBE RESULT: OK ['ping'] [TextContent(...text='pong'...)]`.

`httpx` and `httpx2` are structurally identical forks (documented in
`agent_utilities/mcp/httpx_boundary.py`); `mcp.client.streamable_http` never does
an `isinstance` check against `http_client` itself — it only calls duck-typed
methods (`.stream()`, `.build_request()`, `.send()`, `.aclose()`), and even
`httpx2.EventSource(response)` (constructed from an `httpx.Response`, not
`httpx2.Response`, in the SSE fallback path) iterates correctly, verified directly
against `httpx2.EventSource`. The mismatch is a real *static*-typing fact with a
committed, currently-passing *runtime* proof it doesn't matter here. **No fix
applied; the existing test is the fix's evidence.** Coordination note: the ledger
names GOC-26 as the coordinating lane for this ID; this finding should be shared
with GOC-26 rather than re-litigated there.

### D-HX-2 — DECISION-REQUIRED (unchanged, not this lane's call)

> "Root cause of the httpx/httpx2 duality: deliberate early fastmcp>=4.0.0b1
> adoption, not vendoring/accident -- operator decision needed on consolidation path"

Confirmed accurate and still pending. `pyproject.toml`'s `[tool.uv]
override-dependencies` comment (`fastmcp-slim[client,server]>=4.0.0b1`) documents
the same root cause in the same terms. GOC-87 has since landed the sanctioned
*interim* mitigation while that decision remains open: a staged transport-factory
seam (`agent_utilities/httpsupport/transport_factory.py`,
`docs/architecture/httpx_httpx2_migration.md`) that lets call families migrate to
`httpx2` one at a time, verified, rather than either a wholesale swap or an
indefinite freeze. As of the 2026-08-16 lock referenced in that doc: 17 locked
packages require `httpx`, 3 require `httpx2`; 48 files import `httpx`, 2 import
`httpx2` (denominator: files under `agent_utilities/`, per that doc). A repo-wide
grep run for this revalidation (denominator: all tracked `.py` files, a superset
including `tests/`) found 52 files importing `httpx` and 4 importing `httpx2` —
consistent with that count once test files are included. **No action for this
lane beyond recording that the interim mitigation exists and the consolidation
decision itself remains genuinely unmade; it is an operator call, not a
revalidation outcome.**

### D-VI-2 — CONFIRMED-LIVE, filed as owned blocker (not fixed this pass)

> "Unrecognized uv flags degrade to uv's own sync behaviour"

Confirmed still true by reading `scripts/uv_workspace.py::main`/`uv_plan`
directly: the CLI parser uses `nargs=argparse.REMAINDER` (line ~1810) and every
subcommand branch splices the remainder straight through to the real `uv` binary
(`*tail` for `sync`/`lock`, `command.extend(tail)` for `run`, `[*base, *arguments]`
for anything else) with no allow-list or validation layer. A flag `uv` itself
doesn't recognize will fail via `uv`'s own error (that much is fail-closed), but a
flag `uv` *does* recognize and that changes sync/lock semantics in a way this
wrapper's environment-partitioning, `UV_PROJECT_ENVIRONMENT` pinning, or
pool-gated-heavy-sync assumptions (`_dependency_sync_slot()`) don't account for
would pass straight through and "degrade to uv's own [unpartitioned] sync
behaviour" exactly as the imported title says — this is a real, reproducible gap
against the lane's own authority ("Unsupported uv flags... fail closed with the
exact command and supported alternative; no unknown-flag pass-through is
accepted").

**Not fixed in this pass.** A safe fix requires enumerating `uv`'s actual flag
surface per subcommand (`sync`/`lock`/`run`/others) and classifying each as
safe-to-passthrough vs. wrapper-assumption-breaking — this wrapper is used by
every lane's `uv-lock` pre-commit hook and pool-gates real dependency downloads
fleet-wide (per the workspace `CLAUDE.md` GOC context and this repo's own
`git-hook ambient env breaks ls-files`-class incident history), so an incomplete
allow-list risks a false-positive rejection that blocks every lane's commits —
exactly the failure mode `gates-report-more-coverage-than-they-have` warns
against. This is properly GOC-44-W02/W03 scope (candidate/compatibility contract
definition, flag/interpreter parity) rather than a same-pass patch. **Filed as an
owned blocker**: needs a dedicated audit of `uv sync|lock|run`'s flag surface
before an allow-list can be written and tested without full-gate access.

### D-VS-1 / D-VS-2 — STALE-PREMISE (resolved by an ecosystem architecture change)

> D-VS-1: "★★ ENDGAME: make the bare `uv sync` *correct* via a two-line
> root-manifest change"
> D-VS-2: "Nothing intercepts a human or agent typing `uv sync` directly"

Both resolved — not by intercepting bare `uv sync`, but by removing the shared
`[tool.uv.workspace]` root that made a bare sync's behavior depend on sibling
repos in the first place. `/home/apps/workspace/pyproject.toml` (the former
ecosystem workspace root) now carries an explicit `D-EGSFT-1` header dated
**2026-08-14** ("owner-approved architecture change"): it is "intentionally NOT a
`[tool.uv.workspace]` root anymore." Every repo (69/69 `agents/*` + `agent-utilities`
+ `epistemic-graph` + `agent-webui` + `geniusbot` + `langfuse-agent` +
`leanix-agent`, per that file's own verification note) now resolves and locks
independently against its own `uv.lock`, so a bare `uv sync` run inside any one
repo resolves *that repo's own lock* — the correct behavior — rather than being
pulled into a shared workspace-wide resolve. The former workspace-root-only `[tool.uv]
override-dependencies` (nltk/pytest/cryptography/opentelemetry/fastmcp-slim
floors) were redistributed into each repo that needs them (verified for
fastmcp-slim above under D-2.1b-2).

**Caveat, not a fix needed by this lane:** `agent-utilities/pyproject.toml`
still carries a "Known residual limitation" comment (lines ~844-857) about bare
`uv sync` from the *canonical* checkout (not a worktree) hitting a
per-member-name-collision when it discovers the outer ecosystem workspace as an
ancestor — but that comment is dated **2026-08-13** (`git log -L`), one day
*before* D-EGSFT-1 removed the outer `[tool.uv.workspace]` table entirely. Since
the collision rule this comment describes only fires when uv discovers an
ancestor workspace, and there is no longer one to discover, this documentation is
now very likely stale itself. **Not verified empirically** in this pass (would
require running bare `uv sync` inside the canonical
`agent-packages/agent-utilities` checkout, which this lane's own instructions
prohibit editing/running heavy operations against). Flagged for whoever next
touches that comment to confirm and prune if stale.

### D-W5AGB-1 — STALE-PREMISE (resolved 2026-08-07, before this lane existed)

> "Ecosystem-wide uv.lock has a deliberate cryptography<49 override
> (mlflow/data-science-mcp) that directly contradicts agent-utilities'
> CVE-driven >=50.0.0 floor"

Resolved. `agent-packages/agents/data-science-mcp/pyproject.toml` carries a
dated comment (`2026-08-07 (D-W5AGB-1)`) explaining that `mlflow` — the sole
source of the `cryptography<49` requirement — was removed from that package's
optional dependencies specifically because no published `mlflow` release (as of
that date, including the-then-latest 3.15.1) supports `cryptography>=50.0.0`
(the fix for `PYSEC-2026-3552`). Verified for this revalidation: `grep -n
"mlflow" agents/data-science-mcp/pyproject.toml` shows only comment lines (the
historical explanation and reinstatement condition), no active dependency
declaration; `grep -rln "cryptography<49"` across `agent-packages/*/pyproject.toml`
and `agent-packages/agents/*/pyproject.toml` (denominator: 1 + 68 files) returns
only that same commented-out file. `agent-utilities/pyproject.toml`'s own floor
(`"cryptography>=50.0.0"`, line 56) is unchanged and uncontested.

### D-2.1b-2 — STALE-PREMISE (resolved, fully verified)

> "Downstream `agents/*` packages need the same `fastmcp-slim[client,server]`
> override"

Resolved and verified with a full-population check, not a sample. Denominator:
69 directories under `agent-packages/agents/` (68 with a `pyproject.toml`; the
69th is `tests/`, which has none). Of the 68, all 68 mention `fastmcp` and all
68 already carry `"fastmcp-slim[client,server]>=4.0.0b1"` in their own
`[tool.uv] override-dependencies` (`grep -l "fastmcp-slim\[client,server\]"
*/pyproject.toml` → 68/68). **0 of 68 are missing the override.**

### D-2.1b-3 — STALE-PREMISE (resolved, live-tested)

> "`tests/integration/` not re-run under fastmcp 4 (only `tests/unit/mcp/` +
> the rewritten live-path tests)"

Resolved. `tests/integration/` contains 81 test files, 16 of which reference
`fastmcp`/`mcp.` (denominator: files under `tests/integration/`, `grep -rl`).
A dedicated fastmcp-4 live-server integration test exists —
`tests/integration/mcp/test_fastmcp_cross_version.py` — whose own docstring
states it *replaces* the prior dual-extra fastmcp-3-vs-4 cross-version guard
because fastmcp 4 is now au's single default; it drives
`agent_utilities.mcp.toolset_factory.build_stdio_toolset` (au's actual
production call path) against a real fastmcp-4 stdio server end-to-end. Re-run
today: **PASSED** (`goc-87/.venv`, `pytest
tests/integration/mcp/test_fastmcp_cross_version.py -v` → `1 passed, 4 warnings
in 23.47s`; warnings are pydantic-ai `DeprecationWarning`s about MCP SDK v2
field renames, unrelated to this ID). Git history for `tests/integration/`
(`git log --oneline -5`) shows recent, active maintenance (`fix(tests): resolve
the long tail of single-test failures`, etc.), not an abandoned suite.

### D-W2P-7 — STALE-PREMISE (self-described transient state)

> "guardrail-surface-parity / guardrail-cpd-drift / 1 gate-meta-test: worktree
> .venv not yet synced (contended, not a code defect)"

The imported title self-classifies this as *not a code defect* — a point-in-time
contention state in one worktree (`w2-precommit-0802.md`) that no longer exists.
Not independently reproducible today; no corresponding worktree or contention
state exists in the current environment. No action needed; recorded as
STALE-PREMISE per its own original disposition, not a new finding.

### D-CIP-17 — CONFIRMED-LIVE, already adequately mitigated

> "uv-managed CPython (python-build-standalone) lacks os.memfd_create --
> promote_local_release tests fail under the project's own sanctioned local
> test runner"

The capability gap itself is confirmed, live, on the current interpreter:
`python3 -c "import os; print('memfd_create' in dir(os))"` under
`/home/genius/.local/share/uv/python/cpython-3.14.4-linux-x86_64-gnu/bin/python3.14`
prints `False`, even though the same host/kernel's distro interpreter has it
(this is a `python-build-standalone` build artifact, not a real platform
limitation — the host is Linux 7.0.0-28, which supports the syscall).

This is **already mitigated to the standard this lane's authority requires**,
not left silently degraded: `scripts/release/promote_local_release.py`'s
`_require_supported_platform()` explicitly checks
`hasattr(os, "memfd_create")` (and the matching `fcntl` seal constants) and
raises `ReleaseError("unsupported-platform")` — a typed, fail-closed refusal,
not a crash or silent skip. `tests/unit/release/test_promote_local_release.py`
documents the exact root cause in two places (`HAVE_MEMFD_CREATE=0` under uv's
pinned CPython 3.14.4, contrasted with the distro interpreter on the same
host/kernel) and deliberately monkeypatches around the irrelevant precondition
in tests that aren't about platform support, while a dedicated test
(`test_native_promoter_rejects_unsupported_platform`, referenced by name in
those comments) covers the fail-closed gate itself. **No fix applied — the
existing gate and tests already meet the bar.** The residual fact that
`promote_local_release`'s real signing flow cannot execute end-to-end under
the sanctioned uv-managed interpreter on this host is a genuine operational
constraint (needs either a memfd_create-capable interpreter build or an
alternate sealed-fd primitive), not a design defect, and is outside a single
revalidation pass to resolve.

### D-CIP-3 — CONFIRMED-LIVE at the PyPI level, already worked around

> "epistemic-graph[full]>=2.23.2 is unsatisfiable from PyPI (max published
> 2.23.0)"

Confirmed still numerically true: `curl -s
https://pypi.org/pypi/epistemic-graph/json` → latest published `2.23.0`;
`pyproject.toml` line 144 requires `"epistemic-graph[full]>=2.23.2,<3.0.0"`.
**But no live workflow actually resolves this against bare PyPI.**
`[tool.uv.sources]` (`pyproject.toml` line 858) overrides `epistemic-graph` to
a local editable path (`.uv-workspace-siblings/epistemic-graph`) — materialized
by `scripts/uv_workspace.py`'s `materialize_own_siblings` for local/lane use and
by `.github/workflows/security.yml`'s `actions/checkout` for the one CI job that
needs the real package; every other CI workflow instead excludes the package
entirely (`--no-install-package epistemic-graph --no-install-package
langfuse-agent`, per the same comment block). So the unsatisfiable-from-PyPI
fact is real, but every actual resolution path either bypasses PyPI for this
package or excludes it — there is no reproducible failure to fix here. The root
fix (publish `epistemic-graph>=2.23.2` to PyPI) is GOC-31's territory per the
ledger's own coordination column, consistent with the wider "PyPI publish gate
never runs" finding GOC-42 already made for `agent-utilities` itself (see
D-W6TG-1 below for the same pattern hitting a downstream consumer).

### D-MQR-4 — CONFIRMED-LIVE (version divergence), partially unverified

> "Reinstall repository_manager editable against python3.14 and reconcile
> 1.3.53-vs-3.0.0 version divergence"

The version-divergence half is confirmed and, if anything, larger than the
imported title states: `agent-packages/agents/repository-manager/pyproject.toml`
declares local `version = "3.4.0"`; PyPI's published `repository-manager` is
`2.0.1` (`curl -s https://pypi.org/pypi/repository-manager/json`). This is the
same class of finding already recorded in the operator's own memory
(`gate-tool-published-before-gate-adopted.md`: "PyPI had 2.0.1... local shipped
3.4.0"), so this is a **known, tracked, still-open** condition, not a new
regression — this revalidation independently reconfirms the same numbers.
`requires-python = ">=3.12,<3.15"` in that `pyproject.toml` covers 3.14, so the
python3.14 *compatibility declaration* is fine on its face.

**Unverified in this pass:** whether `repository_manager` is actually
*installed editable against a python3.14 interpreter* anywhere it needs to run
(the mission-relevant half of the imported title). No editable install of
`repository_manager` was found in the one venv inspected
(`goc-87/.venv`, `pip show repository-manager` → module not present, and that
venv has no `pip` at all) — but that venv is `agent-utilities`' own, not
`repository-manager`'s, so its absence there is not evidence either way. This
lane did not locate or inspect wherever `repository-manager`'s own editable
install actually lives operationally (its own worktree/venv, or the graph-os
container). Left unverified rather than guessed at.

### D-W6TG-1 — CONFIRMED-LIVE, root cause outside this lane, coordinate with GOC-42

> "geniusbot Build workflow red on origin/main: requirements.txt/pyproject.toml
> require agent-utilities>=2.0.0, unpublished on PyPI (== D-W2PU-1 root cause)"

Confirmed still live, and slightly worse than the title states once both of
geniusbot's dependency declarations are checked, not just one:
- PyPI `agent-utilities` latest published is **1.26.4**
  (`curl -s https://pypi.org/pypi/agent-utilities/json`) — consistent with
  GOC-42's prior finding cited in this lane's own briefing.
- `agent-packages/geniusbot/pyproject.toml` line 27 requires
  `"agent-utilities>=2.0.0"` — unsatisfiable from PyPI alone, exactly as the
  imported title says. It is worked around the same way as D-CIP-3 above: a
  `[tool.uv.sources]` local editable-path override (line 85) resolves it via
  `.uv-workspace-siblings/agent-utilities` for uv-based workflows.
- `agent-packages/geniusbot/requirements.txt` line 7 pins
  `agent-utilities==1.0.0` — a *third*, even more stale expectation, and one the
  `[tool.uv.sources]` override cannot reach at all (plain `pip install -r
  requirements.txt` resolves straight against PyPI). Any CI job that installs
  from `requirements.txt` rather than through `uv` sees a target that is
  neither what `pyproject.toml` wants (`>=2.0.0`) nor what PyPI actually has
  latest (`1.26.4`).

This is the same root cause GOC-42 already identified for `agent-utilities`
itself ("PyPI `agent-utilities` is capped at 1.26.4 while local is far ahead");
D-W6TG-1 is that same root cause surfacing in a downstream consumer's CI. The
fix is a real PyPI publish of `agent-utilities>=2.0.0`, which is GOC-42's
release-authority territory and outside this lane's non-goals (no publication
without explicit authorization). **Recommendation only, not applied:**
`geniusbot/requirements.txt`'s `agent-utilities==1.0.0` pin is additionally
inconsistent with `geniusbot/pyproject.toml`'s own `>=2.0.0` floor regardless
of the publish gap, and independent of the PyPI-publish fix — that file is
outside this lane's owned-repository list (`geniusbot` is not `agent-utilities`,
`agents/*`, or `repository-manager`) and was left untouched.

## What is unverified

- The "Known residual limitation" comment in `agent-utilities/pyproject.toml`
  (bare `uv sync` from the canonical checkout) was not empirically re-tested
  against the post-D-EGSFT-1 (2026-08-14) architecture — this lane's
  instructions prohibit running heavy operations against the canonical
  checkout, and this worktree (`goc-44`) has no venv of its own to test with
  in isolation either.
- D-MQR-4's `repository_manager` editable-install-against-python3.14 claim:
  this lane did not locate where that install actually runs operationally.
- D-VI-2's full `uv` flag surface was not enumerated; only the wrapper's own
  passthrough behavior was confirmed by code reading, not exhaustive testing
  against every real `uv` flag.
- No new code was authored or changed in `agent_utilities/` in this pass; the
  two live tests cited (`test_multiplexer_remote_transport_live.py`,
  `test_fastmcp_cross_version.py`) were re-run, not modified.
