# The shared workspace `.venv` — sync, flip-on-merge, drift, upgrade

> CONCEPT:AU-OS.deployment.workspace-venv-reconciler ·
> CONCEPT:AU-OS.safety.destructive-sync-refusal ·
> CONCEPT:AU-OS.deployment.merge-triggered-venv-flip ·
> CONCEPT:AU-OS.host.venv-drift-detector

The ecosystem is developed against **one** virtualenv at the uv-workspace root,
with ~75 members installed editable into it and ~26 git worktrees running their
tests through it. This document is the contract for changing that environment.

Everything here is implemented in `agent_utilities/deployment/venv_sync.py`
(engine) and `agent_utilities/deployment/venv_autosync.py` (merge trigger), and
exposed as `agent-utilities-venv` plus the dependency-free launcher
`scripts/venvctl`.

---

## The one rule

```
uv sync --locked --all-packages --inexact
```

**Never a bare `uv sync`.** The workspace root project declares zero
dependencies, so its target set is empty and a bare sync prunes everything that
is not in it. Measured on 2026-07-31 in the live workspace: **557 uninstalls,
including all 75 editable members.**

`--all-packages` and `--inexact` have no `UV_*` environment equivalents, so the
safe form cannot be made the shell default. It is instead made the *only* form
this code can produce:

* `SyncInvocation` has no field that can drop a flag from
  `SANCTIONED_SYNC_FLAGS`;
* `_assert_sanctioned()` re-checks the argv immediately before `subprocess` sees
  it, so an argv assembled anywhere else still cannot be executed;
* every plan is computed with `--dry-run` first and passed through the
  guardrails before anything is applied.

### Recommended follow-up (not applied here)

The structural cure is to make the bare form *correct* rather than merely
guarded: give the root project real dependencies on the workspace members (the
`[tool.uv.sources]` entries already exist, unused, at the workspace root), so
`uv sync`'s target set is the whole workspace. That requires a relock and was
deliberately out of scope — see `reports/deferred/lane-venv-autosync.md`.

---

## Commands

| Command | What it does |
|---|---|
| `venvctl status` | Drift report: lock currency, environment currency, member metadata, MCP SDK floor, stuck flips. Exit 3 on `fail`. |
| `venvctl plan` | Read-only: the sanctioned plan plus the guardrail verdict. |
| `venvctl sync` | Make the environment match `uv.lock`. Guardrailed. |
| `venvctl upgrade --package X` | Move `X` forward: back up → relock → sync → verify → **auto-roll-back on failure**. |
| `venvctl relock` | Same loop, re-resolving everything (`uv lock --upgrade`). |
| `venvctl rollback [--to ID]` | Restore an archived `uv.lock` and re-sync onto it. |
| `venvctl backups` | List archived lock revisions. |
| `venvctl verify` | Run the health probes on their own. |
| `venvctl members` | Per-member editable install state. |
| `venvctl activity` | What the activity probes currently see. |
| `venvctl lease acquire --owner me --ttl 3600` | Declare a do-not-swap window. |
| `venvctl autosync on\|off\|status\|install\|uninstall\|drain` | The merge trigger. |

---

## Guardrails

Evaluated in two phases — context first (cheap, can `defer`), then the plan.
**A refusal always outranks a deferral**: a plan that would destroy the
environment is wrong regardless of when it runs.

| Guardrail | Verdict | Fires when |
|---|---|---|
| `activity` | `defer` | Another lane is mid-test/build (see below). |
| `lock_consistency` | `refuse` | `uv lock --check` fails — a manifest moved; that needs an explicit, backed-up relock, not a silent sync. |
| `member_uninstall` | `refuse` | The plan would **net-remove** an editable workspace member. |
| `locked_uninstall` | `refuse` | The plan would net-remove something `uv.lock` still requires. |
| `uninstall_budget` | `refuse` | Net removals exceed the caller's sanctioned budget (default **0**). |

Register your own with `venv_sync.register_guardrail(...)`; the defaults are a
list, not a policy branch.

### Removals vs replacements

uv renders a package *replacement* as an uninstall line plus an install line,
and writes the reinstall of a local package as `name @ file:///…` rather than
`name==version`. Every editable member looks exactly like that whenever its
metadata is rebuilt. `SyncPlan.removals` is therefore uninstalls **not** paired
with a reinstall, and that is what the guardrails judge — otherwise the
guardrails refuse the correct sync of a dependency change.

### Activity detection ("never fire while another lane is mid-test")

`ProcessActivityProbe` reads `/proc` for three independent signals:

1. `argv[0]` under `<venv>/bin` — `/proc/<pid>/exe` is useless here because uv's
   venv python is a symlink into uv's own python store;
2. `VIRTUAL_ENV` pointing at the shared venv;
3. a build/test **program name** (`pytest`, `cargo`, `maturin`, `uv run`, …)
   whose cwd is inside the workspace or a sibling worktree. Matched against argv
   *tokens*, never the joined command line — substring matching flagged every
   `bash -c '…pytest…'` wrapper.

`LeaseActivityProbe` honours explicit, TTL-bounded leases for work the process
scan cannot see. Both are registries (`register_activity_probe`). A probe that
*errors* reports activity, so an unreadable system defers rather than proceeds.

---

## Flip on local merge to `main`

### Why git hooks

`post-merge` fires exactly once, synchronously, when a merge lands, and via
`ORIG_HEAD` it knows precisely which files moved — which is what decides whether
any work is needed. A watcher would poll ~75 repositories and cannot tell a
merge from an editor save; a `make` target is not automatic (the failure being
fixed is that nobody remembered to run it); CI cannot help because the flip is
local and the workspace root is not even a git repository.

Hooks are written into the repository's **common** git directory, so one install
covers the canonical checkout *and* every linked worktree. `post-checkout` and
`post-rewrite` are installed too, so a branch switch or a rebase onto `main` is
seen as well. Each hook is a delimited managed block appended to whatever is
already there (pre-commit owns some hooks in this repo) and removed cleanly on
uninstall.

### The hook only enqueues

It writes an intent record and returns. The reconciler runs **detached**, takes
the exclusive writer lock, and applies the flip under the same guardrails as any
other mutation. A deferred or refused flip stays queued, and `venvctl status`
reports the backlog as `pending_flips` — late, never lost, never silent.

### Only the live checkout counts

A merge in a linked worktree changes nothing about what the shared venv runs;
only the member directory the editable install points at is "live". The trigger
checks that first, before the branch check.

### Source change vs metadata change

This is the distinction the whole feature turns on.

| Change | Live already? | Action |
|---|---|---|
| `.py`/docs inside an editable member | **Yes** — the member *and every downstream dependent* import through the source tree | none; recorded as `already-live` |
| `pyproject.toml` / `setup.cfg` deps, version, scripts | No — baked into `.dist-info` at install time, and a dependency change invalidates the resolution every dependent shares | `on_metadata_change` policy |
| `.rs`/`.c`/`Cargo.toml` | No — the extension must be rebuilt | sync |
| `uv.lock` | No | sync |

Classification from the merge diff is the cheap trigger-time signal and is
biased toward doing work: an unreadable diff escalates to `metadata` rather than
being read as "nothing changed". `member_install_states()` is the authoritative
check that runs anyway — it compares each member's *source* metadata against its
installed `.dist-info` (version, console scripts, editability). It deliberately
does **not** re-diff requirements: `uv lock --check` already answers that
exactly, and a second fuzzier answer would only disagree.

**Downstream dependents never need reinstalling.** They resolve imports at
runtime from `site-packages`; what has to change is the shared *resolution*,
which is exactly `uv lock` followed by the sanctioned sync.

### `on_metadata_change`

* **`propose`** (default) — record it loudly and leave the relock to an
  operator. A relock re-resolves a lock shared by ~26 worktrees; that is not a
  side effect to take silently.
* **`relock`** — run the full backed-up, verified, auto-rolled-back
  `upgrade --all` automatically.
* **`sync-only`** — sync against the existing lock and report the staleness.

---

## Turning it on and off

```bash
python3 agent-packages/agent-utilities/scripts/venvctl install-launcher   # ~/.local/bin/venvctl
venvctl autosync install          # hooks into every workspace member (idempotent)
venvctl autosync on               # ← the switch
venvctl autosync status
venvctl autosync off              # hooks stay installed and become inert
venvctl autosync uninstall        # remove the managed hook blocks entirely
```

State lives in `~/.local/state/agent-utilities/venv-autosync/<root>-<hash>/`:
`autosync.json` (the switch and policy), `intents/` (queued flips),
`lock-backups/`, `leases/`, `runs/`, `writer.lock`, and the trigger logs. The
path reads **no** environment variable on purpose — a hook, a detached
reconciler and an interactive shell run with three different environments, and a
movable state path would split the queue between them.

---

## Upgrading and rolling back

```bash
venvctl upgrade --package fastmcp        # one dependency forward
venvctl relock                           # everything forward
venvctl backups                          # every archived uv.lock
venvctl rollback                         # restore the most recent
venvctl rollback --to 20260731T133043Z-1e2f98d701ee
```

`uv.lock` is **untracked and lives at a non-git root**, so `LockBackupStore` is
the only rollback path that exists. Every mutation archives the lock first
(content-addressed, with a digest verified on restore); `rollback` archives the
pre-rollback state too, so a rollback is itself undoable. Retention never
discards a backup marked `verified`.

`upgrade` is the whole loop: back up → relock → guardrailed sync → verify →
**auto-roll-back and re-sync** if any probe fails. Verify probes:

| Probe | Asserts |
|---|---|
| `lock_check` | `uv lock --check` still passes |
| `clean_plan` | the environment matches the lock exactly |
| `imports` | canary modules import **inside the venv** |
| `mcp_sdk_floor` | delegates to `check_mcp_sdk_floor()` when present |

An import canary that is *absent* is `ok=None` (inapplicable); only
present-but-unimportable is a failure — which is precisely the shape of the
ten-day outage. `agent_utilities.mcp.child_resilience` is a default canary for
that reason.

---

## Drift detection

`venvctl status` and the `venv_drift` check in `agent-utilities doctor` run the
same `detect_drift()`. Findings: `lock_current`, `env_current`,
`env_extraneous`, `member_metadata`, `mcp_sdk_floor`, `pending_flips`. The
doctor check `skip`s cleanly where there is no uv workspace (a deployed
runtime).

This is the piece whose absence caused the incident: the venv ran `fastmcp`
3.4.4 / `mcp` 1.28.1 against a lock wanting 4.0.0b1 / 2.0.0 for ten days, one
import failed, an entire test module stopped collecting, and thirteen real
defects were invisible. Nothing checked, so nothing was known.

---

## When the venv is broken

`venv_sync` imports **only** the standard library, and `scripts/venvctl` falls
back to loading it by file path under namespace stand-ins rather than executing
`agent_utilities/__init__.py` (which pulls `httpx` and friends). A single module
instance is kept, so there is one guardrail registry, and the degraded mode is
announced on stderr rather than being silent.
