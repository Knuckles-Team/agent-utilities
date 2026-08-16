# `agent_utilities.measurement` — the measurement harness

Makes a whole class of "instrument differed from the thing being claimed
about" false alarms mechanically impossible. Built from eight real,
catalogued incidents in one session — see each module's docstring for the
exact incident it fixes, and `plans/graph-os-completion-program/BUG-LEDGER.md`
BUG-220..BUG-229 for the defects found while building it.

Stdlib only. Works from a bare `python3` — no venv, no install:

```
python3 -c "
import sys; sys.path.insert(0, '.')
from agent_utilities.measurement import ProvenanceHeader
print(ProvenanceHeader.start(['true']).finish())
"
```

## Why agent-utilities, not epistemic-graph

`epistemic-graph` is the Rust engine; `agent-utilities` is the Python layer
every gate, script, and CI job in BOTH repos' `scripts/` trees is already
written in and already imports from. The incidents this harness fixes
(pipeline exit codes, git diffs, rsync copies, load averages, process
kills, systemd-run) are shell/process/git-level concerns any repo's tooling
hits, not epistemic-graph-engine-specific ones. Putting the harness in
`agent-utilities` and letting `epistemic-graph`'s own Python-side scripts
`import agent_utilities.measurement` (it is already an `agent-utilities`
consumer, per its own `pyproject.toml`) avoids a second copy to keep in
sync. Rust-side gates in `epistemic-graph` remain out of scope for a
Python stdlib package; that gap is real and not mechanized here (see
"What's not mechanical" below).

## The eight modules, one per incident

| Module | Capability | Incident it fixes |
|---|---|---|
| `provenance.py` | A | Header records interpreter/host/cpu-affinity/env-fingerprint; catches the venv (5), taskset-vs-CI (6), and sccache (7) drift incidents when two headers are compared |
| `load_gate.py` | B | Refuses to emit a verdict above a load threshold — the load-15.82-vs-4.03 noise incident (4) |
| `copy_integrity.py` | C | Manifest-hash-verifies a copy before it's measured — the `rsync -a` (no `-H`) dropped-hardlink incident (3) |
| `run.py` | D | Captures the real exit status of the measured process, never a pipeline stage's — the `cmd \| tail` + `$?` incident (1) — plus a linter for the shape |
| `merged_tree.py` | E | `git merge-tree --write-tree`, not a two-dot diff — the false branch-deletion alarm (2) |
| `proc_safety.py` | F | `kill_by_pattern` excludes caller + ancestors and requires a real argv-token match — the `pgrep -f` self-kill incident (8) |
| `background.py` | G | `systemd-run` with the output redirect INSIDE the unit's command — the journal-vs-file incident (twice) |

## Adopting it in a lane

1. **Wrap a verdict-producing script.** Replace `subprocess.run(cmd, shell=True)`
   / raw pipelines with `agent_utilities.measurement.run.run(argv_list)`, and
   attach a `ProvenanceHeader.start(argv_list).finish()` to whatever you
   report. Before trusting/printing that report, call
   `require_provenance(result)` — it raises if the header is missing or
   incomplete.
2. **Before a heavy/parallel gate run**, call
   `agent_utilities.measurement.load_gate.gate_or_raise()` (or `check_load()`
   if you want to handle `TOO_LOADED_TO_MEASURE` yourself instead of an
   exception). This repo's own `CLAUDE.md`/memory already document the
   danger zone (load ~62 on a 24-core box, swap exhausted) this defends
   against.
3. **Before measuring a copied tree** (a test host, a scratch clone), use
   `copy_tree(source, dest)` instead of a bare `rsync`/`cp` — it raises
   `CopyIntegrityError` if the copy doesn't manifest-match the source,
   instead of letting a silent partial copy be measured.
4. **Before asking "does branch X delete file Y"**, use
   `files_deleted_by_merge(repo, base, branch)`, never a two-dot
   `git diff base..branch`.
5. **Before `pkill`/`pgrep`-style cleanup** in any script, use
   `kill_by_pattern(pattern, dry_run=True)` first to see what it WOULD hit;
   flip `dry_run=False` only once the dry run's `matched` list looks right.
6. **Before backgrounding long work**, use `run_background(cmd)` — never
   hand-roll a `systemd-run ... bash -c "cmd"` without the redirect inside
   the quoted string.
7. **Wire the linter into pre-commit**: add
   `python3 scripts/check_measurement_exit_code_antipattern.py` as a fast-tier
   hook (mirrors the existing `check_*.py` scripts already wired into this
   repo's pre-commit config).

None of the above requires installing anything beyond what a lane's
checkout already has — this package has zero third-party dependencies.

## What's not mechanical (and why)

* **`environment_mismatches`/`require_same_environment` only catch a drift
  if BOTH runs actually captured a header.** Nothing forces every gate in
  the fleet to adopt `ProvenanceHeader` retroactively; adoption is
  per-lane, per module 1-7 above. A gate that never wraps its verdict in a
  header is invisible to this whole capability — same as before.
* **The D linter is a regex/lookahead heuristic, not a real shell parser.**
  It correctly catches the incident's own shape and the common
  `tail`/`head`/`grep`/... siblings, and correctly ignores a
  `pipefail`/`PIPESTATUS`-guarded pipeline, but a sufficiently obfuscated
  pipeline (e.g. built up across `eval`/variables) can evade it. A full
  shell AST parser was judged not worth the dependency-light constraint for
  a pre-commit-speed check.
* **`proc_safety`'s `require_token_match` is a heuristic name check**
  (`tok == program or tok.endswith("/" + program) or Path(tok).name ==
  program`), not a full "is this literally the same binary" check (it does
  not resolve symlinks or compare inodes). It is deliberately conservative
  in the safe direction: it under-matches (misses some real targets) rather
  than over-matches (risks a false positive kill).
* **`load_gate`'s default threshold (1.5x core count) is a judgment call**,
  not derived from a formal model — it is loose enough to avoid
  false-flagging ordinary CI parallelism while staying well below the
  documented ~2.6x-cores/swap-exhausted danger zone. Override via
  `MEASUREMENT_LOAD_THRESHOLD` per host if that default doesn't fit.
* **`background.py` requires a real `systemd-run --user` session.** There
  is no fallback to `nohup`/`disown`/a plain background `&` — those don't
  give the same "list units, `--collect` GC, `is-active` poll" semantics,
  and silently substituting one would reintroduce a different version of
  the same "watching the wrong thing" class of bug this package exists to
  prevent. On a host without a systemd user session, `run_background`
  raises `SystemdRunUnavailableError` rather than degrading silently.
* **Rust-side (epistemic-graph) gates are out of scope.** This package is
  Python stdlib; a Cargo/Rust-native equivalent (e.g. for the sccache/CI
  incident, which is specifically a Cargo build concern) is not built here.
  epistemic-graph's Python-side scripts can still `import
  agent_utilities.measurement` for the parts that apply to them (git,
  process, load, background-run).
