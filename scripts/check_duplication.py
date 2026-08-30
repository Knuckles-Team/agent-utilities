#!/usr/bin/env python3
"""CX program — the ONLY sanctioned way to invoke jscpd in this workspace.

Adopts jscpd v5 (the Rust-engine rewrite; NOT the old jscpd@4.x TypeScript
tool) as the ONE clone/duplication engine, replacing kiss for this purpose:
kiss's own `.kissconfig` sets `duplication_enabled = false`, and
`plans/complex/scripts/run_kiss.sh` documents that `kiss check` given more
than one positional path silently reports 0 violations regardless of
reality. jscpd finds real, independently-verified duplication kiss never
could (see the WD4-TOOL-01 report for the full BUG-CX-044/BUG-CX-027 proof).

Two modes:

  census   (DEFAULT, also `--mode census`) — unconditional. NEVER fails on
           findings; always prints the REAL number for the given paths (or
           the whole repo if none given). This is the no-ratchet policy in
           code: nothing is hidden, nothing is gated here.

  enforce  (`--mode enforce`) — diff-scoped. Recomputes the clone set of the
           BASE ref's live tree and of the tree this change would actually
           produce if merged (`git merge-tree --write-tree`, materialized
           into a throwaway detached worktree — never a ref, never a stash;
           see AGENTS.md-level guidance against `git stash` in a shared
           multi-worktree repo), diffs the two clone-pair sets by content
           hash, and fails ONLY on pairs that are NEW in the after-state.
           Pre-existing clones are still printed in full — reported, not
           gated. NO BASELINE FILE is ever read or written; the "before"
           state is recomputed live from the base ref on every run.

The production differential pass is intentionally bounded to the reviewed
``jscpd_diff_formats`` list in ``pyproject.toml``. It includes code as well as
templates, configuration, and documentation: dupehound catches structurally
equivalent whole functions, while jscpd must still catch a copied block inside
two otherwise different functions. Dupehound runs at pre-commit and jscpd at
pre-push, so one incident does not create two simultaneous blocking hooks. The
all-format ``census`` remains advisory.

Exit codes: 0 clean (or census mode, always), 1 NEW duplication in `enforce`
mode, 2 CANNOT RUN. A gate that could not run has NOT found nothing.

────────────────────────────────────────────────────────────────────────────
Traps verified in THIS workspace 2026-08-28, all defended below:

 TRAP-J1  `jscpd --version` prints `cpd 5.0.16`, NOT `jscpd 5.0.16` — the
          published crate/binary is literally named `cpd`. A version guard
          that checks for the string "jscpd" would CANNOT-RUN forever on a
          perfectly good install. The expected string is derived from the
          central pyproject.toml scanner table.

 TRAP-J2  jscpd silently auto-loads a `.jscpd.json` (or `.jscpdrc*`) from the
          CURRENT WORKING DIRECTORY with NO `--config`/`-c` flag needed —
          confirmed live: `{"minTokens": 999999, "threshold": 100}` in cwd
          made every run report 0 clones with no error, no notice. Exact
          kiss-`.kissconfig` shape of trap: a file nobody reviewed silently
          neuters every threshold. Config discovery is cwd-EXACT (does not
          walk parent directories — verified) but this script still refuses
          to run if any such file exists in the scan cwd, and separately
          pins every threshold-relevant flag on the command line so an
          ambient file cannot silently win even for a flag this script
          forgot to harden.

 TRAP-J3  `--ignore <globs>` filters the REPORT, it does NOT prune the
          directory walk. Verified: `agent-utilities`'s own canonical `.git`
          has accumulated 21,065 leftover files under `.git/agent-lanes/`
          from an old unfinished pytest run (itself containing NESTED throw-
          away git repos/worktrees) — `jscpd --ignore "**/.git/**" .`
          correctly reported 0 clones from that content but never returned
          (killed after 150s+); scanning `.git/agent-lanes` alone in
          isolation cost 18.5s of pure walk/read time that produced nothing.
          Defence: never hand jscpd a directory that is itself a git repo
          root (has a `.git` entry) — decompose to its own top-level
          children first (`_repo_scan_targets`). `--ignore` is kept ONLY as
          defense-in-depth for junk nested one level deeper than that.

 TRAP-J4  Without an explicit `--exit-code`, jscpd's process exit code is
          ALWAYS 0 — confirmed with real, non-trivial clones present and
          printed to console, `$?` was still 0. A hook that trusts jscpd's
          bare exit code is a permanent false green. This script never
          relies on it: it parses the JSON reporter's own clone COUNT for
          every decision, in both modes.

 TRAP-J5  `--exit-code` takes an OPTIONAL value (clap "greedy" parsing): an
          unquoted `--exit-code` immediately followed by a positional PATH
          consumes that path as if it were the numeric exit code and fails
          with `invalid digit found in string`. Combined with TRAP-J4, this
          script never passes `--exit-code` at all — see TRAP-J4's fix.

 TRAP-J7  `enforce` mode's two throwaway worktrees (~5,400 files each on this
          repo) land wherever `tempfile.gettempdir()` points, which defaults
          to `/tmp` — a FIXED-SIZE tmpfs (inode count set at boot from RAM,
          independent of how much space is actually used). On this shared,
          heavily multi-lane box that ceiling was hit live twice while
          verifying this script: `df -i /tmp` swung from 100% (7,650 free)
          to 87% (143,499 free) to 100% again within one hour with no
          action of this script's own — other concurrent lanes' own temp
          usage, not this script, drives it. A worktree checkout that dies
          97% of the way through from ENOSPC is indistinguishable from a
          real jscpd/git failure unless you know to check `df -i`. Fix:
          `_workdir()` below defaults enforce mode's throwaway worktrees to
          `<repo>/../.cx-dup-enforce-tmp` on the SAME real disk as the repo
          (`/home`, 62M inodes / 17% used at the time of writing — orders of
          magnitude more headroom than tmpfs's fixed ceiling), overridable
          with `$CX_DUP_WORKDIR`. `census` mode's report tempdir stays on
          tmpfs (small, short-lived, auto-cleaned) since it never remotely
          approaches this ceiling on its own.

 TRAP-J6  jscpd reports file counts in TWO places that sound like the same
          thing and are not. The CONSOLE table's "Files analyzed" column
          only counts files that participate in an ALREADY-detected clone
          (two 100%-identical 12-line files, below jscpd's own default
          `min-tokens=50`, reported "Files analyzed: 0" — indistinguishable
          at a glance from TRAP-J3's "walked nothing" failure). The JSON
          reporter's `statistics.*.sources` field is a DIFFERENT thing: the
          total count of files of that format actually scanned, clone or
          not — confirmed by matching it 1670-vs-1701 against an independent
          `find *.py` count on `agent_utilities/`. This script's own
          `_print_stats` reports `sources` (the real corpus-scanned count)
          plus an INDEPENDENT `_iter_files` count computed without asking
          jscpd anything, so a real zero can be told apart from a threshold
          miss or a walk that found nothing — trust neither number alone.
────────────────────────────────────────────────────────────────────────────

Usage::

    python scripts/check_duplication.py                    # census, this repo
    python scripts/check_duplication.py census PATH [PATH...]
    python scripts/check_duplication.py enforce [--base-ref main]

Install the exact version declared in ``pyproject.toml`` (do NOT let a hook
install it)::

    npm install -g jscpd@<jscpd_version>
    # then either put the `jscpd` bin on PATH, or:
    export JSCPD_BIN=/path/to/jscpd
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn

_AU_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _clone_scanner_config import (  # noqa: E402
    CloneScannerConfig,
    CloneScannerConfigError,
    is_excluded_path,
    is_jscpd_diff_path,
    load_clone_scanner_config,
)
from _git_subprocess_env import (  # noqa: E402
    sanitized_git_env,
    strip_inherited_git_repository_env,
)

# BUG-180-class defense: this script shells out to `git` (enforce mode). See
# scripts/_git_subprocess_env.py — a real `git commit`/merge-queue invocation
# exports GIT_DIR/GIT_INDEX_FILE into every hook subprocess it runs, and
# `-C <dir>` does NOT override them.
strip_inherited_git_repository_env()

_CONFIG_PATH = _AU_ROOT / "pyproject.toml"

# Directory NAMES that are never product source anywhere in this workspace.
# Matched by exact basename during the (shallow, one-level) decomposition in
# _repo_scan_targets — NOT a substitute for TRAP-J3's fix, which is never
# handing jscpd a `.git` root at all.
_JUNK_DIR_NAMES = {
    ".git",  # TRAP-J3 — the whole reason this set exists.
    ".venv",
    "venv",
    ".venv-base",  # interpreter trees; never product code.
    "node_modules",  # npm/pnpm dependency trees, not our code.
    "target",
    "target-isolated",  # cargo build output (isolated variant is
    # this workspace's own convention, see
    # AGENTS.md "eg shared cargo-target
    # corruption").
    "dist",
    "dist-primary",
    "dist-reproduction",
    "build",
    "build-artifacts",
    # build/packaging output, reproducible-build byproducts.
    "__pycache__",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    ".pytest_tmp",
    ".tox",  # tool caches / test scratch.
    ".eggs",
    "site-packages",
    "vendor",
    "third_party",
    "fixtures",
    "fixture",
    "samples",
    "sample",
    "examples",
    "htmlcov",  # vendored/3rd-party or
    # coverage HTML output.
    "coverage",
    "__snapshots__",  # snapshot-test fixtures: duplication here is BY
    # DESIGN (a snapshot IS a copy of expected output) —
    # scanning it only produces noise, per the brief's
    # agreed exclusion list.
}

# A repository's metadata must never become a scan target, even if a caller
# supplies a reduced prune set in a fixture or a future config edit. Other
# hidden directories (notably `.github`) remain eligible when their files map
# to a configured jscpd format.
_MANDATORY_PRUNE_DIRECTORIES = frozenset({".git"})


def _prune_names(prune_directories: frozenset[str] | None) -> frozenset[str]:
    configured = _JUNK_DIR_NAMES if prune_directories is None else prune_directories
    return frozenset(configured) | _MANDATORY_PRUNE_DIRECTORIES


def _die(msg: str) -> NoReturn:
    print(f"jscpd gate: CANNOT RUN: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _config() -> CloneScannerConfig:
    """Read the checked-in scanner contract without substituting defaults."""

    try:
        return load_clone_scanner_config(_CONFIG_PATH)
    except CloneScannerConfigError as exc:
        _die(str(exc))


def _setting(name: str, default: str) -> str:
    """Read a live process override through the repository config boundary."""

    try:
        from agent_utilities.core.config import setting

        value = setting(name, default, cast=str)
    except (
        ImportError,
        ModuleNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        _die(f"could not read repository setting {name}: {exc}")
    return str(value or default).strip()


def _resolve_jscpd(config: CloneScannerConfig | None = None) -> str:
    """Find jscpd WITHOUT consulting any package index at hook time — the
    same discipline as scripts/check_complexity.py's `_resolve_cccc`: a hook
    that resolves a tool from an index at hook time is how a previous fleet
    sweep shipped a gate that could not pass anywhere (69/226 push failures).
    """
    configured = _configured_binary("JSCPD_BIN")
    if configured:
        return configured
    for cand in (Path.home() / ".local/bin/jscpd", Path("/usr/local/bin/jscpd")):
        if cand.is_file():
            return str(cand)
    found = shutil.which("jscpd")
    if found:
        return found
    version = (config or _config()).jscpd_version
    _die(
        "`jscpd` not found. Looked at $JSCPD_BIN, ~/.local/bin/jscpd, "
        "/usr/local/bin/jscpd and $PATH. Install the pinned version with "
        f"`npm install -g jscpd@{version}` and either put it on PATH or set "
        "JSCPD_BIN. This gate never installs anything itself."
    )


def _configured_binary(setting_name: str) -> str | None:
    configured = _setting(setting_name, "")
    if not configured:
        return None
    candidate = Path(configured).expanduser()
    if not candidate.is_file():
        _die(f"{setting_name} points to a non-file path: {candidate}")
    return str(candidate)


def _run_version(exe: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
        _die(f"could not run `{exe} --version`: {exc}")


def _check_version(exe: str, config: CloneScannerConfig) -> None:
    expected = config.jscpd_version_output
    r = _run_version(exe)
    if r.returncode != 0:
        _die(
            f"`{exe} --version` exited {r.returncode}: "
            f"{_output_text(r.stderr).strip()[:400]}"
        )
    got = _output_text(r.stdout).strip()
    if got != expected:
        _die(
            f"version drift: want '{expected}', got '{got}' (see "
            "TRAP-J1 — the version string is 'cpd X.Y.Z', not 'jscpd "
            "X.Y.Z'). Thresholds/behaviour are calibrated per version; a "
            "different build may add, rename, or silently change defaults. "
            "The expected version is loaded from pyproject.toml."
        )


def _guard_ambient_config(cwd: Path) -> None:
    """TRAP-J2. A config file jscpd would silently auto-load from `cwd` with
    no flag needed. Confirmed name: `.jscpd.json`. The other names below are
    the conventional `rc`-file family this class of tool typically also
    honours; not independently confirmed against this binary, but refused
    on the same no-ratchet, no-ambient-config principle as kiss's
    `.kissconfig` guard — delete it and pass real flags instead."""
    for name in (
        ".jscpd.json",
        ".jscpdrc",
        ".jscpdrc.json",
        ".jscpdrc.yaml",
        ".jscpdrc.yml",
        "jscpd.config.js",
        "jscpd.config.cjs",
    ):
        p = cwd / name
        if _ambient_config_exists(p):
            _die(_ambient_config_message(p))


def _ambient_config_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as exc:
        _die(f"could not inspect jscpd config candidate {path}: {exc}")
    return True


def _ambient_config_message(path: Path) -> str:
    return (
        f"{path} exists. jscpd auto-loads this from the CURRENT "
        "WORKING DIRECTORY with no --config flag (TRAP-J2) and it "
        "silently overrides thresholds this script does not "
        "control. Delete it — thresholds are pinned on the command "
        "line, in this script, in git history, in the open."
    )


# File-level glob exclusions (`--ignore`) are loaded from the central
# ``pyproject.toml`` table.  They are defense-in-depth ONLY (TRAP-J3 proved
# that jscpd's ignore option does not prune the walk); root decomposition and
# the bounded diff target selection below keep the walk fast.


def _repo_scan_targets(
    root: Path, prune_directories: frozenset[str] | None = None
) -> list[Path]:
    """Safe scan target(s) for one directory.

    TRAP-J3's actual fix: never hand jscpd a path that is ITSELF a git repo
    root (has a `.git` entry) — decompose one level instead, dropping only
    configured junk directories and the metadata directory itself. Other
    dot-entries (including `.github`) are valid scan roots when their files map
    to a configured format. Pass the survivors (files AND directories both —
    an earlier version of this function kept only subdirectories and silently
    dropped every top-level *file*, losing e.g. a repo's own top-level
    *.py/*.md content from every census run).

    A `root` that is NOT itself a git repo root (e.g. `scripts/`, one
    directory inside a repo already decomposed one level up) is handed to
    jscpd WHOLE, unmodified — there is nothing to dodge one level down that
    --ignore does not already cover, and decomposing unconditionally is what
    silently dropped files in the first place.
    """
    if not _is_directory(root):
        return [root]
    if not _has_git_entry(root):
        return [root]

    return _kept_scan_children(root, _prune_names(prune_directories))


def _is_directory(path: Path) -> bool:
    try:
        path_stat = path.stat()
    except (OSError, ValueError) as exc:
        _die(f"could not inspect scan root {path}: {exc}")
    return stat.S_ISDIR(path_stat.st_mode)


def _has_git_entry(root: Path) -> bool:
    try:
        root.joinpath(".git").lstat()
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as exc:
        _die(f"could not inspect scan root {root}: {exc}")
    return True


def _kept_scan_children(root: Path, junk_names: frozenset[str]) -> list[Path]:
    try:
        children = sorted(
            root.iterdir(), key=lambda child: (child.is_symlink(), child.name)
        )
    except OSError as exc:
        _die(f"could not enumerate scan root {root}: {exc}")
    return [child for child in children if _keep_scan_child(child, junk_names)]


def _keep_scan_child(child: Path, junk_names: frozenset[str]) -> bool:
    return child.name not in junk_names and not child.name.endswith(".egg-info")


def _expand_roots(
    paths: list[Path], prune_directories: frozenset[str] | None = None
) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        out.extend(_repo_scan_targets(p, prune_directories))
    return _deduplicate_scan_targets(out)


def _deduplicate_scan_targets(targets: Sequence[Path]) -> list[Path]:
    """Keep one materialization of each real scan path.

    A symlink supplied alongside its target makes jscpd read the same file
    twice and can produce a false clone whose two locations are identical.
    Compare resolved paths and drop targets covered by an already-kept root;
    non-symlink paths are considered first so an alias cannot replace the
    repository's decomposed children with a broad, metadata-containing root.
    """
    ordered = sorted(
        enumerate(targets), key=lambda item: (item[1].is_symlink(), item[0])
    )
    kept: list[Path] = []
    resolved_kept: list[Path] = []
    for _, target in ordered:
        resolved = _canonical_scan_target(target)
        if target.is_symlink() and _has_git_entry(resolved):
            continue
        if any(
            resolved == existing or resolved.is_relative_to(existing)
            for existing in resolved_kept
        ):
            continue
        if target.is_symlink() and any(
            existing == resolved or existing.is_relative_to(resolved)
            for existing in resolved_kept
        ):
            continue
        if not target.is_symlink():
            _discard_nested_scan_targets(kept, resolved_kept, resolved)
        kept.append(target)
        resolved_kept.append(resolved)
    return kept


def _discard_nested_scan_targets(
    kept: list[Path], resolved_kept: list[Path], resolved: Path
) -> None:
    nested = [
        index
        for index, existing in enumerate(resolved_kept)
        if existing.is_relative_to(resolved)
    ]
    for index in reversed(nested):
        del kept[index]
        del resolved_kept[index]


def _canonical_scan_target(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        _die(f"could not resolve scan target {path}: {exc}")


def run_jscpd(
    exe: str,
    targets: list[Path],
    out_dir: Path,
    cwd: Path,
    *,
    echo_console: bool = True,
    config: CloneScannerConfig | None = None,
    formats: tuple[str, ...] | None = None,
) -> dict:
    """Run jscpd over `targets`, writing a JSON report into `out_dir`
    (caller owns cleanup — always a tempdir in this script, never a path
    inside a repo). Never passes --exit-code (TRAP-J4/J5) — the return
    value's JSON is the only source of truth this script trusts.

    `cwd` is jscpd's OWN process working directory, which is what TRAP-J2's
    ambient `.jscpd.json` auto-load keys off — NOT necessarily any of
    `targets` (jscpd accepts absolute target paths regardless of cwd,
    verified). Guarded here, against the actual invocation cwd, rather than
    at each call site, so the two can never drift apart.
    """
    config = config or _config()
    _guard_ambient_config(cwd)
    if not targets:
        _die("no scan targets resolved — refusing to report that as clean")
    try:
        out_dir_stat = out_dir.stat()
    except (OSError, ValueError) as exc:
        _die(f"could not inspect jscpd report directory {out_dir}: {exc}")
    if not stat.S_ISDIR(out_dir_stat.st_mode):
        _die(f"jscpd report directory does not exist: {out_dir}")
    cmd = _jscpd_command(exe, targets, out_dir, config, echo_console, formats)
    _execute_jscpd(cmd, cwd, targets, echo_console)
    return _load_report(
        out_dir / "jscpd-report.json",
        out_dir=out_dir,
        roots=targets,
        formats=formats,
    )


def _jscpd_command(
    exe: str,
    targets: list[Path],
    out_dir: Path,
    config: CloneScannerConfig,
    echo_console: bool,
    formats: tuple[str, ...] | None,
) -> list[str]:
    command = [
        exe,
        "--min-tokens",
        str(config.jscpd_min_tokens),
        "--min-lines",
        str(config.jscpd_min_lines),
        "--mode",
        config.jscpd_mode,
        # ^ jscpd's own upstream defaults, PINNED explicitly in pyproject.toml
        # rather than implied, so neither an ambient config (TRAP-J2) nor a
        # future upstream default change can silently drift them.
        "--ignore",
        ",".join(config.exclusions),
        "--absolute",
        "-r",
        "console,json" if echo_console else "json",
        "-o",
        str(out_dir),
    ]
    if config.jscpd_format_names_arg:
        command += ["--formats-names", config.jscpd_format_names_arg]
    if config.jscpd_format_exts_arg:
        command += ["--formats-exts", config.jscpd_format_exts_arg]
    if formats:
        command += ["--format", ",".join(formats)]
    if not echo_console:
        command += ["--silent", "--no-tips"]
    return [*command, *(str(target) for target in targets)]


def _execute_jscpd(
    command: list[str],
    cwd: Path,
    targets: list[Path],
    echo_console: bool,
) -> None:
    try:
        r = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=None if echo_console else subprocess.PIPE,
            stderr=subprocess.STDOUT if not echo_console else None,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        _die(f"jscpd timed out over {len(targets)} target(s) after 900s")
    except (OSError, UnicodeError) as exc:
        _die(f"could not execute {command[0]}: {exc}")
    if r.returncode != 0:
        tail = (r.stdout or "")[-2000:] if not echo_console else ""
        _die(
            f"jscpd exited {r.returncode} (TRAP-J4 means this is a real "
            f"failure, not 'clones found'): {tail}"
        )


def _resolved_path(path: str | Path, base: Path | None = None) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        if base is None:
            _die(f"relative scanner path has no trusted root: {path!r}")
        candidate = base / candidate
    try:
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        _die(f"could not resolve scanner path {path!r}: {exc}")


def _path_is_under(path: str | Path, root: Path) -> bool:
    """Return whether a reported path stays inside one scan target.

    jscpd is invoked with ``--absolute``. This primitive can resolve a relative
    path only for trusted internal callers; clone locations in a report are
    required to be absolute before reaching it. ``..`` traversal and symlinks
    that leave the target fail closed instead of becoming an untrusted key.
    """

    candidate = _resolved_path(path, root)
    trusted_root = _resolved_path(root)
    return candidate == trusted_root or candidate.is_relative_to(trusted_root)


def _require_under_root(
    path: str | Path,
    roots: Sequence[Path],
    label: str,
    *,
    require_absolute: bool = True,
) -> None:
    if require_absolute and not Path(path).is_absolute():
        _die(f"{label} is not absolute: {path}")
    if not roots or not any(_path_is_under(path, root) for root in roots):
        formatted = ", ".join(str(root) for root in roots) or "<none>"
        _die(f"{label} escapes its trusted root(s) {formatted}: {path}")


def _load_report(
    report_path: Path,
    out_dir: Path | None = None,
    roots: Sequence[Path] | None = None,
    formats: Sequence[str] | None = None,
) -> dict:
    try:
        report_stat = report_path.lstat()
    except FileNotFoundError:
        _die(f"jscpd exited 0 but wrote no report to {report_path}")
    except (OSError, ValueError) as exc:
        _die(f"could not inspect jscpd report {report_path}: {exc}")
    if stat.S_ISLNK(report_stat.st_mode) or not stat.S_ISREG(report_stat.st_mode):
        _die(f"jscpd report is not a regular file: {report_path}")
    if out_dir is not None:
        _require_under_root(
            report_path,
            (out_dir,),
            "jscpd report",
            require_absolute=False,
        )
    try:
        doc = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        _die(f"{report_path} was not valid JSON: {exc}")
    return _validate_report(doc, report_path, roots=roots, formats=formats)


def _validate_report(
    doc: object,
    report_path: Path,
    roots: Sequence[Path] | None = None,
    formats: Sequence[str] | None = None,
) -> dict:
    """Validate the report fields consumed by the census/diff code.

    A pinned binary should emit this shape, but a truncated or incompatible
    report must remain a gate error rather than becoming an uncaught traceback
    (or, worse, an empty clone set).
    """

    if not isinstance(doc, dict):
        _die(f"{report_path} did not contain a JSON object")
    duplicates = doc.get("duplicates")
    if not isinstance(duplicates, list):
        _die(f"{report_path} has no duplicates array")
    _validate_total_statistics(doc, report_path)
    for index, clone in enumerate(duplicates):
        _validate_clone(index, clone, report_path, roots=roots, formats=formats)
    return doc


def _validate_total_statistics(doc: dict, report_path: Path) -> None:
    statistics = doc.get("statistics")
    if not isinstance(statistics, dict):
        _die(f"{report_path} has malformed statistics")
    total = statistics.get("total")
    if not isinstance(total, dict):
        _die(f"{report_path} has malformed total statistics")
    _validate_integer_metrics(total, report_path)
    _validate_percentage(total, report_path)
    duplicates = doc.get("duplicates")
    if isinstance(duplicates, list) and total["clones"] != len(duplicates):
        _die(
            f"{report_path} total.clones ({total['clones']}) does not match "
            f"the duplicates array ({len(duplicates)})"
        )


def _validate_integer_metrics(total: dict, report_path: Path) -> None:
    for field in ("clones", "sources", "duplicatedLines", "lines"):
        if field not in total:
            _die(f"{report_path} has no total.{field}")
        value = total[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _die(f"{report_path} has invalid total.{field}")


def _validate_percentage(total: dict, report_path: Path) -> None:
    if "percentage" not in total:
        _die(f"{report_path} has no total.percentage")
    percentage = total["percentage"]
    if (
        isinstance(percentage, bool)
        or not isinstance(percentage, (int, float))
        or not 0.0 <= float(percentage) <= 100.0
    ):
        _die(f"{report_path} has invalid total.percentage")


_REQUIRED_CLONE_FIELDS = (
    "format",
    "fragment",
    "lines",
    "tokens",
    "firstFile",
    "secondFile",
)
_REPORT_LOCATION_SUFFIX = re.compile(
    r":(?P<format>[A-Za-z0-9_-]+)(?::(?P<start>[0-9]+)-(?P<end>[0-9]+))?$"
)


def _report_file_path(name: str, format_name: str) -> str:
    """Strip jscpd's virtual-format location suffix from a report path.

    jscpd appends ``:<format>`` (sometimes followed by
    ``:<start>-<end>``) to locations for formats such as Markdown.  The suffix
    is metadata, not part of the filesystem path; treating it as a filename
    makes a top-level file target look like it escaped its trusted root.  Keep
    ordinary paths (and paths with unrelated colon components) untouched.
    """

    match = _REPORT_LOCATION_SUFFIX.search(name)
    if match and match.group("format") == format_name:
        return name[: match.start()]
    return name


def _validate_clone(
    index: int,
    clone: object,
    report_path: Path,
    roots: Sequence[Path] | None = None,
    formats: Sequence[str] | None = None,
) -> None:
    clone = _clone_mapping(index, clone, report_path)
    format_name = _clone_format(index, clone, report_path, formats)
    _validate_clone_payload(index, clone, report_path)
    for side in ("firstFile", "secondFile"):
        _validate_clone_location(
            index,
            clone[side],
            side,
            report_path,
            roots=roots,
            format_name=format_name,
        )
    if _is_identical_clone_location(
        clone["firstFile"], clone["secondFile"], format_name
    ):
        _die(
            f"{report_path} duplicate {index} is a self-pair: both locations "
            "name the same file and line range"
        )


def _clone_mapping(index: int, clone: object, report_path: Path) -> dict:
    if not isinstance(clone, dict):
        _die(f"{report_path} duplicate {index} is not an object")
    missing = [field for field in _REQUIRED_CLONE_FIELDS if field not in clone]
    if missing:
        _die(f"{report_path} duplicate {index} is missing {', '.join(missing)}")
    return clone


def _clone_format(
    index: int,
    clone: dict,
    report_path: Path,
    formats: Sequence[str] | None,
) -> str:
    format_name = clone["format"]
    if not isinstance(format_name, str) or not format_name:
        _die(f"{report_path} duplicate {index} has an invalid format")
    if formats is not None and format_name not in formats:
        _die(
            f"{report_path} duplicate {index} has format {format_name!r} "
            "outside the requested jscpd format scope"
        )
    return format_name


def _validate_clone_payload(index: int, clone: dict, report_path: Path) -> None:
    if not isinstance(clone["fragment"], str) or not clone["fragment"]:
        _die(f"{report_path} duplicate {index} has no text fragment")
    if not _positive_clone_sizes(clone):
        _die(f"{report_path} duplicate {index} has invalid size fields")


def _positive_clone_sizes(clone: dict) -> bool:
    return all(
        not isinstance(clone[field], bool)
        and isinstance(clone[field], int)
        and clone[field] > 0
        for field in ("lines", "tokens")
    )


def _validate_clone_location(
    index: int,
    location: object,
    side: str,
    report_path: Path,
    roots: Sequence[Path] | None = None,
    format_name: str = "",
) -> None:
    location = _location_mapping(index, location, side, report_path)
    report_name = _report_file_path(location["name"], format_name)
    if roots is not None:
        _validate_location_root(
            index, side, location["name"], report_name, report_path, roots
        )
    _validate_location_lines(index, side, location, report_path)


def _location_mapping(
    index: int, location: object, side: str, report_path: Path
) -> dict:
    if (
        not isinstance(location, dict)
        or not isinstance(location.get("name"), str)
        or not location["name"]
    ):
        _die(f"{report_path} duplicate {index} has an invalid {side}")
    return location


def _validate_location_root(
    index: int,
    side: str,
    original_name: str,
    report_name: str,
    report_path: Path,
    roots: Sequence[Path],
) -> None:
    if not Path(report_name).is_absolute():
        _die(
            f"{report_path} duplicate {index} {side} path is not absolute: "
            f"{original_name!r}"
        )
    if not any(_path_is_under(report_name, root) for root in roots):
        _die(
            f"{report_path} duplicate {index} {side} path escapes the scan "
            f"roots: {original_name!r}"
        )


def _validate_location_lines(
    index: int, side: str, location: dict, report_path: Path
) -> None:
    for point in ("startLoc", "endLoc"):
        position = location.get(point)
        line = position.get("line") if isinstance(position, dict) else None
        if isinstance(line, bool) or not isinstance(line, int) or line <= 0:
            _die(f"{report_path} duplicate {index} has an invalid {side}.{point}.line")
    start = location["startLoc"]["line"]
    end = location["endLoc"]["line"]
    if end < start:
        _die(f"{report_path} duplicate {index} has a reversed {side} range")


def _is_identical_clone_location(first: dict, second: dict, format_name: str) -> bool:
    first_name = _report_file_path(first["name"], format_name)
    second_name = _report_file_path(second["name"], format_name)
    if Path(first_name).is_absolute() and Path(second_name).is_absolute():
        same_file = _canonical_scan_target(Path(first_name)) == _canonical_scan_target(
            Path(second_name)
        )
    else:
        same_file = os.path.normpath(first_name) == os.path.normpath(second_name)
    first_range = (
        first.get("start"),
        first.get("end"),
        first["startLoc"],
        first["endLoc"],
    )
    second_range = (
        second.get("start"),
        second.get("end"),
        second["startLoc"],
        second["endLoc"],
    )
    return same_file and first_range == second_range


def _print_stats(
    doc: dict,
    targets: list[Path],
    label: str,
    prune_directories: frozenset[str] | None = None,
) -> None:
    stats = doc.get("statistics", {}).get("total", {})
    n_files_handed = sum(1 for _ in _iter_files(targets, prune_directories))
    print(
        f"\njscpd gate [{label}]: {stats.get('clones', 0)} clone(s) found "
        f"across {stats.get('sources', 0)} scanned file(s) of a matched "
        f"format ({stats.get('duplicatedLines', 0)} of {stats.get('lines', 0)} "
        f"duplicated lines, {stats.get('percentage', 0):.2f}%)"
    )
    print(
        f"jscpd gate [{label}]: {n_files_handed} real file(s) existed "
        f"under the {len(targets)} scanned root(s), counted independently "
        f"of jscpd — cross-check against TRAP-J6 if 'sources' above looks "
        f"suspiciously low relative to this."
    )


def _iter_files(targets: list[Path], prune_directories: frozenset[str] | None = None):
    """Independent (no jscpd involved) file count for TRAP-J6's cross-check.
    MUST prune the same junk it would otherwise walk into: a bare `rglob`
    over a target that is not itself a decomposed git-repo root (e.g. the
    single `agents` root handed to a fleet-wide census, which contains 74
    UN-decomposed nested repos each with their own .venv/.git/node_modules)
    is not an approximation of jscpd's real corpus, it is a multi-million-
    file stat() storm that dwarfs jscpd's own (ignore-scoped) runtime.
    Verified: killed after 2m37s still running on a 7-root fleet census."""
    junk_names = _prune_names(prune_directories)
    for target in targets:
        yield from _files_under_target(target, junk_names)


def _files_under_target(target: Path, junk_names: frozenset[str]):
    try:
        target_stat = target.stat()
    except OSError as exc:
        _die(f"could not inspect scan target {target}: {exc}")
    if stat.S_ISREG(target_stat.st_mode):
        yield target
        return
    if not stat.S_ISDIR(target_stat.st_mode):
        return

    def onerror(error: OSError) -> NoReturn:
        _die(f"could not enumerate files under {target}: {error}")

    try:
        walker = os.walk(target, onerror=onerror)
        for dirpath, dirnames, filenames in walker:
            dirnames[:] = [
                name for name in dirnames if _kept_directory(name, junk_names)
            ]
            yield from (
                Path(dirpath) / filename for filename in filenames if filename != ".git"
            )
    except OSError as exc:
        _die(f"could not enumerate files under {target}: {exc}")


def _kept_directory(name: str, junk_names: frozenset[str]) -> bool:
    return name not in junk_names and not name.endswith(".egg-info")


@contextmanager
def _temporary_directory(prefix: str) -> Iterator[Path]:
    """Create a report directory and map cleanup failures to CANNOT RUN."""

    try:
        with tempfile.TemporaryDirectory(
            prefix=prefix,
            ignore_cleanup_errors=False,
        ) as directory:
            yield Path(directory)
    except (OSError, UnicodeError) as exc:
        _die(f"could not create or clean temporary report directory: {exc}")


def _worst_clones(doc: dict, n: int = 15) -> list[dict]:
    return sorted(doc.get("duplicates", []), key=lambda c: -c.get("lines", 0))[:n]


def _print_worst(doc: dict, n: int = 15) -> None:
    worst = _worst_clones(doc, n)
    if not worst:
        return
    print(f"\n  worst {len(worst)} clone(s) by line count:")
    for c in worst:
        f1, f2 = c["firstFile"], c["secondFile"]
        print(
            f"    {c['lines']:>5}L/{c['tokens']:>6}tok [{c['format']}]  "
            f"{f1['name']}:{f1['startLoc']['line']}-{f1['endLoc']['line']}"
            f"  <->  "
            f"{f2['name']}:{f2['startLoc']['line']}-{f2['endLoc']['line']}"
        )


def cmd_census(paths: list[str]) -> int:
    config = _config()
    exe = _resolve_jscpd(config)
    _check_version(exe, config)
    roots = [Path(p).resolve() for p in paths] if paths else [_AU_ROOT]
    targets = _expand_roots(roots, config.prune_directories)
    print(
        f"jscpd gate [census]: scanning {len(targets)} root(s): "
        + ", ".join(
            str(t.relative_to(_AU_ROOT)) if t.is_relative_to(_AU_ROOT) else str(t)
            for t in targets
        )
    )
    with _temporary_directory(prefix="cx-jscpd-census-") as tmp:
        doc = run_jscpd(
            exe,
            targets,
            tmp,
            _AU_ROOT,
            echo_console=False,
            config=config,
        )
        _print_stats(doc, targets, "census", config.prune_directories)
        _print_worst(doc)
    print(
        "\njscpd gate [census]: unconditional — this mode never fails on "
        "findings. Real numbers only, no baseline, nothing written to "
        "disk beyond this run's now-deleted temp dir."
    )
    return 0


# ─── enforce mode ────────────────────────────────────────────────────────


def _workdir() -> Path:
    """Where enforce mode's throwaway worktrees live. TRAP-J7 — defaults to
    a real-disk sibling of the repo, NOT tempfile.gettempdir()/tmpfs."""
    configured = _setting("CX_DUP_WORKDIR", "")
    d = (
        Path(configured).expanduser()
        if configured
        else _AU_ROOT.parent / ".cx-dup-enforce-tmp"
    )
    return _ensure_workdir(d)


def _ensure_workdir(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
        if not stat.S_ISDIR(path.stat().st_mode):
            _die(f"jscpd enforce workdir is not a directory: {path}")
        return path.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        _die(f"could not create jscpd enforce workdir {path}: {exc}")


def _run_git(args: list[str], **kw) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(_AU_ROOT),
            text=True,
            capture_output=True,
            env=sanitized_git_env(),
            **kw,
        )
    except (OSError, UnicodeError) as exc:
        _die(f"could not execute git ({' '.join(args)}): {exc}")


def _output_text(value: str | None) -> str:
    return value if value is not None else ""


def _git(args: list[str], **kw) -> str:
    r = _run_git(args, **kw)
    if r.returncode != 0:
        _die(f"`git {' '.join(args)}` failed: {_output_text(r.stderr).strip()}")
    return _output_text(r.stdout).strip()


def _clone_keys(doc: dict, worktree_root: Path) -> set[tuple]:
    """Content-addressed key per clone pair.

    Cross-file pairs use only their two worktree-relative paths, so a line
    shift elsewhere in either file (or the fact that before/after live in two
    different temp dirs) does not produce a spurious NEW/GONE pair.  An
    intra-file pair also carries each location's range: a set of filenames
    cannot distinguish two legitimate occurrences in the same file.

    The third tuple member always contains two ``(path, range)`` locations.
    ``range`` is ``None`` for cross-file pairs and a canonical, displayable
    range tuple for intra-file pairs.  Keeping both locations, rather than a
    set of paths, preserves same-file findings for the enforce renderer.
    """
    keys = set()
    for c in doc.get("duplicates", []):
        format_name = c["format"]
        first_name = _report_file_path(c["firstFile"]["name"], format_name)
        second_name = _report_file_path(c["secondFile"]["name"], format_name)
        f1 = _relative_clone_path(first_name, worktree_root)
        f2 = _relative_clone_path(second_name, worktree_root)
        same_file = f1 == f2
        first_location = _clone_location_key(
            c["firstFile"], f1, include_range=same_file
        )
        second_location = _clone_location_key(
            c["secondFile"], f2, include_range=same_file
        )
        digest = hashlib.sha256(
            c["fragment"].encode("utf-8", "surrogatepass")
        ).hexdigest()
        keys.add(
            (c["format"], digest, tuple(sorted((first_location, second_location))))
        )
    return keys


def _clone_location_key(
    location: dict, path: str, *, include_range: bool
) -> tuple[str, tuple[str, int, int] | None]:
    if not include_range:
        return path, None
    range_payload = {
        "start": location.get("start"),
        "end": location.get("end"),
        "startLoc": location["startLoc"],
        "endLoc": location["endLoc"],
    }
    range_key = json.dumps(
        range_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        path,
        (
            range_key,
            location["startLoc"]["line"],
            location["endLoc"]["line"],
        ),
    )


def _format_clone_location(location: object) -> str:
    """Render one canonical clone location, failing closed on bad identity."""

    path, range_key = _clone_location_parts(location)
    if range_key is None:
        return path
    start_line, end_line = _clone_range_lines(range_key, location)
    return f"{path}:{start_line}-{end_line}"


def _clone_location_parts(location: object) -> tuple[str, object]:
    if (
        not isinstance(location, tuple)
        or len(location) != 2
        or not isinstance(location[0], str)
        or not location[0]
    ):
        _die(f"jscpd enforce produced an invalid clone location: {location!r}")
    path, range_key = location
    return path, range_key


def _clone_range_lines(range_key: object, location: object) -> tuple[int, int]:
    if (
        not isinstance(range_key, tuple)
        or len(range_key) != 3
        or not isinstance(range_key[0], str)
        or not isinstance(range_key[1], int)
        or isinstance(range_key[1], bool)
        or not isinstance(range_key[2], int)
        or isinstance(range_key[2], bool)
    ):
        _die(f"jscpd enforce produced an invalid clone range: {location!r}")
    _range_identity, start_line, end_line = range_key
    return start_line, end_line


def _format_clone_pair(locations: object) -> str:
    """Render both sides of a clone finding without collapsing same-file pairs."""

    if not isinstance(locations, tuple) or len(locations) != 2:
        _die(f"jscpd enforce produced an invalid clone pair: {locations!r}")
    first, second = locations
    return f"{_format_clone_location(first)}  <->  {_format_clone_location(second)}"


def _relative_clone_path(name: str, worktree_root: Path) -> str:
    """Return a trusted worktree-relative report path or fail closed."""

    candidate = _resolved_path(name, worktree_root)
    root = _resolved_path(worktree_root)
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        _die(f"jscpd report path escapes worktree {worktree_root}: {name!r}")
    if not relative.parts:
        _die(f"jscpd report path names the worktree root: {name!r}")
    return relative.as_posix()


def _scan_worktree(
    exe: str,
    root: Path,
    label: str,
    config: CloneScannerConfig,
) -> tuple[dict, set[tuple]]:
    return _scan_worktree_impl(exe, root, label, config)


def _scan_worktree_impl(
    exe: str,
    root: Path,
    label: str,
    config: CloneScannerConfig,
) -> tuple[dict, set[tuple]]:
    roots = _expand_roots([root], config.prune_directories)
    if not roots:
        _die(f"jscpd gate [{label}] resolved no scan roots")
    # The differential gate is intentionally bounded to reviewed code and
    # non-code formats. Handing jscpd the already filtered file list also
    # avoids walking nested repositories that its --ignore option would merely
    # discard after reading.
    if not _has_in_scope_file(root, roots, config):
        # A change may delete the last file in this scope. Empty is a valid
        # after-state; it is not a scanner failure or a clean baseline file.
        print(f"jscpd gate [{label}]: no in-scope files")
        return {"duplicates": []}, set()
    # Pass the bounded top-level roots rather than thousands of individual
    # files. Large repositories can otherwise exceed ARG_MAX before jscpd
    # starts. The pinned format list and exclusions retain the same corpus.
    targets = _in_scope_roots(root, roots, config)
    if not targets:
        _die(f"jscpd gate [{label}] resolved no scan roots")
    with _temporary_directory(prefix=f"cx-jscpd-{label}-") as tmp:
        doc = run_jscpd(
            exe,
            targets,
            tmp,
            root,
            echo_console=False,
            config=config,
            formats=config.jscpd_diff_formats,
        )
    _print_stats(doc, targets, label, config.prune_directories)
    return doc, _clone_keys(doc, root)


def _has_in_scope_file(
    root: Path, roots: list[Path], config: CloneScannerConfig
) -> bool:
    for path in _iter_files(roots, config.prune_directories):
        if is_jscpd_diff_path(path, config) and not is_excluded_path(
            path.relative_to(root), config.exclusions
        ):
            return True
    return False


def _in_scope_roots(
    root: Path, roots: list[Path], config: CloneScannerConfig
) -> list[Path]:
    return [
        path
        for path in roots
        if not is_excluded_path(path.relative_to(root), config.exclusions)
    ]


def _changed_enforce_paths(base_ref: str, config: CloneScannerConfig) -> list[str]:
    """Select changed paths in the reviewed jscpd format scope."""

    raw = _git(
        [
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            f"{base_ref}...HEAD",
        ]
    )
    paths = []
    for line in raw.splitlines():
        rel = line.strip().replace("\\", "/")
        if (
            rel
            and is_jscpd_diff_path(rel, config)
            and not is_excluded_path(rel, config.exclusions)
        ):
            paths.append(rel)
    return sorted(set(paths))


def cmd_enforce(base_ref: str) -> int:
    config = _config()
    changed_paths = _changed_enforce_paths(base_ref, config)
    if not changed_paths:
        print(
            "jscpd gate [enforce]: no changed file in the reviewed code, "
            "template, or configuration scope"
        )
        return 0

    exe = _resolve_jscpd(config)
    _check_version(exe, config)

    head_sha = _git(["rev-parse", "HEAD"])
    base_sha = _git(["rev-parse", base_ref])
    merged_tree = _git(["merge-tree", "--write-tree", base_ref, "HEAD"])
    # Wrap the merge-tree result in a throwaway, UNREFERENCED commit object
    # (no ref/branch created) purely so `git worktree add` has a commit-ish
    # to check out. This is a normal git-plumbing write of loose objects —
    # NOT a stash (AGENTS.md forbids `git stash` in a shared multi-worktree
    # repo because refs/stash is repo-wide; this touches no ref at all) and
    # not EnterWorktree (a real `git worktree add --detach` is the
    # CLAUDE.md-endorsed way to get a throwaway clean tree).
    synth_commit = _git(
        [
            "commit-tree",
            merged_tree,
            "-p",
            base_sha,
            "-p",
            head_sha,
            "-m",
            "CX-DUP-ENFORCE throwaway snapshot — not a ref, do not reuse",
        ]
    )

    uid = uuid.uuid4().hex
    workdir = _workdir()
    before_wt = workdir / f"cx-jscpd-before-{uid}"
    after_wt = workdir / f"cx-jscpd-after-{uid}"
    try:
        _git(["worktree", "add", "--detach", str(before_wt), base_sha])
        _git(["worktree", "add", "--detach", str(after_wt), synth_commit])

        print(
            f"jscpd gate [enforce]: BEFORE = {base_ref} ({base_sha[:10]}), "
            f"AFTER = HEAD ({head_sha[:10]}) merged onto {base_ref} live "
            f"(tree {merged_tree[:10]}, via git merge-tree --write-tree)"
        )

        _, before_keys = _scan_worktree(exe, before_wt, "enforce-before", config)
        _, after_keys = _scan_worktree(exe, after_wt, "enforce-after", config)
    finally:
        _cleanup_throwaway_worktrees((before_wt, after_wt))

    new_pairs = after_keys - before_keys
    gone_pairs = before_keys - after_keys
    print(
        f"\njscpd gate [enforce]: {len(before_keys)} pre-existing clone "
        f"pair(s), {len(after_keys)} clone pair(s) after this change, "
        f"{len(new_pairs)} NEW, {len(gone_pairs)} resolved."
    )

    if not new_pairs:
        print(
            "jscpd gate [enforce]: PASS — this change introduces no new "
            "duplication. Pre-existing duplication is reported above, "
            "not gated (no ratchet, no baseline)."
        )
        return 0

    print(
        f"\njscpd gate [enforce]: FAIL — {len(new_pairs)} NEW duplicate "
        "pair(s) introduced by this change:"
    )
    for fmt, digest, locations in sorted(new_pairs, key=lambda k: str(k[2])):
        print(f"    [{fmt}] {_format_clone_pair(locations)}  (fragment {digest[:12]})")
    print(
        "\nRemove the duplication, or if it is a deliberate, reviewed "
        "exception, say so in the PR description — this gate has no "
        "suppression mechanism by design (no # noqa / baseline entry can "
        "silence it)."
    )
    return 1


def _remove_throwaway_worktree(path: Path) -> None:
    try:
        result = subprocess.run(
            ["git", "worktree", "remove", "--force", str(path)],
            cwd=str(_AU_ROOT),
            env=sanitized_git_env(),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, UnicodeError) as exc:
        _die(f"could not remove throwaway worktree {path}: {exc}")
    if result.returncode != 0:
        _die(
            f"could not remove throwaway worktree {path}: "
            f"{(result.stderr or '').strip()}"
        )
    try:
        remains = os.path.lexists(path)
    except OSError as exc:
        _die(f"could not verify throwaway worktree cleanup {path}: {exc}")
    if remains:
        _die(f"throwaway worktree cleanup left path behind: {path}")


def _cleanup_throwaway_worktrees(paths: Sequence[Path]) -> None:
    """Attempt every cleanup and fail closed if any worktree remains."""

    failures: list[str] = []
    for path in paths:
        try:
            if not os.path.lexists(path):
                continue
            _remove_throwaway_worktree(path)
        except SystemExit as exc:
            failures.append(f"{path} (exit {exc.code})")
        except Exception as exc:  # pragma: no cover - defensive cleanup boundary
            failures.append(f"{path} ({exc})")
    if failures:
        _die("throwaway worktree cleanup failed: " + "; ".join(failures))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="mode")

    p_census = sub.add_parser(
        "census", help="unconditional real-number report (default)"
    )
    p_census.add_argument("paths", nargs="*", default=[])

    p_enforce = sub.add_parser(
        "enforce", help="diff-scoped: fail only on NEW duplication"
    )
    p_enforce.add_argument("--base-ref", default=_setting("CX_DUP_BASE_REF", "main"))

    # Bare invocation (no subcommand) behaves as `census` with no paths —
    # census is always the safe default; enforce must be requested explicitly.
    args, extra = ap.parse_known_args()
    if args.mode is None:
        args = ap.parse_args(["census", *extra])

    if args.mode == "census":
        return cmd_census(args.paths)
    return cmd_enforce(args.base_ref)


if __name__ == "__main__":
    sys.exit(main())
