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

Exit codes: 0 clean (or census mode, always), 1 NEW duplication in `enforce`
mode, 2 CANNOT RUN. A gate that could not run has NOT found nothing.

────────────────────────────────────────────────────────────────────────────
Traps verified in THIS workspace 2026-08-28, all defended below:

 TRAP-J1  `jscpd --version` prints `cpd 5.0.16`, NOT `jscpd 5.0.16` — the
          published crate/binary is literally named `cpd`. A version guard
          that checks for the string "jscpd" would CANNOT-RUN forever on a
          perfectly good install. EXPECT_VERSION below is the real string.

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

 TRAP-J6  "Files analyzed" in jscpd's own summary table only counts files
          that participate in an ALREADY-detected clone (i.e. cleared
          min-tokens/min-lines) — NOT the count of files actually scanned.
          Two 100%-identical 12-line files with jscpd's own default
          `min-tokens=50` reported "Files analyzed: 0" — indistinguishable
          at a glance from TRAP-J3's "walked nothing" failure. This script
          therefore ALSO reports the raw file count it handed to jscpd
          (independent of jscpd's own table) so a real zero can be told
          apart from a threshold miss or a walk that found nothing.
────────────────────────────────────────────────────────────────────────────

Usage::

    python scripts/check_duplication.py                    # census, this repo
    python scripts/check_duplication.py census PATH [PATH...]
    python scripts/check_duplication.py enforce [--base-ref main]

Install jscpd (pin the exact version; do NOT let a hook install it)::

    npm install -g jscpd@5.0.16
    # then either put the `jscpd` bin on PATH, or:
    export JSCPD_BIN=/path/to/jscpd
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

_AU_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _git_subprocess_env import (  # noqa: E402
    sanitized_git_env,
    strip_inherited_git_repository_env,
)

# BUG-180-class defense: this script shells out to `git` (enforce mode). See
# scripts/_git_subprocess_env.py — a real `git commit`/merge-queue invocation
# exports GIT_DIR/GIT_INDEX_FILE into every hook subprocess it runs, and
# `-C <dir>` does NOT override them.
strip_inherited_git_repository_env()

EXPECT_VERSION = "cpd 5.0.16"  # TRAP-J1 — see module docstring.

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
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    ".pytest_tmp",
    ".tox",  # tool caches / test scratch.
    "site-packages",
    ".eggs",
    "vendor",
    "htmlcov",  # vendored/3rd-party or
    # coverage HTML output.
    "__snapshots__",  # snapshot-test fixtures: duplication here is BY
    # DESIGN (a snapshot IS a copy of expected output) —
    # scanning it only produces noise, per the brief's
    # agreed exclusion list.
}


def _die(msg: str) -> None:
    print(f"jscpd gate: CANNOT RUN: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _resolve_jscpd() -> str:
    """Find jscpd WITHOUT consulting any package index at hook time — the
    same discipline as scripts/check_complexity.py's `_resolve_cccc`: a hook
    that resolves a tool from an index at hook time is how a previous fleet
    sweep shipped a gate that could not pass anywhere (69/226 push failures).
    """
    env = os.environ.get("JSCPD_BIN")
    if env and Path(env).is_file():
        return env
    for cand in (Path.home() / ".local/bin/jscpd", Path("/usr/local/bin/jscpd")):
        if cand.is_file():
            return str(cand)
    found = shutil.which("jscpd")
    if found:
        return found
    _die(
        "`jscpd` not found. Looked at $JSCPD_BIN, ~/.local/bin/jscpd, "
        "/usr/local/bin/jscpd and $PATH. Install the pinned version with "
        "`npm install -g jscpd@5.0.16` and either put it on PATH or set "
        "JSCPD_BIN. This gate never installs anything itself."
    )


def _check_version(exe: str) -> None:
    try:
        r = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _die(f"could not run `{exe} --version`: {exc}")
    got = r.stdout.strip()
    if got != EXPECT_VERSION:
        _die(
            f"version drift: want '{EXPECT_VERSION}', got '{got}' (see "
            "TRAP-J1 — the version string is 'cpd X.Y.Z', not 'jscpd "
            "X.Y.Z'). Thresholds/behaviour are calibrated per version; a "
            "different build may add, rename, or silently change defaults."
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
        if p.exists():
            _die(
                f"{p} exists. jscpd auto-loads this from the CURRENT "
                "WORKING DIRECTORY with no --config flag (TRAP-J2) and it "
                "silently overrides thresholds this script does not "
                "control. Delete it — thresholds are pinned on the command "
                "line, in this script, in git history, in the open."
            )


# File-level glob exclusions (`--ignore`). Defense-in-depth ONLY (TRAP-J3
# proved this does not prune the walk, so it does not save time by itself);
# _repo_scan_targets's root decomposition is what keeps the walk fast. This
# still matters for junk nested one level *inside* a kept top-level dir.
_IGNORE_GLOBS = [
    "**/.git/**",  # belt-and-suspenders on TRAP-J3.
    "**/node_modules/**",  # npm dependency trees.
    "**/target/**",
    "**/target-isolated/**",  # cargo build output.
    "**/dist/**",
    "**/build/**",  # packaging output.
    "**/.venv/**",
    "**/venv/**",  # interpreter trees.
    "**/__pycache__/**",
    "**/*.lock",  # lockfiles: machine-generated,
    # deliberately repetitive, not
    # source a human wrote twice.
    "**/*.snap",
    "**/__snapshots__/**",  # snapshot-test fixtures — see
    # _JUNK_DIR_NAMES's __snapshots__
    # entry for the rationale.
    "**/__generated__/**",
    "**/generated/**",
    "**/openapi_client/**",
    "**/graphql_client/**",
    # generated GraphQL/OpenAPI clients — machine-generated from a schema;
    # any "duplication" here is a property of the schema, not a human
    # decision, and the fix (if any) is upstream of this gate.
]


def _repo_scan_targets(root: Path) -> list[Path]:
    """Safe scan targets for one directory. TRAP-J3's actual fix: never hand
    jscpd a path that is itself a git repo root. Decomposes into `root`'s
    own top-level children, dropping dot-dirs and _JUNK_DIR_NAMES. Falls
    back to `[root]` for a leaf directory with no such children (e.g.
    pointing this gate directly at a small module dir)."""
    if not root.is_dir():
        return [root]
    kept = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("."):
            continue
        if name in _JUNK_DIR_NAMES or name.endswith(".egg-info"):
            continue
        kept.append(child)
    return kept or [root]


def _expand_roots(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        out.extend(_repo_scan_targets(p))
    return out


def run_jscpd(
    exe: str,
    targets: list[Path],
    out_dir: Path,
    cwd: Path,
    *,
    echo_console: bool = True,
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
    _guard_ambient_config(cwd)
    if not targets:
        _die("no scan targets resolved — refusing to report that as clean")
    cmd = [
        exe,
        "--min-tokens",
        "50",
        "--min-lines",
        "5",
        "--mode",
        "mild",
        # ^ jscpd's own upstream defaults, PINNED explicitly rather than
        # implied, so neither an ambient config (TRAP-J2) nor a future
        # upstream default change can silently drift them.
        "--ignore",
        ",".join(_IGNORE_GLOBS),
        "--absolute",
        "-r",
        "console,json" if echo_console else "json",
        "-o",
        str(out_dir),
    ]
    if not echo_console:
        cmd += ["--silent", "--no-tips"]
    cmd += [str(t) for t in targets]
    try:
        r = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=None if echo_console else subprocess.PIPE,
            stderr=subprocess.STDOUT if not echo_console else None,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        _die(f"jscpd timed out over {len(targets)} target(s) after 900s")
    except OSError as exc:
        _die(f"could not execute {exe}: {exc}")
    if r.returncode != 0:
        tail = (r.stdout or "")[-2000:] if not echo_console else ""
        _die(
            f"jscpd exited {r.returncode} (TRAP-J4 means this is a real "
            f"failure, not 'clones found'): {tail}"
        )
    report_path = out_dir / "jscpd-report.json"
    if not report_path.exists():
        _die(f"jscpd exited 0 but wrote no report to {report_path}")
    try:
        doc = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _die(f"{report_path} was not valid JSON: {exc}")
    return doc


def _print_stats(doc: dict, targets: list[Path], label: str) -> None:
    stats = doc.get("statistics", {}).get("total", {})
    n_files_handed = sum(1 for _ in _iter_files(targets))
    print(
        f"\njscpd gate [{label}]: {stats.get('clones', 0)} clone(s) across "
        f"{stats.get('sources', 0)} file(s) with reported duplication "
        f"({stats.get('duplicatedLines', 0)} of {stats.get('lines', 0)} "
        f"lines, {stats.get('percentage', 0):.2f}%)"
    )
    print(
        f"jscpd gate [{label}]: {n_files_handed} real file(s) existed "
        f"under the {len(targets)} scanned root(s) — cross-check against "
        f"TRAP-J6 if 'sources' above looks suspiciously low."
    )


def _iter_files(targets: list[Path]):
    for t in targets:
        if t.is_file():
            yield t
        elif t.is_dir():
            yield from (p for p in t.rglob("*") if p.is_file())


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
    exe = _resolve_jscpd()
    _check_version(exe)
    roots = [Path(p).resolve() for p in paths] if paths else [_AU_ROOT]
    targets = _expand_roots(roots)
    print(
        f"jscpd gate [census]: scanning {len(targets)} root(s): "
        + ", ".join(
            str(t.relative_to(_AU_ROOT)) if t.is_relative_to(_AU_ROOT) else str(t)
            for t in targets
        )
    )
    with tempfile.TemporaryDirectory(prefix="cx-jscpd-census-") as tmp:
        doc = run_jscpd(exe, targets, Path(tmp), _AU_ROOT)
        _print_stats(doc, targets, "census")
        _print_worst(doc)
    print(
        "\njscpd gate [census]: unconditional — this mode never fails on "
        "findings. Real numbers only, no baseline, nothing written to "
        "disk beyond this run's now-deleted temp dir."
    )
    return 0


# ─── enforce mode ────────────────────────────────────────────────────────


def _git(args: list[str], **kw) -> str:
    r = subprocess.run(
        ["git", *args],
        cwd=str(_AU_ROOT),
        text=True,
        capture_output=True,
        env=sanitized_git_env(),
        **kw,
    )
    if r.returncode != 0:
        _die(f"`git {' '.join(args)}` failed: {r.stderr.strip()}")
    return r.stdout.strip()


def _clone_keys(doc: dict, worktree_root: Path) -> set[tuple]:
    """Content-addressed key per clone pair: (format, sha256(fragment),
    frozenset of the two files' paths RELATIVE to their worktree root).
    Relative + content-hashed so a trivial line-shift elsewhere in either
    file (or the fact that before/after live in two different temp dirs)
    never produces a spurious NEW/GONE pair."""
    keys = set()
    for c in doc.get("duplicates", []):
        try:
            f1 = Path(c["firstFile"]["name"]).relative_to(worktree_root)
            f2 = Path(c["secondFile"]["name"]).relative_to(worktree_root)
        except ValueError:
            f1, f2 = c["firstFile"]["name"], c["secondFile"]["name"]
        digest = hashlib.sha256(
            c["fragment"].encode("utf-8", "surrogatepass")
        ).hexdigest()
        keys.add((c["format"], digest, frozenset({str(f1), str(f2)})))
    return keys


def _scan_worktree(exe: str, root: Path, label: str) -> tuple[dict, set[tuple]]:
    targets = _expand_roots([root])
    with tempfile.TemporaryDirectory(prefix=f"cx-jscpd-{label}-") as tmp:
        doc = run_jscpd(exe, targets, Path(tmp), root)
    _print_stats(doc, targets, label)
    return doc, _clone_keys(doc, root)


def cmd_enforce(base_ref: str) -> int:
    exe = _resolve_jscpd()
    _check_version(exe)

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

    uid = uuid.uuid4().hex[:8]
    before_wt = Path(tempfile.gettempdir()) / f"cx-jscpd-before-{uid}"
    after_wt = Path(tempfile.gettempdir()) / f"cx-jscpd-after-{uid}"
    try:
        _git(["worktree", "add", "--detach", str(before_wt), base_sha])
        _git(["worktree", "add", "--detach", str(after_wt), synth_commit])

        print(
            f"jscpd gate [enforce]: BEFORE = {base_ref} ({base_sha[:10]}), "
            f"AFTER = HEAD ({head_sha[:10]}) merged onto {base_ref} live "
            f"(tree {merged_tree[:10]}, via git merge-tree --write-tree)"
        )

        _, before_keys = _scan_worktree(exe, before_wt, "enforce-before")
        _, after_keys = _scan_worktree(exe, after_wt, "enforce-after")
    finally:
        for wt in (before_wt, after_wt):
            if wt.exists():
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(wt)],
                    cwd=str(_AU_ROOT),
                    env=sanitized_git_env(),
                    capture_output=True,
                    text=True,
                )

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
    for fmt, digest, files in sorted(new_pairs, key=lambda k: sorted(k[2])):
        a, b = sorted(files)
        print(f"    [{fmt}] {a}  <->  {b}  (fragment {digest[:12]})")
    print(
        "\nRemove the duplication, or if it is a deliberate, reviewed "
        "exception, say so in the PR description — this gate has no "
        "suppression mechanism by design (no # noqa / baseline entry can "
        "silence it)."
    )
    return 1


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
    p_enforce.add_argument(
        "--base-ref", default=os.environ.get("CX_DUP_BASE_REF", "main")
    )

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
