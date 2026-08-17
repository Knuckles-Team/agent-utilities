#!/usr/bin/env python3
"""Guard against BUG-067: pre-commit's stash mechanism parks uncommitted work in
an out-of-tree cache file, and a killed run leaves it there -- silently.

**The mechanism.** ``pre-commit`` clears unstaged changes before running hooks
(so hooks only see the staged diff) by writing them to
``$PRE_COMMIT_HOME/patch<timestamp>-<pid>`` and running
``git checkout -- .`` to blank the working tree, then restoring the patch in a
``finally`` block once hooks finish (see the vendored
``pre_commit/staged_files_only.py:_unstaged_changes_cleared``). If the process
is killed (OOM, ``timeout``, a stalled agent's session ending) between the
checkout and the restore, the working tree has ALREADY been blanked and the
ONLY surviving copy of that work is the patch file sitting in what looks like
disposable cache.

**Confirmed near-miss (BUG-067).** A killed ``pre-commit`` run left its entire
uncommitted diff in ``~/.cache/pre-commit/patch*``. An orchestrator audit
found 19 such patch files (332 KB) from a single session; each had to be
hand-classified with ``git apply --check``/``--check --reverse`` against the
tree to determine whether it was already-in-tree, stale/superseded, or still
live. This script automates exactly that triage so it never again depends on
someone remembering to do it by hand before a cache-reclaim sweep runs
``rm -rf`` (or equivalent) over ``$PRE_COMMIT_HOME``.

**Verdict, deliberately conservative.** This tool never deletes or classifies
anything as safe-to-remove -- BUG-067's own remediation note is explicit:
"treat ~/.cache/pre-commit/patch* as live working-tree state, never as
disposable cache." It only reports, and it FAILS (exit 1) the moment any
patch looks like it would apply cleanly against the target tree AND is not
already present there -- the exact "would destroy a worker's uncommitted
diff outright" scenario BUG-067 recorded as a near-miss. Run it before any
``$PRE_COMMIT_HOME``/``~/.cache/pre-commit`` reclaim, ideally from every repo
that shares the cache (it is a per-user cache, not per-repo)::

    python3 scripts/check_pre_commit_patch_safety.py
    python3 scripts/check_pre_commit_patch_safety.py --repository-root DIR --patch-dir DIR
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

PATCH_NAME_RE = re.compile(r"^patch\d+-\d+$")


def _default_patch_dir() -> Path:
    env_home = os.environ.get("PRE_COMMIT_HOME")
    if env_home:
        return Path(env_home)
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "pre-commit"
    return Path.home() / ".cache" / "pre-commit"


@dataclass(frozen=True)
class PatchVerdict:
    path: Path
    classification: str  # "already-in-tree" | "stale-or-foreign" | "LIVE-UNAPPLIED"
    detail: str


def _git_apply_check(repo_root: Path, patch: Path, *, reverse: bool) -> bool:
    args = ["git", "-C", str(repo_root), "apply", "--check"]
    if reverse:
        args.append("--reverse")
    args.append(str(patch))
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return result.returncode == 0


def classify(repo_root: Path, patch: Path) -> PatchVerdict:
    applies_forward = _git_apply_check(repo_root, patch, reverse=False)
    if applies_forward:
        return PatchVerdict(
            patch,
            "LIVE-UNAPPLIED",
            f"applies cleanly against {repo_root} and is NOT yet present in the "
            "tree -- this is uncommitted work that would be DESTROYED if this "
            f"file were deleted. Recover with: git -C {repo_root} apply "
            f"{patch}",
        )
    applies_reverse = _git_apply_check(repo_root, patch, reverse=True)
    if applies_reverse:
        return PatchVerdict(
            patch,
            "already-in-tree",
            f"reverse-applies cleanly against {repo_root} -- its content is "
            "already present in the working tree (safe with respect to THIS "
            "repo).",
        )
    return PatchVerdict(
        patch,
        "stale-or-foreign",
        f"applies neither forward nor reverse against {repo_root} -- either "
        "superseded content, or a patch from a DIFFERENT repository that "
        "shares this same per-user cache. NOT proof of safety: re-run against "
        "every other repo checked out on this host before treating it as "
        "disposable.",
    )


def find_patches(patch_dir: Path) -> list[Path]:
    if not patch_dir.is_dir():
        return []
    return sorted(
        path
        for path in patch_dir.iterdir()
        if path.is_file() and PATCH_NAME_RE.match(path.name)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-pre-commit-patch-safety")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repo to classify patches against (default: this repo's root)",
    )
    parser.add_argument(
        "--patch-dir",
        type=Path,
        default=None,
        help="override the pre-commit patch cache dir (default: $PRE_COMMIT_HOME, "
        "else $XDG_CACHE_HOME/pre-commit, else ~/.cache/pre-commit)",
    )
    args = parser.parse_args(argv)

    patch_dir = args.patch_dir or _default_patch_dir()
    patches = find_patches(patch_dir)

    if not patches:
        print(f"check-pre-commit-patch-safety: no patch files in {patch_dir} — clean.")
        return 0

    print(
        f"check-pre-commit-patch-safety: {len(patches)} patch file(s) in "
        f"{patch_dir} — classifying against {args.repository_root}\n"
        "(BUG-067: these are a killed pre-commit run's parked uncommitted "
        "diffs, never disposable cache — see this script's module docstring.)"
    )

    live: list[PatchVerdict] = []
    for patch in patches:
        verdict = classify(args.repository_root, patch)
        print(f"  - {patch.name}: {verdict.classification}\n      {verdict.detail}")
        if verdict.classification == "LIVE-UNAPPLIED":
            live.append(verdict)

    if live:
        print(
            f"\nFAILED: {len(live)} patch(es) represent live, unapplied "
            "uncommitted work. Recover each with the `git apply` command shown "
            "above BEFORE touching this cache directory in any way."
        )
        return 1

    print(
        "\nNo patch classifies as live-unapplied against "
        f"{args.repository_root}. Any 'stale-or-foreign' entries above are "
        "still NOT proof of safety for other repos sharing this cache — "
        "re-run with --repository-root pointed at each one before reclaiming."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
