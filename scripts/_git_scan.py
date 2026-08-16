#!/usr/bin/env python3
"""Shared ambient-git-env-safe tracked-file discovery for the gate scripts.

CONCEPT:AU-OS.governance.fail-closed-degraded-read

BUG-043 (root cause found by the GOC-70 liveness investigation) / BUG-174
(this sweep, `plans/graph-os-completion-program/BUG-LEDGER.md`): git sets
``GIT_DIR``/``GIT_INDEX_FILE`` in the environment of *every* hook
subprocess (``git commit``, incl. concluding a merge). Under that ambient
env, ``git -C <subdir> ls-files -- <pattern>`` silently reverts its
output-path base from "relative to ``-C``'s target" to "relative to the
ambient work-tree root" -- while ``-C``'s own
``rev-parse --show-toplevel``/``--show-prefix`` simultaneously and
inconsistently claim ``<subdir>`` itself IS the toplevel with an empty
prefix. Confirmed reproducible directly:

    GIT_DIR=<repo>/.git GIT_INDEX_FILE=<repo>/.git/index \\
      git -C agent_utilities ls-files -- '*.py' | head -1
    # plain:  __init__.py
    # ambient: agent_utilities/__init__.py   (doubled once reconstructed)

A caller that reconstructs absolute paths as ``target / line`` then doubles
the ``target``-relative segment onto a path that never exists on disk;
every reconstructed path fails ``is_file()``, and the *pre-filter* list was
non-empty (git returned real lines, just wrongly rooted), so a naive
"if tracked: return filtered" shape returns that EMPTY filtered list
directly instead of falling through to its own -- correct -- ``rglob``
fallback. The gate then measures an empty universe and reports a
confident, wrong "all clear": the exact "component that cannot do its job
returns a value its caller reads as all clear" shape AGENTS.md's
*Fail closed* section prohibits.

This exact ~20-line helper was independently copy-pasted into roughly
twenty gate scripts before being extracted here -- the duplication IS the
bug (one mistake, cloned ~20 times, each copy silently wrong under the same
ambient condition). Anchor at the caller's own ``root`` (computed once via
pure ``Path`` math -- e.g. ``Path(__file__).resolve().parents[N]`` --
immune to git's ambient-env behavior, never derived from git itself) with a
pathspec scoped to ``target``'s position under it, and reconstruct every
returned path as ``root / line``: correct under both the plain and the
ambient-env-polluted invocation shapes. Verified against both directly in
``tests/unit/scripts/test_git_scan.py``.

A handful of gates (``check_entrypoint_engine_construction.py``,
``check_coupling.py``, the fleet-wide ``check_fleet_tls_policy.py`` /
``check_fleet_privacy_policy.py``) scan a subtree that belongs to a
DIFFERENT repository than the one running the hook -- e.g. an
agent-utilities pre-commit hook inspecting ``../geniusbot/geniusbot``.
There, "the caller's own root" is not this repo's ``ROOT`` at all, and
passing it anyway would hit the ``target.relative_to(root)`` ``ValueError``
fallback -- which reproduces the ORIGINAL bug's `-C target` shape 1:1.
Confirmed empirically to be even worse cross-repo: under agent-utilities'
ambient ``GIT_DIR``, ``git -C <geniusbot path> ls-files`` doesn't error --
it silently returns agent-utilities' OWN tracked file list mislabeled as
geniusbot's. :func:`repo_root_of` finds the correct anchor for a foreign
subtree the same ambient-env-immune way: pure filesystem ``.git``
detection, walking upward from the target, never invoking git at all.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def repo_root_of(target: Path) -> Path | None:
    """The nearest ancestor of ``target`` (inclusive) that looks like a git
    repository root, found by pure filesystem inspection -- never by asking
    git itself, since git's own discovery is exactly the ambient-env-
    dependent signal this module exists to route around. Returns ``None``
    if no ancestor has a ``.git`` entry (file or directory -- a linked
    worktree's ``.git`` is a file, not a directory).
    """
    for candidate in (target, *target.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def tracked_or_walked(target: Path, *patterns: str, root: Path) -> list[Path]:
    """Files under ``target`` (optionally matching ``patterns``), preferring
    the git-tracked set over a raw filesystem walk.

    A raw ``rglob`` also picks up gitignored, generated build output, which
    can carry a stale copy of an already-fixed file and reintroduce a
    violation this gate already cleared in real source. Falls back to a
    filesystem walk when the ``git`` invocation fails outright (not a git
    working tree, no ``git`` binary) or when ``target`` is not inside
    ``root`` at all (e.g. a synthetic test fixture rooted at its own
    ``tmp_path``, with no relation to ``root``).

    ``root`` MUST be the caller's own repository root, computed via pure
    ``Path`` math -- see the module docstring for why. Keyword-only so a
    caller can never accidentally pass it positionally where a pattern was
    meant.

    With no ``patterns``, matches every tracked/walked file under
    ``target``.
    """
    try:
        rel = target.relative_to(root)
        anchor = root
        prefix = "" if str(rel) == "." else f"{rel.as_posix()}/"
        pathspecs = [f"{prefix}{p}" for p in patterns] or (
            [prefix] if prefix else ["."]
        )
    except ValueError:
        # `target` is not under `root` at all -- nothing to anchor to;
        # preserve the pre-extraction `-C target` behavior, which is
        # correct there since no ambient GIT_DIR of THIS repo applies to an
        # unrelated tree.
        anchor = target
        pathspecs = list(patterns) or ["."]
    try:
        out = subprocess.run(
            ["git", "-C", str(anchor), "ls-files", "--", *pathspecs],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        tracked = [anchor / line for line in out.splitlines() if line]
        if tracked:
            return [p for p in tracked if p.is_file()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    if patterns:
        results: list[Path] = []
        for pattern in patterns:
            results.extend(target.rglob(pattern))
        return sorted(results)
    return sorted(target.rglob("*"))
