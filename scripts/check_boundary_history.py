#!/usr/bin/env python3
"""Check time-scoped evidence for two historical boundary decisions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DEPLOYMENT = "fca0c5ee42b36107f810d928c080f76f56acd96c"
TOOLKIT_GATE = "2456745c1e4d3c90d77f5d3e3c3eff8cee4cc98b"
BACKENDS_DELETION = "db29b2da626e4d600ea64c037066c902ba1c559f"
HARNESS_CUT = "55e4f61a681f798680f1311ee2add721492b4a95"
BACKENDS = "agent_utilities/deployment/backends.py"
OIDC = "agent_utilities/deployment/certification_oidc.py"
BASE = "agent_utilities/base_utilities.py"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _exists(root: Path, revision: str, path: str) -> bool:
    return _git(root, "ls-tree", "-r", "--name-only", revision, "--", path) == path


def _deleted(root: Path, revision: str, path: str) -> bool:
    return (
        _git(root, "diff", "--name-status", f"{revision}^", revision, "--", path)
        == f"D\t{path}"
    )


def _ancestor(root: Path, revision: str, head: str) -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, head],
        cwd=root,
        check=True,
        capture_output=True,
    )


def check_deployment(
    root: Path, original: str, backends_cut: str, oidc_cut: str, head: str
) -> None:
    """Confirm the original deployment state and both later deletions."""
    for revision in (original, backends_cut, oidc_cut):
        _ancestor(root, revision, head)
    _ancestor(root, original, backends_cut)
    _ancestor(root, original, oidc_cut)
    if not all(_exists(root, original, path) for path in (BACKENDS, OIDC)):
        raise ValueError("original deployment evidence does not contain both files")
    if not _deleted(root, backends_cut, BACKENDS) or not _deleted(root, oidc_cut, OIDC):
        raise ValueError("deployment deletion evidence is missing")
    if any(_exists(root, head, path) for path in (BACKENDS, OIDC)):
        raise ValueError("retired deployment file exists at head")


def check_toolkit(root: Path, original: str, cut: str, head: str) -> None:
    """Confirm the gate-only change, former toolkit file and later deletion."""
    for revision in (original, cut):
        _ancestor(root, revision, head)
    _ancestor(root, original, cut)
    changed = _git(
        root, "diff-tree", "--no-commit-id", "--name-only", "-r", original
    ).splitlines()
    allowed = (".config/", ".github/", "architecture/", "scripts/", "tests/gates/")
    if not changed or any(not path.startswith(allowed) for path in changed):
        raise ValueError("original toolkit evidence is not gate-only")
    if not _exists(root, original, BASE) or not _deleted(root, cut, BASE):
        raise ValueError("toolkit history evidence is missing")
    if _exists(root, head, BASE):
        raise ValueError("retired toolkit file exists at head")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        check_deployment(root, DEPLOYMENT, BACKENDS_DELETION, HARNESS_CUT, "HEAD")
        check_toolkit(root, TOOLKIT_GATE, HARNESS_CUT, "HEAD")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"boundary history evidence invalid: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
