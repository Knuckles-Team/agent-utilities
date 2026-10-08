"""Shared synthetic Git repositories for merge-queue integration tests.

The small checker is a fixture executable; production source-comparison behavior
is verified separately by the liveness and queue-checker regression tests.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _run(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        args, cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def _write(root: Path, rel: str, body: str) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def _commit(root: Path, message: str) -> str:
    _run(["git", "add", "-A"], root)
    _run(["git", "commit", "-qm", message], root)
    return _run(["git", "rev-parse", "HEAD"], root)


def _lane(canonical: Path, name: str) -> Path:
    path = canonical.parent / name
    _run(["git", "worktree", "add", "-q", str(path), "-b", name], canonical)
    return path


def _branch(canonical: Path, name: str, files: dict[str, str]) -> Path:
    """A lane worktree that has already committed *files* on its own branch."""
    tree = _lane(canonical, name)
    for rel, body in files.items():
        _write(tree, rel, body)
    _commit(tree, f"{name}: work")
    return tree


def seed_repository(root: Path) -> Path:
    """Create the importable synthetic repo required by real queue integration tests."""
    root.mkdir(parents=True)
    _run(["git", "init", "-b", "main"], root)
    _run(["git", "config", "user.email", "queue@test"], root)
    _run(["git", "config", "user.name", "Queue Test"], root)
    for name, content in {
        "pkg/__init__.py": "",
        "pkg/core.py": "VALUE = 1\n",
        "scripts/check_liveness_source.py": "print('fixture liveness gate')\n",
    }.items():
        _write(root, name, content)
    _commit(root, "base")
    return root
