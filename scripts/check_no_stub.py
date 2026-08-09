#!/usr/bin/env python3
"""No-stub gate (Plan 10 Vectors 2 & 4).

Fails if production code contains stub markers:
  - "[Mock]" string returns
  - "dummy_embedding"
  - "Fallback equal-weight" / "equal weighting since"
  - `raise NotImplementedError` outside a line/file marked `# ABSTRACT-OK`

Scans `agent_utilities/` only (production code), skipping tests and the gate
scripts themselves. Exit 0 = clean, 1 = stub found.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BANNED_SUBSTRINGS = (
    '"[Mock]',
    "[Mock]'",
    "dummy_embedding",
    "Fallback equal-weight",
    "equal weighting since",
)
SKIP_DIRS = {"__pycache__", ".venv", "tests", "scripts"}


def _tracked_or_walked_py_files(pkg_root: Path) -> list[Path]:
    """``.py`` files under ``pkg_root``, preferring the git-tracked set (BUG-043).

    A raw ``rglob`` also picks up gitignored, generated build output, which
    can carry a stale copy of an already-fixed source file and reintroduce a
    cleared stub marker. Falls back to a filesystem walk only when
    ``pkg_root`` is not inside a git working tree (e.g. a test fixture).
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(pkg_root), "ls-files", "--", "*.py"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        tracked = [pkg_root / line for line in out.splitlines() if line]
        if tracked:
            return [p for p in tracked if p.is_file()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return list(pkg_root.rglob("*.py"))


def scan(pkg_root: Path) -> list[str]:
    violations: list[str] = []
    for path in _tracked_or_walked_py_files(pkg_root):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel = path.relative_to(pkg_root.parent)
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            is_comment_or_doc = stripped.startswith("#") or stripped.startswith(
                ('"', "'")
            )
            for needle in BANNED_SUBSTRINGS:
                if needle in line and not is_comment_or_doc:
                    violations.append(f"{rel}:{i}: stub marker {needle!r}")
            if "raise NotImplementedError" in line and "# ABSTRACT-OK" not in line:
                # Allow genuine abstract methods: @abstractmethod within the
                # preceding 6 lines (decorator on the enclosing def).
                preceding = "\n".join(lines[max(0, i - 7) : i])
                if "@abstractmethod" in preceding:
                    continue
                violations.append(
                    f"{rel}:{i}: raise NotImplementedError "
                    "(mark abstract with # ABSTRACT-OK)"
                )
    return violations


def main() -> int:
    if len(sys.argv) > 1:
        pkg_root = Path(sys.argv[1])
    else:
        pkg_root = Path(__file__).resolve().parents[1] / "agent_utilities"
    violations = scan(pkg_root)
    if violations:
        print("No-stub gate FAILED:", file=sys.stderr)
        for v in sorted(violations):
            print(f"  - {v}", file=sys.stderr)
        return 1
    print("OK: no stub markers in production code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
