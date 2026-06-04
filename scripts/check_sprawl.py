#!/usr/bin/env python3
"""Anti-sprawl + hygiene gate (Plan 10 Vectors 1 & 8).

Fails on:
  - versioned-clone filenames: *_v2.py, *_old.py, *_new.py
  - merge/conflict artifacts: *.orig, *.rej, *.bak
  - the literal botched-merge marker `# --- Merged from`
  - tracked binaries above a size threshold

Usage: python3 scripts/check_sprawl.py [ROOT]   (default: repo root)
Exit 0 = clean, 1 = violations found.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CLONE_RE = re.compile(r".*_(v\d+|old|new)\.py$")
ARTIFACT_SUFFIXES = (".orig", ".rej", ".bak")
MERGE_MARKER = "# --- Merged from"
MAX_BINARY_BYTES = 1_000_000
# Files permitted to contain MERGE_MARKER literally: this gate defines it, and
# its tests write/assert on it as a fixture. Flagging these is a false positive.
MARKER_ALLOWLIST = {
    "scripts/check_sprawl.py",
    "tests/gates/test_gates_meta.py",
    "tests/unit/graph/test_learned_strategy.py",
}
SKIP_DIRS = {
    ".git", ".venv", "node_modules", "target", "__pycache__",
    ".hypothesis", ".ruff_cache", ".mypy_cache", ".pytest_cache", "dist", "build",
}
TEXT_SUFFIXES = {
    ".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".md", ".txt", ".toml",
    ".yaml", ".yml", ".json", ".ttl", ".cfg", ".ini", ".sh", ".html", ".css",
}


def _candidate_files(root: Path):
    """Prefer git-tracked files (ignores build/data junk & .gitignored files);
    fall back to a filtered filesystem walk outside a git repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            capture_output=True, text=True, check=True,
        ).stdout
        tracked = [root / line for line in out.splitlines() if line]
        if tracked:
            return tracked
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return [p for p in root.rglob("*") if not any(d in p.parts for d in SKIP_DIRS)]


def scan(root: Path) -> list[str]:
    violations: list[str] = []
    for path in _candidate_files(root):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        name = path.name
        rel = path.relative_to(root)

        if CLONE_RE.match(name):
            violations.append(f"versioned-clone file: {rel}")
        if name.endswith(ARTIFACT_SUFFIXES):
            violations.append(f"merge/conflict artifact: {rel}")

        if path.suffix in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if MERGE_MARKER in text and rel.as_posix() not in MARKER_ALLOWLIST:
                violations.append(f"botched-merge marker in: {rel}")
        else:
            try:
                if path.stat().st_size > MAX_BINARY_BYTES:
                    violations.append(
                        f"tracked binary > {MAX_BINARY_BYTES} bytes: {rel} "
                        f"({path.stat().st_size} bytes)"
                    )
            except OSError:
                continue
    return violations


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    violations = scan(root)
    if violations:
        print("Anti-sprawl gate FAILED:", file=sys.stderr)
        for v in sorted(violations):
            print(f"  - {v}", file=sys.stderr)
        return 1
    print("OK: no sprawl/hygiene violations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
