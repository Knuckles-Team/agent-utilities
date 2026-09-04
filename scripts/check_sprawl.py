#!/usr/bin/env python3
"""Anti-sprawl + hygiene gate (Plan 10 Vectors 1 & 8).

Fails on:
  - versioned-clone filenames: *_v2.py, *_old.py, *_new.py
  - merge/conflict artifacts: *.orig, *.rej, *.bak
  - the literal botched-merge marker (quoted occurrences in Markdown code
    spans and fences are not violations)
  - tracked binaries above a size threshold

Usage: python3 scripts/check_sprawl.py [ROOT]   (default: repo root)
Exit 0 = clean, 1 = violations found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._git_scan import tracked_or_walked  # noqa: E402

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

# Markdown that *documents* this gate necessarily quotes MERGE_MARKER. A real
# botched merge lands as bare line content; a quoted one sits inside backticks
# or a fence. Stripping code spans in Markdown removes that false-positive class
# without weakening detection on source files, which stay byte-exact. Growing
# MARKER_ALLOWLIST instead would require an entry per document, which is the
# sprawl this gate exists to prevent.
_MD_FENCE_RE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
_MD_INLINE_RE = re.compile(r"`[^`\n]*`")


def _strip_markdown_code(text: str) -> str:
    """Return *text* with fenced blocks and inline code spans removed."""
    return _MD_INLINE_RE.sub("", _MD_FENCE_RE.sub("", text))
SKIP_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "target",
    "__pycache__",
    ".hypothesis",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
}
TEXT_SUFFIXES = {
    ".py",
    ".rs",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ttl",
    ".cfg",
    ".ini",
    ".sh",
    ".html",
    ".css",
    ".lock",  # lockfiles (uv.lock, …) are large text files, not binary sprawl
}


def _candidate_files(root: Path):
    """Prefer git-tracked files (ignores build/data junk & .gitignored files);
    fall back to a filesystem walk outside a git repo (BUG-043) -- ``scan()``
    below re-filters ``SKIP_DIRS`` on every candidate regardless of source."""
    return tracked_or_walked(root, root=ROOT)


def _name_violations(name: str, rel: Path) -> list[str]:
    """Violations derivable from the filename alone."""
    found: list[str] = []
    if CLONE_RE.match(name):
        found.append(f"versioned-clone file: {rel}")
    if name.endswith(ARTIFACT_SUFFIXES):
        found.append(f"merge/conflict artifact: {rel}")
    return found


def _text_violations(path: Path, rel: Path) -> list[str]:
    """Violations found by reading a text file's content."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    haystack = _strip_markdown_code(text) if path.suffix == ".md" else text
    if MERGE_MARKER in haystack and rel.as_posix() not in MARKER_ALLOWLIST:
        return [f"botched-merge marker in: {rel}"]
    return []


def _binary_violations(path: Path, rel: Path) -> list[str]:
    """Violations found by sizing a non-text file."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size > MAX_BINARY_BYTES:
        return [f"tracked binary > {MAX_BINARY_BYTES} bytes: {rel} ({size} bytes)"]
    return []


def scan(root: Path) -> list[str]:
    violations: list[str] = []
    for path in _candidate_files(root):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        violations.extend(_name_violations(path.name, rel))
        if path.suffix in TEXT_SUFFIXES:
            violations.extend(_text_violations(path, rel))
        else:
            violations.extend(_binary_violations(path, rel))
    return violations


def main() -> int:
    root = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    )
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
