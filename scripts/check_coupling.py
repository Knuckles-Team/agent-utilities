#!/usr/bin/env python3
"""Coupling gate (Plan 10 — layering / decoupling vector).

Enforces that the ``geniusbot`` desktop app depends on ``agent_utilities`` ONLY
through a thin, allowlisted adapter seam. Any *other* geniusbot module that
imports ``agent_utilities`` (directly or via ``from agent_utilities ...``) is a
layering violation: it couples the UI to the platform internals and defeats the
decoupling effort.

Allowlisted adapter paths (relative to the geniusbot package root
``geniusbot/geniusbot/``):

    services/backend_adapter.py
    services/gateway_client.py

Exit 0 = clean (imports only via the adapter, or no imports at all).
Exit 1 = a non-adapter geniusbot module imports agent_utilities.

Usage::

    python3 scripts/check_coupling.py [GENIUSBOT_PKG_ROOT]

``GENIUSBOT_PKG_ROOT`` defaults to ``<repo>/geniusbot/geniusbot``. The argument
exists so the meta-test can point the gate at a synthetic fixture tree.

NOTE (advisory wiring): at the time this gate was written geniusbot was NOT yet
decoupled — several modules still import agent_utilities outside the adapter, so
this gate intentionally FAILS against the live tree. It is therefore wired as an
ADVISORY (non-blocking) CI step until the concurrent decoupling work lands. Once
``backend_adapter.py`` exists and the stray imports are routed through it, flip
the CI step to blocking.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Allowlist is relative to the geniusbot *package* root (geniusbot/geniusbot/).
ALLOWLISTED_ADAPTERS = {
    "services/backend_adapter.py",
    "services/gateway_client.py",
}
TARGET_PKG = "agent_utilities"
SKIP_DIRS = {"__pycache__", ".venv", "build", "dist", "tests", "test"}


def _imports_target(source: str) -> bool:
    """True if the Python source imports ``agent_utilities`` in any form.

    Uses the AST so commented-out or string-literal mentions never trip the
    gate; falls back to a substring scan only if the file fails to parse.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return f"import {TARGET_PKG}" in source or f"from {TARGET_PKG}" in source
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == TARGET_PKG or alias.name.startswith(TARGET_PKG + "."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == TARGET_PKG or mod.startswith(TARGET_PKG + "."):
                return True
    return False


def scan(pkg_root: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(pkg_root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel = path.relative_to(pkg_root).as_posix()
        if rel in ALLOWLISTED_ADAPTERS:
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _imports_target(source):
            violations.append(rel)
    return violations


def main() -> int:
    if len(sys.argv) > 1:
        pkg_root = Path(sys.argv[1])
    else:
        repo_root = Path(__file__).resolve().parents[2]
        pkg_root = repo_root / "geniusbot" / "geniusbot"

    if not pkg_root.exists():
        print(
            f"Coupling gate SKIPPED: {pkg_root} not found (geniusbot absent).",
            file=sys.stderr,
        )
        return 0

    violations = scan(pkg_root)
    if violations:
        print("Coupling gate FAILED:", file=sys.stderr)
        print(
            f"  geniusbot must import {TARGET_PKG!r} only via the adapter seam "
            f"({', '.join(sorted(ALLOWLISTED_ADAPTERS))}).",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  - {v} imports {TARGET_PKG} outside the adapter", file=sys.stderr)
        return 1
    print(f"OK: geniusbot imports {TARGET_PKG} only via the allowlisted adapter seam.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
