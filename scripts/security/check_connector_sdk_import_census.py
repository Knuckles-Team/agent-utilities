#!/usr/bin/env python3
"""Per-package import census for the connector-SDK migration (AU-BOUNDARY-R006/R009).

Scans sibling connector packages for the banned Agent Utilities import
surface (``mcp``, ``core.config``, ``core.exceptions``,
``core.transport_security``, ``base_utilities``) so that each connector
repo's own migration PR can be checked against one shared instrument
instead of a bespoke, repo-local grep.

This is the AU-side census instrument named by AU-BOUNDARY-R006/R009's
verification text. It does not itself migrate any connector package; the
per-package migrations are tracked as further ``AU-BOUNDARY-R006.n`` /
``AU-BOUNDARY-R009.n`` rows against those repos.

Usage:
  python3 scripts/security/check_connector_sdk_import_census.py --range fj
  python3 scripts/security/check_connector_sdk_import_census.py --range tz --fleet-root DIR
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

BANNED_MODULES = (
    "agent_utilities.mcp",
    "agent_utilities.core.config",
    "agent_utilities.core.exceptions",
    "agent_utilities.core.transport_security",
    "agent_utilities.base_utilities",
)

# Bare "agent_utilities" (e.g. "import agent_utilities") is not itself
# banned -- the public API (agent_utilities.api) is an allowed surface --
# so only dotted submodule imports that resolve under one of the banned
# prefixes above are flagged.

_SKIP_DIR_NAMES = {
    "tests",
    "test",
    ".git",
    "build",
    "dist",
    "scripts",
    ".uv-workspace-siblings",
    ".venv",
    "venv",
    "node_modules",
    "site-packages",
}


def _has_skipped_part(relative_parts: tuple) -> bool:
    for part in relative_parts:
        if part in _SKIP_DIR_NAMES or part.endswith(".egg-info"):
            return True
    return False


def _qualified_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    if isinstance(node, ast.Import):
        names.extend(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
        for alias in node.names:
            names.append(f"{node.module}.{alias.name}")
            names.append(node.module)
    return names


def banned_imports_in_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return []
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for qualified in _qualified_names(node):
            for banned in BANNED_MODULES:
                if qualified == banned or qualified.startswith(banned + "."):
                    hits.append(banned)
    return hits


def _package_dirs(fleet_root: Path, letter_start: str, letter_end: str) -> list[Path]:
    lo, hi = letter_start.lower(), letter_end.lower()
    out = []
    for child in sorted(fleet_root.iterdir()):
        if not child.is_dir():
            continue
        first = child.name[:1].lower()
        if lo <= first <= hi:
            out.append(child)
    return out


def census(fleet_root: Path, letter_start: str, letter_end: str) -> dict:
    """Return {package_name: [banned_import, ...]} for non-compliant packages."""

    violations: dict = {}
    for package_dir in _package_dirs(fleet_root, letter_start, letter_end):
        found: list[str] = []
        for py_file in package_dir.rglob("*.py"):
            if _has_skipped_part(py_file.relative_to(package_dir).parts[:-1]):
                continue
            found.extend(banned_imports_in_file(py_file))
        if found:
            violations[package_dir.name] = sorted(set(found))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range", choices=["fj", "tz"], required=True)
    parser.add_argument(
        "--fleet-root",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "agents",
    )
    args = parser.parse_args()
    letters = {"fj": ("f", "j"), "tz": ("t", "z")}[args.range]
    result = census(args.fleet_root, *letters)
    for name, hits in result.items():
        print(f"{name}: {', '.join(hits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
