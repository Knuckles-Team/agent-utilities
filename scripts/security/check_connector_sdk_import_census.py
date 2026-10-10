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
  python3 scripts/security/check_connector_sdk_import_census.py
      # no --range: census BOTH the F-J (AU-BOUNDARY-R006.1) and T-Z
      # (AU-BOUNDARY-R009.1) ranges, so the fast-tier contract-checks
      # forwarder (scripts/security/run_contract_checks.py, which invokes
      # every scripts/security/check_*.py with no arguments at all) gets
      # full coverage of both ranges instead of an argparse usage error.
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


LETTER_RANGES = {"fj": ("f", "j"), "tz": ("t", "z")}


def census_selected(fleet_root: Path, selected_range: str | None) -> dict:
    """Census one named range, or BOTH ranges when ``selected_range`` is
    ``None`` -- the no-arg invocation used by
    scripts/security/run_contract_checks.py (the fast-tier contract-checks
    forwarder, which calls every scripts/security/check_*.py with no
    arguments). Omitting --range is strictly more coverage, not a weaker
    default.
    """
    ranges = LETTER_RANGES if selected_range is None else {selected_range: LETTER_RANGES[selected_range]}
    merged: dict = {}
    for letters in ranges.values():
        merged.update(census(fleet_root, *letters))
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--range",
        choices=["fj", "tz"],
        default=None,
        help=(
            "Restrict the census to one letter range. Omit to census BOTH "
            "ranges (stricter, not a weaker default) -- required so the "
            "no-arg invocation used by scripts/security/run_contract_checks.py "
            "(the fast-tier contract-checks forwarder) still runs a real check "
            "instead of failing argparse's required-argument validation."
        ),
    )
    parser.add_argument(
        "--fleet-root",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "agents",
    )
    args = parser.parse_args()
    result = census_selected(args.fleet_root, args.range)
    for name, hits in result.items():
        print(f"{name}: {', '.join(hits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
