#!/usr/bin/env python3
"""AU-BOUNDARY-R042: every ``agent_utilities/`` package directory has exactly
one recorded disposition in the deletion-and-relocation inventory.

The inventory lives in ``specs/au-boundary-deconstruction/coverage.md`` (see
``scripts/boundary_inventory.py`` for the shared table parser). This gate
compares that table against the real tree and fails on:

* a directory that exists on disk with no row ("unlisted"),
* a row naming a directory that no longer exists ("stale"),
* a directory named by more than one row ("duplicate"),
* a row whose disposition is exactly "undecided".

Granularity follows the table's own choices: a directory only needs a row of
its own if its parent directory already has one (the root ``agent_utilities``
row always does). A directory whose disposition names it "wholesale" is not
required to additionally list each of its children; one that the table
already chose to split further must list every child it has.

This gate currently fails against the real tree -- 17 directories (about
8,500 lines) have no decision in either boundary spec and none may be
invented here (AU-BOUNDARY-R042's own text forbids guessing a disposition).
It therefore runs report-only in CI (see ``.github/workflows/advisory.yml``)
until an owner resolves those rows; see this script's module-level tests for
the exact list the real tree reports today.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.boundary_inventory import (  # noqa: E402
    COVERAGE_PATH,
    CoverageParseError,
    CoverageRow,
    directory_rows_by_path,
    load_coverage_rows,
)

_IGNORED_DIR_NAMES = frozenset({"__pycache__"})
ROOT_DIRECTORY = "agent_utilities"


def check(root: Path) -> list[str]:
    """Return every tree-versus-inventory finding for ``root``."""
    coverage_file = root / COVERAGE_PATH
    if not coverage_file.is_file():
        return [f"inventory file not found: {COVERAGE_PATH.as_posix()}"]
    try:
        rows = load_coverage_rows(coverage_file)
    except CoverageParseError as exc:
        return [f"inventory table is unparseable: {exc}"]
    by_path = directory_rows_by_path(rows)
    findings = _duplicate_findings(by_path)
    findings += _stale_findings(root, by_path)
    findings += _undecided_findings(by_path)
    findings += _unlisted_findings(root, by_path)
    return sorted(findings)


def _duplicate_findings(by_path: dict[str, list[CoverageRow]]) -> list[str]:
    return [
        f"duplicate row: '{directory}' has {len(entries)} rows "
        f"(lines {sorted(entry.line_no for entry in entries)})"
        for directory, entries in by_path.items()
        if len(entries) > 1
    ]


def _stale_findings(root: Path, by_path: dict[str, list[CoverageRow]]) -> list[str]:
    return [
        f"stale row: '{directory}' (line {entries[0].line_no}) no longer exists on disk"
        for directory, entries in by_path.items()
        if not (root / directory).is_dir()
    ]


def _undecided_findings(by_path: dict[str, list[CoverageRow]]) -> list[str]:
    return [
        f"undecided row: '{directory}' (line {entries[0].line_no}, "
        f"requirement {entries[0].requirement_cell})"
        for directory, entries in by_path.items()
        if len(entries) == 1 and entries[0].is_undecided
    ]


def _unlisted_findings(root: Path, by_path: dict[str, list[CoverageRow]]) -> list[str]:
    """Walk only where the table itself already chose to split a directory.

    A directory the table names "wholesale" (none of its direct children has
    its own row) is not required to additionally enumerate every child --
    that is the table's own granularity, not a gap. A directory where the
    table lists at least one direct child must list every child that exists
    on disk; that partial split is exactly the shape a missing row hides in.
    """
    findings: list[str] = []
    _walk_listed_children(ROOT_DIRECTORY, root, by_path, findings, require=True)
    return findings


def _walk_listed_children(
    directory: str,
    root: Path,
    by_path: dict[str, list[CoverageRow]],
    findings: list[str],
    *,
    require: bool = False,
) -> None:
    # Level two (the package root's own direct children) is unconditionally
    # required by AU-BOUNDARY-R042's own text ("the top two levels"). Below
    # that, a directory is only walked further if the table itself already
    # chose to split it -- see this function's caller and
    # ``_has_listed_children``.
    disk_path = root / directory
    if not disk_path.is_dir() or not (
        require or _has_listed_children(directory, by_path)
    ):
        return
    for child in sorted(
        (
            p
            for p in disk_path.iterdir()
            if p.is_dir() and p.name not in _IGNORED_DIR_NAMES
        ),
        key=lambda p: p.name,
    ):
        child_directory = f"{directory}/{child.name}"
        if child_directory not in by_path:
            findings.append(
                f"unlisted directory: '{child_directory}' has no row in "
                f"{COVERAGE_PATH.as_posix()}"
            )
            continue
        _walk_listed_children(child_directory, root, by_path, findings)


def _has_listed_children(directory: str, by_path: dict[str, list[CoverageRow]]) -> bool:
    prefix = f"{directory}/"
    return any(
        path.startswith(prefix) and "/" not in path[len(prefix) :] for path in by_path
    )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]).resolve() if args else ROOT
    findings = check(root)
    if findings:
        print(
            "Boundary coverage inventory gate failed (AU-BOUNDARY-R042):",
            file=sys.stderr,
        )
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 1
    print("Boundary coverage inventory gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
