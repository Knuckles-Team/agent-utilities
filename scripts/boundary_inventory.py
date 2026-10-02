"""Shared parsing for the AU deletion-and-relocation inventory table.

``check_boundary_coverage.py`` (AU-BOUNDARY-R042) and ``check_boundary_owners.py``
(AU-BOUNDARY-R038) both read the one disposition table in
``specs/au-boundary-deconstruction/coverage.md`` -- the "Deletion and relocation
inventory" under that heading -- and must agree on exactly what each row says.
This module is the single place that table is parsed, so the two gates can
never silently drift against each other.

Parsing is strict on purpose: a row that does not match the expected five-cell
shape is a hard failure (:class:`CoverageParseError`), never a row silently
skipped. An inventory a gate cannot fully read is not evidence of anything.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

COVERAGE_PATH = Path("specs/au-boundary-deconstruction/coverage.md")

_TABLE_HEADER = (
    "| Directory | Approx. lines | Disposition | Covering requirement ID(s) | Notes |"
)
_SEPARATOR = re.compile(r"^\|(?:\s*:?-+:?\s*\|)+$")
_DIRECTORY_CELL = re.compile(
    r"^`(agent_utilities(?:/[A-Za-z0-9_][A-Za-z0-9_.\-]*)*)/?`$"
)
_LINES_CELL = re.compile(r"^[0-9][0-9,]*$")
_REQUIREMENT_TOKEN = re.compile(r"AU-[A-Z]+-R[0-9]+")
_UNDECIDED = "undecided"
_EXPECTED_CELL_COUNT = 5


class CoverageParseError(ValueError):
    """The AU boundary disposition table could not be parsed strictly."""


@dataclass(frozen=True, slots=True)
class CoverageRow:
    """One row of the deletion-and-relocation inventory table."""

    directory: str
    disposition: str
    requirement_cell: str
    notes: str
    line_no: int

    @property
    def is_undecided(self) -> bool:
        return self.disposition.strip().lower() == _UNDECIDED


def parse_coverage_rows(text: str) -> list[CoverageRow]:
    """Strictly parse every data row of the inventory table in ``text``."""
    lines = text.splitlines()
    header_index = _find_table_header(lines)
    rows: list[CoverageRow] = []
    for offset, line in enumerate(lines[header_index + 2 :]):
        if not line.startswith("|"):
            break
        rows.append(_parse_row(line, header_index + 3 + offset))
    if not rows:
        raise CoverageParseError("no data rows found under the inventory table header")
    return rows


def load_coverage_rows(coverage_file: Path) -> list[CoverageRow]:
    """Read and strictly parse the inventory table from ``coverage_file``."""
    try:
        text = coverage_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise CoverageParseError(f"cannot read {coverage_file}: {exc}") from exc
    return parse_coverage_rows(text)


def directory_rows_by_path(rows: list[CoverageRow]) -> dict[str, list[CoverageRow]]:
    """Group rows by their (not yet deduplicated) directory path."""
    by_path: dict[str, list[CoverageRow]] = {}
    for row in rows:
        by_path.setdefault(row.directory, []).append(row)
    return by_path


def owner_recorded_present(disposition: str, owner_repository: str) -> bool:
    """Whether ``disposition`` records this path as relocating to
    ``owner_repository`` while still physically present in AU -- coverage.md's
    own "relocate to <repository>" phrasing, which is how it marks a module
    "moving, still present" rather than already cut over.
    """
    return f"relocate to {owner_repository}".lower() in disposition.lower()


def best_matching_row(
    module_path: str, by_path: dict[str, list[CoverageRow]]
) -> CoverageRow | None:
    """Return the single-row directory that is the longest matching prefix of
    ``module_path``, or ``None`` if no unambiguous row covers it.

    A directory with more than one row is a separate ``AU-BOUNDARY-R042``
    finding (duplicate row); it never silently wins a prefix match here.
    """
    best: CoverageRow | None = None
    best_len = -1
    for directory, entries in by_path.items():
        if len(entries) != 1 or not _is_prefix(directory, module_path):
            continue
        if len(directory) > best_len:
            best, best_len = entries[0], len(directory)
    return best


def _is_prefix(directory: str, module_path: str) -> bool:
    return module_path == directory or module_path.startswith(directory + "/")


def _find_table_header(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if line.strip() != _TABLE_HEADER:
            continue
        separator = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if not _SEPARATOR.match(separator):
            raise CoverageParseError(
                f"line {index + 2}: expected a table separator after the header"
            )
        return index
    raise CoverageParseError("inventory table header not found")


def _split_cells(line: str, line_no: int) -> list[str]:
    if not line.endswith("|"):
        raise CoverageParseError(f"line {line_no}: row does not end with '|'")
    parts = line.split("|")
    if len(parts) != _EXPECTED_CELL_COUNT + 2 or parts[0].strip() or parts[-1].strip():
        raise CoverageParseError(
            f"line {line_no}: expected exactly {_EXPECTED_CELL_COUNT} table cells"
        )
    return [cell.strip() for cell in parts[1:-1]]


def _parse_row(line: str, line_no: int) -> CoverageRow:
    directory_cell, lines_cell, disposition, requirement_cell, notes = _split_cells(
        line, line_no
    )
    directory = _parse_directory(directory_cell, line_no)
    if not _LINES_CELL.match(lines_cell.replace(" ", "")):
        raise CoverageParseError(
            f"line {line_no}: unparseable line-count cell {lines_cell!r}"
        )
    if not disposition:
        raise CoverageParseError(f"line {line_no}: empty disposition cell")
    if not _REQUIREMENT_TOKEN.search(requirement_cell):
        raise CoverageParseError(
            f"line {line_no}: requirement cell has no requirement ID {requirement_cell!r}"
        )
    return CoverageRow(
        directory=directory,
        disposition=disposition,
        requirement_cell=requirement_cell,
        notes=notes,
        line_no=line_no,
    )


def _parse_directory(directory_cell: str, line_no: int) -> str:
    match = _DIRECTORY_CELL.match(directory_cell)
    if not match:
        raise CoverageParseError(
            f"line {line_no}: unparseable directory cell {directory_cell!r}"
        )
    return match.group(1)


def resolve_cli_root(argv: list[str] | None, default_root: Path) -> Path:
    """Return the root a gate CLI should scan: an explicit argv[0], else the default."""
    args = sys.argv[1:] if argv is None else argv
    return Path(args[0]).resolve() if args else default_root


def report_findings(findings: list[str], *, gate_label: str) -> int:
    """Print ``findings`` in the one shared gate CLI shape and return its exit code."""
    if findings:
        print(f"{gate_label} gate failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 1
    print(f"{gate_label} gate passed")
    return 0
