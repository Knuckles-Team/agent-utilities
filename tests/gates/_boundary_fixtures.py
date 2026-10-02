"""Shared fixture builders for the AU boundary gate tests.

Not itself a test module (leading underscore keeps it out of pytest
collection): both ``test_boundary_coverage_gate.py`` and
``test_boundary_owners_gate.py`` build the same small coverage.md fixture
shape, so the row/table builders live here once.
"""

from __future__ import annotations

from pathlib import Path

from scripts.boundary_inventory import COVERAGE_PATH

HEADER = (
    "| Directory | Approx. lines | Disposition | Covering requirement ID(s) | Notes |\n"
    "|---|---:|---|---|---|\n"
)


def row(directory: str, disposition: str, requirement: str = "AU-FIXTURE-R7") -> str:
    """One well-formed fixture row for the inventory table."""
    return f"| `{directory}/` | 10 | {disposition} | {requirement} | fixture row |\n"


def write_coverage(root: Path, rows: str) -> None:
    """Write a minimal, strictly-parseable coverage.md fixture under ``root``."""
    path = root / COVERAGE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Fixture inventory\n\n## Deletion and relocation inventory\n\n"
        + HEADER
        + rows,
        encoding="utf-8",
    )
