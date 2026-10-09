"""Tests for the AU boundary tree-versus-inventory coverage gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.boundary_inventory import COVERAGE_PATH
from scripts.check_boundary_coverage import check
from tests.gates._boundary_fixtures import HEADER, row, write_coverage


@pytest.mark.spec(
    "AU-BOUNDARY-R003",
    "AU-BOUNDARY-R004",
    "AU-BOUNDARY-R010",
    "AU-BOUNDARY-R011",
    "AU-BOUNDARY-R015",
    "AU-BOUNDARY-R042",
    "AU-BOUNDARY-R048",
    "AU-DEV-R002",
)
def test_real_tree_reports_zero_undecided_rows() -> None:
    """The 17 directories coverage.md once left undecided (see its own
    "Directories with no covering requirement" section) were resolved by an
    operator ruling on 2026-10-03: moved, merged, deleted, or marked keep/
    relocate. The owning boundary requirement forbids inventing a
    disposition, so this gate was red against the real tree until that
    ruling landed; it must report nothing wrong now.
    """
    root = Path(__file__).resolve().parents[2]
    assert check(root) == []


def _make_package(root: Path, directories: tuple[str, ...]) -> None:
    for directory in directories:
        package_dir = root / directory
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")


@pytest.mark.spec(
    "AU-BOUNDARY-R003",
    "AU-BOUNDARY-R004",
    "AU-BOUNDARY-R010",
    "AU-BOUNDARY-R011",
    "AU-BOUNDARY-R015",
    "AU-BOUNDARY-R042",
    "AU-DEV-R002",
)
def test_passing_case_clean_fixture_tree(tmp_path: Path) -> None:
    _make_package(tmp_path, ("agent_utilities", "agent_utilities/widgets"))
    rows = row("agent_utilities", "keep in AU") + row(
        "agent_utilities/widgets", "keep in AU"
    )
    write_coverage(tmp_path, rows)
    assert check(tmp_path) == []


@pytest.mark.spec(
    "AU-BOUNDARY-R003",
    "AU-BOUNDARY-R004",
    "AU-BOUNDARY-R010",
    "AU-BOUNDARY-R011",
    "AU-BOUNDARY-R015",
    "AU-BOUNDARY-R042",
)
def test_unlisted_directory_is_a_finding(tmp_path: Path) -> None:
    _make_package(tmp_path, ("agent_utilities", "agent_utilities/widgets"))
    write_coverage(tmp_path, row("agent_utilities", "keep in AU"))
    findings = check(tmp_path)
    assert any("unlisted directory: 'agent_utilities/widgets'" in f for f in findings)


@pytest.mark.spec("AU-BOUNDARY-R038", "AU-DEV-R002")
def test_stale_row_is_a_finding(tmp_path: Path) -> None:
    _make_package(tmp_path, ("agent_utilities",))
    rows = row("agent_utilities", "keep in AU") + row(
        "agent_utilities/ghost", "keep in AU"
    )
    write_coverage(tmp_path, rows)
    findings = check(tmp_path)
    assert any("stale row: 'agent_utilities/ghost'" in f for f in findings)


def test_duplicate_row_is_a_finding(tmp_path: Path) -> None:
    _make_package(tmp_path, ("agent_utilities",))
    rows = row("agent_utilities", "keep in AU") + row("agent_utilities", "delete")
    write_coverage(tmp_path, rows)
    findings = check(tmp_path)
    assert any("duplicate row: 'agent_utilities'" in f for f in findings)


@pytest.mark.spec("AU-BOUNDARY-R038")
def test_undecided_row_is_a_finding(tmp_path: Path) -> None:
    _make_package(tmp_path, ("agent_utilities", "agent_utilities/widgets"))
    rows = row("agent_utilities", "keep in AU") + row(
        "agent_utilities/widgets", "undecided"
    )
    write_coverage(tmp_path, rows)
    findings = check(tmp_path)
    assert any("undecided row: 'agent_utilities/widgets'" in f for f in findings)


@pytest.mark.spec("AU-BOUNDARY-R038")
def test_unparseable_row_is_a_failure_not_a_skip(tmp_path: Path) -> None:
    _make_package(tmp_path, ("agent_utilities",))
    path = tmp_path / COVERAGE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # Four cells instead of five: a short row is unparseable, not skippable.
    path.write_text(
        "# Fixture inventory\n\n## Deletion and relocation inventory\n\n"
        + HEADER
        + "| `agent_utilities/` | 10 | keep in AU | AU-FIXTURE-R7 |\n",
        encoding="utf-8",
    )
    findings = check(tmp_path)
    assert len(findings) == 1
    assert "unparseable" in findings[0]


def test_missing_inventory_file_is_a_failure(tmp_path: Path) -> None:
    _make_package(tmp_path, ("agent_utilities",))
    findings = check(tmp_path)
    assert len(findings) == 1
    assert "not found" in findings[0]
