"""Tests for the AU boundary tree-versus-inventory coverage gate."""

from __future__ import annotations

from pathlib import Path

from scripts.boundary_inventory import COVERAGE_PATH
from scripts.check_boundary_coverage import check
from tests.gates._boundary_fixtures import HEADER, row, write_coverage

# The 8 directories the real repository currently leaves undecided (see
# coverage.md's own "Directories with no covering requirement" section). The
# owning boundary requirement forbids inventing a disposition for any of
# them, so this gate stays red against the real tree until an owner resolves
# each one; this test is evidence the gate reads today's real table
# correctly, not a claim the gate is clean. The nine graph-os-owned skill
# directories (agent-utilities-source-integration, autonomous-contribution,
# and the seven graph-* skills) were deleted from AU -- moved to graph-os's
# own `graph_os/skills/<name>/` (taken over in graph-os commit 704ce45) --
# and their coverage.md rows dropped with them, so they no longer appear here.
_KNOWN_UNDECIDED = (
    "agent_utilities/data",
    "agent_utilities/images",
    "agent_utilities/protocols/voice_supply_chain",
    "agent_utilities/skills/agent-utilities-deployment",
    "agent_utilities/skills/agent-utilities-development",
    "agent_utilities/skills/agent-utilities-evolution",
    "agent_utilities/skills/agent-utilities-self-evolution",
    "agent_utilities/skills/workflows",
)


def test_real_tree_reports_exactly_the_known_undecided_rows() -> None:
    """Document today's evidence: 8 undecided rows, nothing else wrong.

    The owning boundary requirement cannot be delivered while these rows stay
    undecided, and this gate may not invent a disposition for them
    (requirement text and the lane brief both forbid it). If this test ever
    needs updating, it is because an owner resolved a row or the tree
    changed -- never because the gate was loosened.
    """
    root = Path(__file__).resolve().parents[2]
    findings = check(root)
    undecided = sorted(f for f in findings if f.startswith("undecided row: "))
    other = [f for f in findings if not f.startswith("undecided row: ")]
    assert other == []
    reported_directories = {f.split("'")[1] for f in undecided}
    assert reported_directories == set(_KNOWN_UNDECIDED)
    assert len(undecided) == 8


def _make_package(root: Path, directories: tuple[str, ...]) -> None:
    for directory in directories:
        package_dir = root / directory
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")


def test_passing_case_clean_fixture_tree(tmp_path: Path) -> None:
    _make_package(tmp_path, ("agent_utilities", "agent_utilities/widgets"))
    rows = row("agent_utilities", "keep in AU") + row(
        "agent_utilities/widgets", "keep in AU"
    )
    write_coverage(tmp_path, rows)
    assert check(tmp_path) == []


def test_unlisted_directory_is_a_finding(tmp_path: Path) -> None:
    _make_package(tmp_path, ("agent_utilities", "agent_utilities/widgets"))
    write_coverage(tmp_path, row("agent_utilities", "keep in AU"))
    findings = check(tmp_path)
    assert any("unlisted directory: 'agent_utilities/widgets'" in f for f in findings)


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


def test_undecided_row_is_a_finding(tmp_path: Path) -> None:
    _make_package(tmp_path, ("agent_utilities", "agent_utilities/widgets"))
    rows = row("agent_utilities", "keep in AU") + row(
        "agent_utilities/widgets", "undecided"
    )
    write_coverage(tmp_path, rows)
    findings = check(tmp_path)
    assert any("undecided row: 'agent_utilities/widgets'" in f for f in findings)


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
