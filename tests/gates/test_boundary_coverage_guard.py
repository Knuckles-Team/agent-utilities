"""Check that the package inventory and actual tree stay in agreement."""

from pathlib import Path

import pytest

from scripts.check_boundary_coverage import check


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    (tmp_path / "agent_utilities/harness").mkdir(parents=True)
    (tmp_path / "agent_utilities/skills/development").mkdir(parents=True)
    coverage = tmp_path / "specs/au-boundary-deconstruction/coverage.md"
    coverage.parent.mkdir(parents=True)
    coverage.write_text(
        "| `agent_utilities/harness` | keep | meristem |\n"
        "| `agent_utilities/skills` | keep | meristem |\n"
        "| `agent_utilities/skills/development` | keep | meristem |\n"
    )
    return tmp_path


@pytest.mark.spec("AU-BOUNDARY-R042")
def test_inventory_rejects_unlisted_and_stale_directories(checkout: Path) -> None:
    assert check(checkout) == []
    (checkout / "agent_utilities/unlisted").mkdir()
    (checkout / "agent_utilities/.hidden").mkdir()
    (checkout / "agent_utilities/harness").rmdir()
    assert check(checkout) == [
        "unlisted directory: agent_utilities/.hidden",
        "unlisted directory: agent_utilities/unlisted",
        "stale directory: agent_utilities/harness",
    ]


@pytest.mark.spec("AU-BOUNDARY-R042")
@pytest.mark.parametrize(
    "row",
    [
        "| `agent_utilities/harness` | undecided | meristem |\n",
        "| `agent_utilities/harness` | relocate | unknown |\n",
        "| `agent_utilities/harness` | keep | meristem |\n"
        "| `agent_utilities/harness` | keep | meristem |\n",
    ],
)
def test_inventory_rejects_undecided_invalid_or_duplicate_rows(
    checkout: Path, row: str
) -> None:
    (checkout / "specs/au-boundary-deconstruction/coverage.md").write_text(row)
    with pytest.raises(ValueError):
        check(checkout)
