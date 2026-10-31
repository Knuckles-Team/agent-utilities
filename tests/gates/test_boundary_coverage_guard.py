"""Check that the package inventory and actual tree stay in agreement."""

from pathlib import Path

import pytest

from scripts.check_boundary_coverage import check


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    (tmp_path / "meristem/harness").mkdir(parents=True)
    (tmp_path / "meristem/skills/development").mkdir(parents=True)
    coverage = tmp_path / "specs/au-boundary-deconstruction/plan.md"
    coverage.parent.mkdir(parents=True)
    coverage.write_text(
        "| `meristem/harness` | keep | meristem |\n"
        "| `meristem/skills` | keep | meristem |\n"
        "| `meristem/skills/development` | keep | meristem |\n"
    )
    return tmp_path


@pytest.mark.spec("MERISTEM-BOUNDARY-R042")
def test_inventory_rejects_unlisted_and_stale_directories(checkout: Path) -> None:
    assert check(checkout) == []
    (checkout / "meristem/unlisted").mkdir()
    (checkout / "meristem/.hidden").mkdir()
    (checkout / "meristem/harness").rmdir()
    assert check(checkout) == [
        "unlisted directory: meristem/.hidden",
        "unlisted directory: meristem/unlisted",
        "stale directory: meristem/harness",
    ]


@pytest.mark.spec("MERISTEM-BOUNDARY-R042")
@pytest.mark.parametrize(
    "row",
    [
        "| `meristem/harness` | undecided | meristem |\n",
        "| `meristem/harness` | relocate | unknown |\n",
        "| `meristem/harness` | keep | meristem |\n"
        "| `meristem/harness` | keep | meristem |\n",
    ],
)
def test_inventory_rejects_undecided_invalid_or_duplicate_rows(
    checkout: Path, row: str
) -> None:
    (checkout / "specs/au-boundary-deconstruction/plan.md").write_text(row)
    with pytest.raises(ValueError):
        check(checkout)
