"""Qualify ownership of the development skill retained in Meristem."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.spec("AU-BOUNDARY-R048.11")
def test_every_current_skill_is_recorded_and_has_a_skill_definition() -> None:
    skill_root = ROOT / "agent_utilities/skills"
    coverage = (ROOT / "specs/au-boundary-deconstruction/coverage.md").read_text()
    directories = sorted(
        path
        for path in skill_root.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    )
    assert directories
    for directory in directories:
        relative = directory.relative_to(ROOT).as_posix()
        assert f"| `{relative}` | keep | meristem |" in coverage
        skill = (directory / "SKILL.md").read_text()
        assert skill.startswith("---\n")
        frontmatter = skill.split("---", 2)
        assert len(frontmatter) == 3
        fields = dict(
            line.split(":", 1) for line in frontmatter[1].splitlines() if ":" in line
        )
        assert fields.get("name", "").strip() == directory.name
        assert fields.get("description", "").strip()
