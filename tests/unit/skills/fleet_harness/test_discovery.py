"""Discovery layer: finds every SKILL.md, skips build/VCS noise, de-dupes."""

from __future__ import annotations

from pathlib import Path

from agent_utilities.skills.fleet_harness.discovery import discover_skills

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_discovers_all_fixture_skills():
    records = discover_skills([_FIXTURES])
    paths = {r.relative_path for r in records}
    assert "good-skill/SKILL.md" in paths
    assert "bad_missing_frontmatter/SKILL.md" in paths
    assert "dup_skills/dup-a/SKILL.md" in paths
    assert "dup_skills/dup-b/SKILL.md" in paths
    # fake_package skills nested two levels deep are still found
    assert "fake_package/skills/good-tool-skill/SKILL.md" in paths
    assert len(records) == len(paths)  # no duplicates


def test_skips_build_and_vcs_noise(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "real-skill").mkdir(parents=True)
    (root / "real-skill" / "SKILL.md").write_text("---\nname: x\n---\nbody")
    noisy = root / ".venv" / "lib" / "vendored-skill"
    noisy.mkdir(parents=True)
    (noisy / "SKILL.md").write_text("---\nname: y\n---\nbody")

    records = discover_skills([root])
    paths = {r.relative_path for r in records}
    assert paths == {"real-skill/SKILL.md"}


def test_deduplicates_overlapping_roots(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "skill").mkdir(parents=True)
    (root / "skill" / "SKILL.md").write_text("---\nname: x\n---\nbody")

    records = discover_skills([root, root])  # same root passed twice
    assert len(records) == 1


def test_missing_root_is_skipped_not_raised(tmp_path: Path):
    missing = tmp_path / "does-not-exist"
    records = discover_skills([missing])
    assert records == []


def test_repo_name_is_root_directory_name():
    records = discover_skills([_FIXTURES])
    names = {r.repo_name for r in records}
    assert names == {"fixtures"}
