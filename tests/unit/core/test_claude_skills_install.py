"""AU-DEV-R003.3.4/.3.5: Claude Code skills path and shared harness install."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.core import unified_install as ui
from agent_utilities.core.provider_materialization import MANAGED_PROVIDER_MARKER
from agent_utilities.core.providers import ProviderRegistration


def _registration(source: Path) -> ProviderRegistration:
    return ProviderRegistration(
        name="demo",
        group="agent_utilities.skills",
        target="demo:skills",
        owner_name="demo-dist",
        owner_version="1.0",
        digest="d" * 64,
        source_root=source,
        owned_paths=frozenset({"alpha/SKILL.md"}),
    )


@pytest.fixture
def provider_source(tmp_path: Path) -> Path:
    source = tmp_path / "src"
    (source / "alpha").mkdir(parents=True)
    (source / "alpha" / "SKILL.md").write_text("---\nname: alpha\n---\nbody\n")
    return source


@pytest.mark.spec("AU-DEV-R003.3.4")
def test_claude_skills_dir_honours_config_dir_and_home(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cc"))
    assert ui.claude_skills_dir() == tmp_path / "cc" / "skills"
    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    assert ui.claude_skills_dir() == tmp_path / "h" / ".claude" / "skills"


@pytest.mark.spec("AU-DEV-R003.3.4")
def test_claude_skills_dir_refuses_relative_symlink_and_file(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "relative/dir")
    with pytest.raises(ui.ClaudeSkillsPathError):
        ui.claude_skills_dir()
    home = tmp_path / "cc"
    home.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    (home / "skills").symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home))
    with pytest.raises(ui.ClaudeSkillsPathError):
        ui.claude_skills_dir()
    (home / "skills").unlink()
    (home / "skills").write_text("x")
    with pytest.raises(ui.ClaudeSkillsPathError):
        ui.claude_skills_dir()


@pytest.mark.spec("AU-DEV-R003.3.5")
def test_one_install_unified_feeds_both_harnesses(
    tmp_path: Path, monkeypatch, provider_source: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setattr(
        ui, "provider_registrations", lambda _g: (_registration(provider_source),)
    )
    for name in ("unified_skills_dir", "unified_prompts_dir", "unified_ontologies_dir"):
        monkeypatch.setattr(ui, name, lambda n=name: tmp_path / "xdg" / n)
    monkeypatch.setattr(
        ui, "_own_source", lambda leg: (_ for _ in ()).throw(ValueError("skip"))
    )

    result = ui.install_unified()

    for key, base in (("codex_skills", "codex"), ("claude_skills", "claude")):
        assert result[key]["providers"] == 1
        assert result[key]["failed"] == 0
        assert (tmp_path / base / "skills" / "demo" / MANAGED_PROVIDER_MARKER).is_file()


@pytest.mark.spec("AU-DEV-R003.3.5")
def test_prune_removes_only_managed_dirs(
    tmp_path: Path, monkeypatch, provider_source: Path
) -> None:
    root = tmp_path / "claude" / "skills"
    (root / "handmade").mkdir(parents=True)
    (root / "handmade" / "SKILL.md").write_text("mine")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    summary = ui._install_harness_skills(
        (_registration(provider_source),), ui.claude_skills_dir
    )
    assert summary["providers"] == 1
    assert (root / "handmade" / "SKILL.md").read_text() == "mine"
    ui._install_harness_skills((), ui.claude_skills_dir)
    assert not (root / "demo").exists()
    assert (root / "handmade" / "SKILL.md").exists()
