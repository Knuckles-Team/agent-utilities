"""AU-DEV-R003.3.1: the Codex skills path resolver."""

from pathlib import Path

import pytest

from agent_utilities.core.unified_install import CodexSkillsPathError, codex_skills_dir


@pytest.mark.spec("AU-DEV-R003.3.1")
def test_codex_home_override_and_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert codex_skills_dir() == tmp_path / "skills"
    monkeypatch.delenv("CODEX_HOME")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert codex_skills_dir() == tmp_path / ".codex" / "skills"


@pytest.mark.spec("AU-DEV-R003.3.1")
def test_codex_skills_refuses_symlink_relative_and_file(
    tmp_path: Path, monkeypatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    (home / "skills").symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("CODEX_HOME", str(home))
    with pytest.raises(CodexSkillsPathError):
        codex_skills_dir()
    monkeypatch.setenv("CODEX_HOME", "relative/dir")
    with pytest.raises(CodexSkillsPathError):
        codex_skills_dir()
    filehome = tmp_path / "fh"
    filehome.mkdir()
    (filehome / "skills").write_text("x")
    monkeypatch.setenv("CODEX_HOME", str(filehome))
    with pytest.raises(CodexSkillsPathError):
        codex_skills_dir()
