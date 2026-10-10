"""AU-DEV-R003.3.2: materialize registered skill providers under the Codex path."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.core import unified_install as ui
from agent_utilities.core.provider_materialization import (
    MANAGED_PROVIDER_MARKER,
    read_managed_provider_marker,
)
from agent_utilities.core.providers import ProviderRegistration


def _registration(source: Path, name: str = "demo") -> ProviderRegistration:
    return ProviderRegistration(
        name=name,
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


@pytest.mark.spec("AU-DEV-R003.3.2")
def test_registered_provider_installed_under_codex_with_marker(
    tmp_path: Path, monkeypatch, provider_source: Path
) -> None:
    home = tmp_path / "codex"
    monkeypatch.setenv("CODEX_HOME", str(home))

    summary = ui._install_codex_skills((_registration(provider_source),))

    provider_dir = home / "skills" / "demo"
    assert (provider_dir / MANAGED_PROVIDER_MARKER).is_file()
    assert read_managed_provider_marker(provider_dir, provider="demo", leg="skills")
    assert summary["providers"] == 1
    assert summary["failed"] == 0


@pytest.mark.spec("AU-DEV-R003.3.2")
def test_unsafe_codex_path_is_refused_not_written(
    tmp_path: Path, monkeypatch, provider_source: Path
) -> None:
    home = tmp_path / "codex"
    home.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    (home / "skills").symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("CODEX_HOME", str(home))

    summary = ui._install_codex_skills((_registration(provider_source),))

    assert summary["failed"] == 1
    assert list(real.iterdir()) == []


@pytest.mark.spec("AU-DEV-R003.3.2")
def test_install_unified_reports_codex_leg(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    seen: list[tuple] = []
    monkeypatch.setattr(ui, "provider_registrations", lambda _group: ())
    monkeypatch.setattr(
        ui, "_install_codex_skills", lambda regs: seen.append(regs) or {"providers": 0}
    )
    for name in ("unified_skills_dir", "unified_prompts_dir", "unified_ontologies_dir"):
        monkeypatch.setattr(ui, name, lambda n=name: tmp_path / n)
    monkeypatch.setattr(
        ui, "_own_source", lambda leg: (_ for _ in ()).throw(ValueError("skip"))
    )

    result = ui.install_unified()

    assert seen == [()]
    assert result["codex_skills"] == {"providers": 0}
