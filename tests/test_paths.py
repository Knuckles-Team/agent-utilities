"""Platform config/data/cache/log directory resolution."""

from __future__ import annotations

import os

import pytest

from agent_utilities.core import paths

_DIRS = {
    "AGENT_UTILITIES_CONFIG_DIR": "config",
    "AGENT_UTILITIES_DATA_DIR": "data",
    "AGENT_UTILITIES_CACHE_DIR": "cache",
    "AGENT_UTILITIES_LOG_DIR": "log",
}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for name in ("GRAPH_OS_CONFIG_DIR", *_DIRS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(paths, "_legacy_warned", False)


@pytest.fixture
def overridden(monkeypatch, tmp_path):
    for env, sub in _DIRS.items():
        monkeypatch.setenv(env, str(tmp_path / sub))
    return tmp_path


@pytest.fixture
def warnings(monkeypatch) -> list[tuple]:
    calls: list[tuple] = []
    monkeypatch.setattr(
        paths._platform_logger, "warning", lambda msg, *a: calls.append(a)
    )
    return calls


def test_env_overrides_and_ensure_dirs(overridden):
    assert paths.config_dir() == overridden / "config"
    assert (
        paths.runtime_secrets_path() == overridden / "config" / "runtime-secrets.json"
    )
    assert paths.data_dir() == overridden / "data"
    assert paths.cache_dir() == overridden / "cache"
    assert paths.log_dir() == overridden / "log"
    assert paths.kg_db_path() == overridden / "data" / "kg" / "knowledge_graph.db"
    paths.ensure_dirs()
    for sub in ("config", "data/kg", "cache", "log"):
        assert (overridden / sub).is_dir()


@pytest.mark.skipif(
    not hasattr(os, "getuid") or os.getuid() == 0, reason="needs non-root"
)
def test_ensure_dirs_names_the_unwritable_setting(overridden):
    locked = overridden / "cache"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        with pytest.raises(paths.RuntimeDirectoryNotWritableError) as exc:
            paths.ensure_dirs()
        assert "AGENT_UTILITIES_CACHE_DIR" in str(exc.value)
        assert str(locked) in str(exc.value)
    finally:
        locked.chmod(0o755)


def test_graph_os_env_override_wins(monkeypatch, tmp_path):
    override = tmp_path / "custom"
    override.mkdir()
    (override / "workspace.yml").write_text("name: x\n")
    monkeypatch.setenv("GRAPH_OS_CONFIG_DIR", str(override))
    assert paths.platform_config_file("workspace.yml") == override / "workspace.yml"
    assert paths.platform_config_dir("workspace.yml") == override
    assert paths.config_dir() == override


def test_new_xdg_path_preferred_without_warning(tmp_path, warnings):
    for name in ("graph-os", "agent-utilities"):
        (tmp_path / "xdg" / name).mkdir(parents=True)
        (tmp_path / "xdg" / name / "config.json").write_text("{}")
    assert paths.platform_config_dir() == tmp_path / "xdg" / "graph-os"
    assert warnings == []


def test_neither_present_returns_new_path_without_creating(tmp_path, warnings):
    new = tmp_path / "xdg" / "graph-os"
    assert paths.platform_config_file("workspace.yml") == new / "workspace.yml"
    assert paths.platform_config_dir() == new
    assert not new.exists()
    assert warnings == []


def test_legacy_override_falls_back_and_warns_once(monkeypatch, tmp_path, warnings):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "config.json").write_text("{}")
    (legacy / "workspace.yml").write_text("name: x\n")
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(legacy))
    assert paths.platform_config_file("workspace.yml") == legacy / "workspace.yml"
    assert paths.platform_config_dir() == legacy
    assert len(warnings) == 1
    assert warnings[0][:2] == (legacy, tmp_path / "xdg" / "graph-os")


def test_legacy_xdg_fallback_is_per_file(tmp_path, warnings):
    new = tmp_path / "xdg" / "graph-os"
    new.mkdir(parents=True)
    (new / "workspace.yml").write_text("name: x\n")
    legacy = tmp_path / "xdg" / "agent-utilities"
    legacy.mkdir(parents=True)
    (legacy / "config.json").write_text("{}")
    assert paths.platform_config_file("workspace.yml") == new / "workspace.yml"
    assert paths.platform_config_dir() == legacy
    assert warnings == [(legacy / "config.json", new, "GRAPH_OS_CONFIG_DIR")]
