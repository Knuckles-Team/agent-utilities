"""Platform config/data/cache/log directory resolution."""

from __future__ import annotations

import os

import pytest

from agent_utilities.core import paths

_DIRS = {
    "ARBORA_CONFIG_DIR": "config",
    "AGENT_UTILITIES_DATA_DIR": "data",
    "AGENT_UTILITIES_CACHE_DIR": "cache",
    "AGENT_UTILITIES_LOG_DIR": "log",
}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for name in ("GRAPH_OS_CONFIG_DIR", "AGENT_UTILITIES_CONFIG_DIR", *_DIRS):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def overridden(monkeypatch, tmp_path):
    for env, sub in _DIRS.items():
        monkeypatch.setenv(env, str(tmp_path / sub))
    return tmp_path


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


def test_arbora_env_override_wins(monkeypatch, tmp_path):
    override = tmp_path / "custom"
    override.mkdir()
    (override / "workspace.yml").write_text("name: x\n")
    monkeypatch.setenv("ARBORA_CONFIG_DIR", str(override))
    assert paths.platform_config_file("workspace.yml") == override / "workspace.yml"
    assert paths.platform_config_dir("workspace.yml") == override
    assert paths.config_dir() == override


def test_xdg_path_is_arbora_only(tmp_path):
    for name in ("arbora", "agent-utilities", "graph-os"):
        (tmp_path / "xdg" / name).mkdir(parents=True)
        (tmp_path / "xdg" / name / "config.json").write_text("{}")
    assert paths.platform_config_dir() == tmp_path / "xdg" / "arbora"


def test_missing_dir_returns_arbora_path_without_creating(tmp_path):
    new = tmp_path / "xdg" / "arbora"
    assert paths.platform_config_file("workspace.yml") == new / "workspace.yml"
    assert paths.platform_config_dir() == new
    assert not new.exists()


def test_old_config_settings_are_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPH_OS_CONFIG_DIR", str(tmp_path / "old"))
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path / "au"))
    assert paths.platform_config_dir() == tmp_path / "xdg" / "arbora"
