"""Graph OS platform config location with the one-release legacy fallback."""

from __future__ import annotations

import logging

import pytest

from agent_utilities.core import paths


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("GRAPH_OS_CONFIG_DIR", raising=False)
    monkeypatch.delenv("AGENT_UTILITIES_CONFIG_DIR", raising=False)
    monkeypatch.setattr(paths, "_legacy_warned", False)


@pytest.mark.concept("CONCEPT:AU-OS.config.agent-os-infrastructure")
def test_env_override_wins(monkeypatch, tmp_path):
    override = tmp_path / "custom"
    override.mkdir()
    (override / "workspace.yml").write_text("name: x\n")
    monkeypatch.setenv("GRAPH_OS_CONFIG_DIR", str(override))
    assert paths.platform_config_file("workspace.yml") == override / "workspace.yml"
    assert paths.platform_config_dir("workspace.yml") == override


@pytest.mark.concept("CONCEPT:AU-OS.config.agent-os-infrastructure")
def test_new_xdg_path(tmp_path, caplog):
    new = tmp_path / "xdg" / "graph-os"
    new.mkdir(parents=True)
    (new / "config.json").write_text("{}")
    legacy = tmp_path / "xdg" / "agent-utilities"
    legacy.mkdir(parents=True)
    (legacy / "config.json").write_text("{}")
    with caplog.at_level(logging.WARNING):
        assert paths.platform_config_dir() == new
    assert not caplog.records


@pytest.mark.concept("CONCEPT:AU-OS.config.agent-os-infrastructure")
def _record_warnings(monkeypatch) -> list[tuple]:
    # AU's log redaction masks paths on the record itself; capture the args.
    calls: list[tuple] = []
    monkeypatch.setattr(
        paths._platform_logger, "warning", lambda msg, *args: calls.append(args)
    )
    return calls


@pytest.mark.concept("CONCEPT:AU-OS.config.agent-os-infrastructure")
def test_legacy_fallback_warns_once(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "config.json").write_text("{}")
    (legacy / "workspace.yml").write_text("name: x\n")
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(legacy))
    calls = _record_warnings(monkeypatch)
    assert paths.platform_config_file("workspace.yml") == legacy / "workspace.yml"
    assert paths.platform_config_dir() == legacy
    assert len(calls) == 1
    assert calls[0][0] == legacy
    assert calls[0][1] == tmp_path / "xdg" / "graph-os"


@pytest.mark.concept("CONCEPT:AU-OS.config.agent-os-infrastructure")
def test_default_legacy_xdg_fallback_per_file(monkeypatch, tmp_path):
    new = tmp_path / "xdg" / "graph-os"
    new.mkdir(parents=True)
    (new / "workspace.yml").write_text("name: x\n")
    legacy = tmp_path / "xdg" / "agent-utilities"
    legacy.mkdir(parents=True)
    (legacy / "config.json").write_text("{}")
    calls = _record_warnings(monkeypatch)
    assert paths.platform_config_file("workspace.yml") == new / "workspace.yml"
    assert paths.platform_config_dir() == legacy
    assert calls == [(legacy / "config.json", new, "GRAPH_OS_CONFIG_DIR")]


@pytest.mark.concept("CONCEPT:AU-OS.config.agent-os-infrastructure")
def test_explicit_legacy_override_still_honoured(monkeypatch, tmp_path):
    legacy = tmp_path / "explicit"
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(legacy))
    assert paths.platform_config_dir() == legacy


@pytest.mark.concept("CONCEPT:AU-OS.config.agent-os-infrastructure")
def test_neither_present_returns_new_path_without_creating(tmp_path, caplog):
    new = tmp_path / "xdg" / "graph-os"
    with caplog.at_level(logging.WARNING):
        assert paths.platform_config_file("workspace.yml") == new / "workspace.yml"
        assert paths.platform_config_dir() == new
    assert not new.exists()
    assert not caplog.records


@pytest.mark.concept("CONCEPT:AU-OS.config.agent-os-infrastructure")
def test_config_dir_and_xdg_config_file_follow_the_resolver(monkeypatch, tmp_path):
    from agent_utilities.core.config import _xdg_config_file

    override = tmp_path / "graphos"
    monkeypatch.setenv("GRAPH_OS_CONFIG_DIR", str(override))
    assert paths.config_dir() == override
    assert _xdg_config_file() == override / "config.json"
