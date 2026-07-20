"""Public load_config() — the shared XDG config loader for the agent fleet (ECO-4.82)."""

from __future__ import annotations

import json
import os

from agent_utilities.core import config


def test_load_config_is_public_and_callable():
    assert callable(config.load_config)


def test_load_config_injects_xdg_json(tmp_path, monkeypatch):
    """A config.json key is upper-cased into os.environ so setting()/getenv see it.

    An explicit AGENT_UTILITIES_CONFIG_DIR override bypasses the hermetic
    pytest/TESTING skip (integration-style), exercising the real injection path.
    """
    cfg_dir = tmp_path / "agent-utilities"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(
        json.dumps({"mcp_tool_mode": "verbose"})
    )
    monkeypatch.delenv("AGENT_UTILITIES_TESTING", raising=False)
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(cfg_dir))
    monkeypatch.delenv("MCP_TOOL_MODE", raising=False)

    config.load_config(reload=True)

    # and it flows through the sanctioned read path
    assert config.setting("MCP_TOOL_MODE", "condensed") == "verbose"


def test_load_config_idempotent(monkeypatch):
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "true")
    # Under TESTING it is a deliberate no-op; calling repeatedly must not raise.
    config.load_config()
    config.load_config()
    config.load_config(reload=True)


def test_real_env_wins_over_config_json(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "agent-utilities"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(json.dumps({"mcp_tool_mode": "verbose"}))
    monkeypatch.delenv("AGENT_UTILITIES_TESTING", raising=False)
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("MCP_TOOL_MODE", "both")  # real env set first

    config.load_config(reload=True)

    # config.json must not clobber an already-set environment variable
    assert os.environ.get("MCP_TOOL_MODE") == "both"


def test_reload_replaces_only_values_injected_by_xdg(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "agent-utilities"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "config.json"
    cfg_file.write_text(json.dumps({"mcp_tool_mode": "intent"}))
    monkeypatch.delenv("AGENT_UTILITIES_TESTING", raising=False)
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(cfg_dir))
    monkeypatch.delenv("MCP_TOOL_MODE", raising=False)

    config.load_config(reload=True)
    assert os.environ["MCP_TOOL_MODE"] == "intent"

    cfg_file.write_text(json.dumps({"mcp_tool_mode": "verbose"}))
    config.load_config(reload=True)
    assert os.environ["MCP_TOOL_MODE"] == "verbose"

    # A later process-environment override is not owned by XDG and keeps
    # precedence across subsequent document reloads.
    os.environ["MCP_TOOL_MODE"] = "both"
    cfg_file.write_text(json.dumps({"mcp_tool_mode": "intent"}))
    config.load_config(reload=True)
    assert os.environ["MCP_TOOL_MODE"] == "both"


def test_save_config_atomically_updates_stable_typed_proxy(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "agent-utilities"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(json.dumps({"enable_otel": False}))
    monkeypatch.delenv("AGENT_UTILITIES_TESTING", raising=False)
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(cfg_dir))
    monkeypatch.delenv("ENABLE_OTEL", raising=False)
    config._LAZY_CACHE.clear()

    config.load_config(reload=True)
    stable = config.config
    previous_snapshot = config._LAZY_CACHE["_config"]
    assert stable.enable_otel is False

    config.save_config_item("enable_otel", True)

    assert config.config is stable
    assert previous_snapshot.enable_otel is False
    assert config.config.enable_otel is True
    assert config.DEFAULT_ENABLE_OTEL is True
    assert "OTEL_SDK_DISABLED" not in os.environ
