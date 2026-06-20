"""Public load_config() — the shared XDG config loader for the agent fleet (ECO-4.82)."""

from __future__ import annotations

import json
import os

from agent_utilities.core import config


def test_load_config_is_public_and_callable():
    # Exposed both directly and via the agent-facing facade agents import from.
    from agent_utilities.mcp_utilities import load_config as facade_load_config

    assert callable(config.load_config)
    assert facade_load_config is config.load_config


def test_load_config_injects_xdg_json(tmp_path, monkeypatch):
    """A config.json key is upper-cased into os.environ so setting()/getenv see it.

    An explicit AGENT_UTILITIES_CONFIG_DIR override bypasses the hermetic
    pytest/TESTING skip (integration-style), exercising the real injection path.
    """
    cfg_dir = tmp_path / "agent-utilities"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(
        json.dumps({"my_verbose_test_key": "hello", "mcp_tool_mode": "verbose"})
    )
    monkeypatch.delenv("AGENT_UTILITIES_TESTING", raising=False)
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(cfg_dir))
    monkeypatch.delenv("MY_VERBOSE_TEST_KEY", raising=False)
    monkeypatch.delenv("MCP_TOOL_MODE", raising=False)

    config.load_config(reload=True)

    assert os.environ.get("MY_VERBOSE_TEST_KEY") == "hello"
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
