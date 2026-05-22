"""CONCEPT:OS-5.0"""

import os

import pytest

from agent_utilities.core.config import AgentConfig, get_env_file


@pytest.mark.concept("CONCEPT:OS-5.0")
def test_agent_config_defaults():
    config = AgentConfig()
    assert config.host == "0.0.0.0"
    assert config.port == 9000
    assert config.routing_strategy == "hybrid"
    assert config.tool_guard_mode == "strict"


@pytest.mark.concept("CONCEPT:OS-5.0")
def test_agent_config_overrides():
    os.environ["HOST"] = "1.2.3.4"
    os.environ["PORT"] = "8080"
    try:
        config = AgentConfig()
        assert config.host == "1.2.3.4"
        assert config.port == 8080
    finally:
        os.environ.pop("HOST", None)
        os.environ.pop("PORT", None)


@pytest.mark.concept("CONCEPT:OS-5.0")
def test_get_env_file_default():
    # When not in a specific package, it returns .env (absolute path)
    assert str(get_env_file()).endswith(".env")


@pytest.mark.concept("CONCEPT:OS-5.0")
def test_tool_guard_mode_override():
    os.environ["TOOL_GUARD_MODE"] = "on"
    try:
        config = AgentConfig()
        assert config.tool_guard_mode == "on"
    finally:
        os.environ.pop("TOOL_GUARD_MODE", None)


@pytest.mark.concept("CONCEPT:OS-5.0")
def test_sensitive_tool_patterns_defaults():
    config = AgentConfig()
    assert isinstance(config.sensitive_tool_patterns, list)
    assert r".*delete.*" in config.sensitive_tool_patterns
    assert r".*rm_.*" in config.sensitive_tool_patterns


@pytest.mark.concept("CONCEPT:OS-5.0")
def test_agent_config_langfuse_base_url():
    os.environ["LANGFUSE_BASE_URL"] = "https://custom-langfuse.arpa"
    try:
        config = AgentConfig()
        assert config.langfuse_host == "https://custom-langfuse.arpa"
    finally:
        os.environ.pop("LANGFUSE_BASE_URL", None)
