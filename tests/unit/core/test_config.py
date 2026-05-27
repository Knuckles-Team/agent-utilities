"""CONCEPT:OS-5.0"""

import os

import pytest

from agent_utilities.core.config import AgentConfig, get_env_file


@pytest.mark.concept("CONCEPT:OS-5.0")
def test_agent_config_defaults():
    orig_host = os.environ.pop("HOST", None)
    orig_port = os.environ.pop("PORT", None)
    try:
        config = AgentConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.routing_strategy == "hybrid"
        assert config.tool_guard_mode == "strict"
    finally:
        if orig_host is not None:
            os.environ["HOST"] = orig_host
        if orig_port is not None:
            os.environ["PORT"] = orig_port


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


@pytest.mark.concept("CONCEPT:OS-5.0")
def test_lazy_module_level_getattr():
    from agent_utilities.core.config import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_LLM_PROVIDER
    assert DEFAULT_HOST == "0.0.0.0"
    assert DEFAULT_PORT == 9000
    assert DEFAULT_LLM_PROVIDER == "openai" or DEFAULT_LLM_PROVIDER is not None


@pytest.mark.concept("CONCEPT:OS-5.0")
def test_lazy_mcp_utilities_getattr():
    from agent_utilities.mcp_utilities import DEFAULT_LLM_PROVIDER
    assert DEFAULT_LLM_PROVIDER == "openai" or DEFAULT_LLM_PROVIDER is not None
