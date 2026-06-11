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


@pytest.mark.concept("CONCEPT:KG-2.7")
def test_kg_golden_loop_flags_default_off():
    # The golden-loop family is opt-in: every KG_GOLDEN_* flag now lives on
    # AgentConfig (off bare os.environ) and defaults to the conservative value.
    for k in (
        "KG_GOLDEN_LOOP",
        "KG_GOLDEN_DISTILL",
        "KG_GOLDEN_BREADTH",
        "KG_GOLDEN_STANDARDIZE",
        "KG_GOLDEN_AUTO_MERGE",
    ):
        os.environ.pop(k, None)
    c = AgentConfig()
    assert c.kg_golden_loop is False
    assert c.kg_golden_auto_merge is False
    assert c.kg_golden_loop_interval == 3600.0
    assert c.kg_golden_loop_topics == 5
    assert c.kg_golden_merge_threshold is None


@pytest.mark.concept("CONCEPT:KG-2.7")
def test_kg_golden_loop_override_from_env():
    os.environ["KG_GOLDEN_LOOP"] = "1"
    os.environ["KG_GOLDEN_LOOP_TOPICS"] = "9"
    try:
        c = AgentConfig()
        assert c.kg_golden_loop is True
        assert c.kg_golden_loop_topics == 9
    finally:
        os.environ.pop("KG_GOLDEN_LOOP", None)
        os.environ.pop("KG_GOLDEN_LOOP_TOPICS", None)


@pytest.mark.concept("CONCEPT:KG-2.8")
def test_kg_dev_mode_default_off():
    # Production default: background daemons are on (dev mode off). This single
    # switch replaced the per-daemon KG_*_DAEMON env toggles.
    os.environ.pop("KG_DEV_MODE", None)
    assert AgentConfig().kg_dev_mode is False


@pytest.mark.concept("CONCEPT:KG-2.8")
def test_kg_dev_mode_override_from_env():
    os.environ["KG_DEV_MODE"] = "true"
    try:
        assert AgentConfig().kg_dev_mode is True
    finally:
        os.environ.pop("KG_DEV_MODE", None)


@pytest.mark.concept("CONCEPT:KG-2.8")
def test_kg_dev_mode_helper_reads_config(monkeypatch):
    # The engine's daemon gate reads the SAME typed config source of truth, so
    # all KG background daemons collapse behind this one switch.
    from agent_utilities.core import config as cfg_mod
    from agent_utilities.knowledge_graph.core import engine_tasks

    monkeypatch.setattr(cfg_mod.config, "kg_dev_mode", False, raising=False)
    assert engine_tasks._kg_dev_mode() is False
    monkeypatch.setattr(cfg_mod.config, "kg_dev_mode", True, raising=False)
    assert engine_tasks._kg_dev_mode() is True


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
def test_agent_config_langfuse_host():
    # LANGFUSE_HOST (the official Langfuse SDK var) is the only host var; the
    # deprecated LANGFUSE_BASE_URL fallback was removed (AHE-3.18 cleanup).
    os.environ["LANGFUSE_HOST"] = "https://custom-langfuse.arpa"
    os.environ["LANGFUSE_BASE_URL"] = "https://ignored-legacy.arpa"
    try:
        config = AgentConfig()
        assert config.langfuse_host == "https://custom-langfuse.arpa"
    finally:
        os.environ.pop("LANGFUSE_HOST", None)
        os.environ.pop("LANGFUSE_BASE_URL", None)


@pytest.mark.concept("CONCEPT:OS-5.0")
def test_lazy_module_level_getattr():
    from agent_utilities.core.config import (
        DEFAULT_HOST,
        DEFAULT_LLM_PROVIDER,
        DEFAULT_PORT,
    )

    assert DEFAULT_HOST == "0.0.0.0"
    assert DEFAULT_PORT == 9000
    assert DEFAULT_LLM_PROVIDER == "openai" or DEFAULT_LLM_PROVIDER is not None


@pytest.mark.concept("CONCEPT:OS-5.0")
def test_lazy_mcp_utilities_getattr():
    from agent_utilities.mcp_utilities import DEFAULT_LLM_PROVIDER

    assert DEFAULT_LLM_PROVIDER == "openai" or DEFAULT_LLM_PROVIDER is not None
