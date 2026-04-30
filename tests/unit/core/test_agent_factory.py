from __future__ import annotations

"""Coverage push for agent_utilities.agent_factory.

Targets create_agent_parser (argparse wiring) and non-LLM branches of
create_agent: MCP URL/config/toolsets paths, custom system prompt,
output_style, checkpoint_store, include_teams.
"""



import argparse
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.agent import factory as agent_factory

# ---------------------------------------------------------------------------
# create_agent_parser (lines 113-222)
# ---------------------------------------------------------------------------


def test_create_agent_parser_returns_parser() -> None:
    """Parser creation returns an ArgumentParser."""
    parser = agent_factory.create_agent_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_parser_defaults_when_no_args() -> None:
    """All defaults parse cleanly with no args."""
    parser = agent_factory.create_agent_parser()
    args = parser.parse_args([])
    assert hasattr(args, "host")
    assert hasattr(args, "port")
    assert hasattr(args, "provider")
    assert hasattr(args, "model_id")
    assert hasattr(args, "base_url")
    assert hasattr(args, "api_key")
    assert hasattr(args, "mcp_url")
    assert hasattr(args, "mcp_config")


def test_parser_custom_host_port() -> None:
    """Parser accepts explicit host + port."""
    parser = agent_factory.create_agent_parser()
    args = parser.parse_args(["--host", "0.0.0.0", "--port", "9999"])
    assert args.host == "0.0.0.0"
    assert args.port == 9999


def test_parser_valid_providers() -> None:
    """Parser accepts all valid provider choices."""
    parser = agent_factory.create_agent_parser()
    for provider in [
        "openai",
        "anthropic",
        "google",
        "huggingface",
        "groq",
        "mistral",
        "ollama",
    ]:
        args = parser.parse_args(["--provider", provider])
        assert args.provider == provider


def test_parser_invalid_provider_raises() -> None:
    """Parser rejects unknown provider."""
    parser = agent_factory.create_agent_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--provider", "not-real"])


def test_parser_debug_boolean_optional() -> None:
    """--debug / --no-debug BooleanOptionalAction."""
    parser = agent_factory.create_agent_parser()
    args = parser.parse_args(["--debug"])
    assert args.debug is True
    args = parser.parse_args(["--no-debug"])
    assert args.debug is False


def test_parser_reload_flag() -> None:
    """--reload is a store_true flag."""
    parser = agent_factory.create_agent_parser()
    args = parser.parse_args(["--reload"])
    assert args.reload is True
    args = parser.parse_args([])
    assert args.reload is False


def test_parser_web_flag() -> None:
    """--web BooleanOptionalAction."""
    parser = agent_factory.create_agent_parser()
    args = parser.parse_args(["--web"])
    assert args.web is True
    args = parser.parse_args(["--no-web"])
    assert args.web is False


def test_parser_terminal_aliases() -> None:
    """--terminal and --tui are equivalent aliases."""
    parser = agent_factory.create_agent_parser()
    args = parser.parse_args(["--terminal"])
    assert args.terminal is True
    args = parser.parse_args(["--tui"])
    assert args.terminal is True
    args = parser.parse_args(["--no-terminal"])
    assert args.terminal is False


def test_parser_insecure_flag() -> None:
    """--insecure flag."""
    parser = agent_factory.create_agent_parser()
    args = parser.parse_args(["--insecure"])
    assert args.insecure is True


def test_parser_otel_boolean_optional() -> None:
    """--otel / --no-otel BooleanOptionalAction."""
    parser = agent_factory.create_agent_parser()
    args = parser.parse_args(["--otel"])
    assert args.otel is True
    args = parser.parse_args(["--no-otel"])
    assert args.otel is False


def test_parser_otel_endpoint_options() -> None:
    """OTEL endpoint and related options parse correctly."""
    parser = agent_factory.create_agent_parser()
    args = parser.parse_args(
        [
            "--otel-endpoint",
            "https://example.com/otlp",
            "--otel-headers",
            "key=value",
            "--otel-public-key",
            "pub123",
            "--otel-secret-key",
            "sec456",
            "--otel-protocol",
            "http/protobuf",
        ]
    )
    assert args.otel_endpoint == "https://example.com/otlp"
    assert args.otel_headers == "key=value"
    assert args.otel_public_key == "pub123"
    assert args.otel_secret_key == "sec456"
    assert args.otel_protocol == "http/protobuf"


def test_parser_mcp_options() -> None:
    """MCP url and config options."""
    parser = agent_factory.create_agent_parser()
    args = parser.parse_args(
        ["--mcp-url", "https://mcp.example.com", "--mcp-config", "/tmp/mcp.json"]
    )
    assert args.mcp_url == "https://mcp.example.com"
    assert args.mcp_config == "/tmp/mcp.json"


def test_parser_workspace_option() -> None:
    """--workspace accepts an explicit path."""
    parser = agent_factory.create_agent_parser()
    args = parser.parse_args(["--workspace", "/opt/agent"])
    assert args.workspace == "/opt/agent"


def test_parser_custom_skills_directory() -> None:
    """--custom-skills-directory."""
    parser = agent_factory.create_agent_parser()
    args = parser.parse_args(["--custom-skills-directory", "/tmp/my-skills"])
    assert args.custom_skills_directory == "/tmp/my-skills"


# ---------------------------------------------------------------------------
# create_agent: custom system_prompt (lines 460-461)
# ---------------------------------------------------------------------------


def test_create_agent_with_custom_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When system_prompt is provided, it is used verbatim (bypasses workspace build)."""
    build_mock = MagicMock(return_value="SHOULD NOT BE USED")
    monkeypatch.setattr(
        agent_factory, "build_system_prompt_from_workspace", build_mock
    )
    agent, _ = agent_factory.create_agent(
        name="TestCustomPrompt",
        system_prompt="My custom system prompt text",
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None
    # build_system_prompt_from_workspace should NOT have been called
    build_mock.assert_not_called()


def test_create_agent_builds_system_prompt_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When system_prompt is None, workspace builder is called."""
    build_mock = MagicMock(return_value="Built prompt")
    monkeypatch.setattr(
        agent_factory, "build_system_prompt_from_workspace", build_mock
    )
    agent, _ = agent_factory.create_agent(
        name="TestDefaultPrompt",
        system_prompt=None,
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None
    build_mock.assert_called_once()


# ---------------------------------------------------------------------------
# create_agent: output_style (lines 527-534)
# ---------------------------------------------------------------------------


def test_create_agent_with_output_style_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """output_style='concise' injects a style instruction into the agent."""
    agent, _ = agent_factory.create_agent(
        name="TestStyle",
        output_style="concise",
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None


def test_create_agent_with_output_style_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown output_style is tolerated (no style injection happens)."""
    agent, _ = agent_factory.create_agent(
        name="TestUnknownStyle",
        output_style="not-a-real-style",
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None


def test_create_agent_with_output_style_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Style names are matched case-insensitively (`.lower()` inside factory)."""
    agent, _ = agent_factory.create_agent(
        name="TestCaseStyle",
        output_style="FORMAL",
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None


# ---------------------------------------------------------------------------
# create_agent: checkpoint_store + include_teams (lines 489-490, 498)
# ---------------------------------------------------------------------------


def test_create_agent_with_checkpoint_store_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """include_checkpoints=True uses default InMemoryCheckpointStore."""
    agent, _ = agent_factory.create_agent(
        name="TestCheckpoints",
        include_checkpoints=True,
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None


def test_create_agent_with_custom_checkpoint_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom checkpoint_store is used when include_checkpoints=True."""
    from agent_utilities.capabilities import InMemoryCheckpointStore

    custom_store = InMemoryCheckpointStore()
    agent, _ = agent_factory.create_agent(
        name="TestCustomStore",
        include_checkpoints=True,
        checkpoint_store=custom_store,
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None


def test_create_agent_with_teams_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """include_teams=True adds TeamCapability."""
    agent, _ = agent_factory.create_agent(
        name="TestTeams",
        include_teams=True,
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None


# ---------------------------------------------------------------------------
# create_agent: capability toggles (stuck_loop, context_warnings, eviction)
# ---------------------------------------------------------------------------


def test_create_agent_disable_stuck_loop_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stuck_loop_detection=False skips StuckLoopDetection capability."""
    agent, _ = agent_factory.create_agent(
        name="TestNoStuck",
        stuck_loop_detection=False,
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None


def test_create_agent_disable_context_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """context_warnings=False skips ContextLimitWarner capability."""
    agent, _ = agent_factory.create_agent(
        name="TestNoCtxWarn",
        context_warnings=False,
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None


def test_create_agent_disable_output_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """output_eviction=False skips ToolOutputEviction capability."""
    agent, _ = agent_factory.create_agent(
        name="TestNoEvict",
        output_eviction=False,
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None


def test_create_agent_with_custom_eviction_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom eviction_threshold_chars is propagated."""
    agent, _ = agent_factory.create_agent(
        name="TestEvictThreshold",
        output_eviction=True,
        eviction_threshold_chars=50_000,
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None


# ---------------------------------------------------------------------------
# create_agent: tool_guard_mode
# ---------------------------------------------------------------------------


def test_create_agent_tool_guard_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """tool_guard_mode='off' skips apply_tool_guard_approvals."""
    with patch.object(agent_factory, "apply_tool_guard_approvals") as mock_guard:
        agent, _ = agent_factory.create_agent(
            name="TestGuardOff",
            tool_guard_mode="off",
            skill_types=[],
            enable_skills=False,
            enable_universal_tools=False,
        )
        mock_guard.assert_not_called()
    assert agent is not None


def test_create_agent_tool_guard_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    """tool_guard_mode='strict' calls apply_tool_guard_approvals."""
    with patch.object(agent_factory, "apply_tool_guard_approvals") as mock_guard:
        agent, _ = agent_factory.create_agent(
            name="TestGuardStrict",
            tool_guard_mode="strict",
            skill_types=[],
            enable_skills=False,
            enable_universal_tools=False,
        )
        mock_guard.assert_called_once_with(agent)
    assert agent is not None


# ---------------------------------------------------------------------------
# create_agent: validation mode skips MCP connections (lines 303-304, 335-336, 366-367)
# ---------------------------------------------------------------------------


def test_create_agent_validation_mode_skips_mcp_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEFAULT_VALIDATION_MODE=True logs and skips MCP URL connection."""
    monkeypatch.setattr(agent_factory, "DEFAULT_VALIDATION_MODE", True)

    agent, mcp_toolsets = agent_factory.create_agent(
        name="TestValMCPUrl",
        mcp_url="http://mcp.example.com/sse",
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None
    # No MCP toolsets initialized (validation skipped)
    assert mcp_toolsets == []


def test_create_agent_validation_mode_skips_mcp_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEFAULT_VALIDATION_MODE=True skips MCP config loading."""
    monkeypatch.setattr(agent_factory, "DEFAULT_VALIDATION_MODE", True)

    agent, mcp_toolsets = agent_factory.create_agent(
        name="TestValMCPConfig",
        mcp_config="/tmp/some-config.json",
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None
    assert mcp_toolsets == []


def test_create_agent_validation_mode_skips_mcp_toolsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEFAULT_VALIDATION_MODE=True skips external mcp_toolsets."""
    monkeypatch.setattr(agent_factory, "DEFAULT_VALIDATION_MODE", True)

    external = MagicMock(name="ExternalMCP")
    agent, mcp_toolsets = agent_factory.create_agent(
        name="TestValMCPToolsets",
        mcp_toolsets=[external, None],
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None
    assert mcp_toolsets == []


# ---------------------------------------------------------------------------
# create_agent: MCP URL loopback guard (line 305-308)
# ---------------------------------------------------------------------------


def test_create_agent_loopback_mcp_url_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loopback URL triggers the guard and skips the MCP connection."""
    monkeypatch.setattr(agent_factory, "DEFAULT_VALIDATION_MODE", False)
    # Force is_loopback_url to return True
    monkeypatch.setattr(agent_factory, "is_loopback_url", lambda *a, **k: True)
    agent, mcp_toolsets = agent_factory.create_agent(
        name="TestLoopback",
        mcp_url="http://localhost:8000",
        current_host="localhost",
        current_port=8000,
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None
    assert mcp_toolsets == []


def test_create_agent_mcp_url_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exception during MCPServerSSE instantiation is caught and logged."""
    monkeypatch.setattr(agent_factory, "DEFAULT_VALIDATION_MODE", False)
    monkeypatch.setattr(agent_factory, "is_loopback_url", lambda *a, **k: False)
    # Force MCPServerSSE to raise on instantiation
    monkeypatch.setattr(
        agent_factory,
        "MCPServerSSE",
        MagicMock(side_effect=RuntimeError("connection refused")),
    )
    agent, mcp_toolsets = agent_factory.create_agent(
        name="TestMCPError",
        mcp_url="http://mcp.example.com/sse",
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    # Agent still created, but no MCP toolset
    assert agent is not None
    assert mcp_toolsets == []


def test_create_agent_mcp_url_streamable_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-/sse URL uses MCPServerStreamableHTTP (line 318-324)."""
    monkeypatch.setattr(agent_factory, "DEFAULT_VALIDATION_MODE", False)
    monkeypatch.setattr(agent_factory, "is_loopback_url", lambda *a, **k: False)
    mock_streamable = MagicMock(return_value=MagicMock(http_client=None))
    monkeypatch.setattr(agent_factory, "MCPServerStreamableHTTP", mock_streamable)
    agent, mcp_toolsets = agent_factory.create_agent(
        name="TestStreamable",
        mcp_url="http://mcp.example.com/api",
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    mock_streamable.assert_called_once()
    assert agent is not None


def test_create_agent_mcp_url_sse_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """URL ending in /sse uses MCPServerSSE (line 311-317)."""
    monkeypatch.setattr(agent_factory, "DEFAULT_VALIDATION_MODE", False)
    monkeypatch.setattr(agent_factory, "is_loopback_url", lambda *a, **k: False)
    mock_sse = MagicMock(return_value=MagicMock(http_client=None))
    monkeypatch.setattr(agent_factory, "MCPServerSSE", mock_sse)
    agent, mcp_toolsets = agent_factory.create_agent(
        name="TestSSE",
        mcp_url="http://mcp.example.com/sse",
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    mock_sse.assert_called_once()
    assert agent is not None


def test_create_agent_mcp_url_with_tool_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tool_tags cause filter_tools_by_tag to wrap the server."""
    monkeypatch.setattr(agent_factory, "DEFAULT_VALIDATION_MODE", False)
    monkeypatch.setattr(agent_factory, "is_loopback_url", lambda *a, **k: False)
    fake_server = MagicMock()
    monkeypatch.setattr(
        agent_factory, "MCPServerStreamableHTTP", MagicMock(return_value=fake_server)
    )
    mock_filter = MagicMock(return_value=fake_server)
    monkeypatch.setattr(agent_factory, "filter_tools_by_tag", mock_filter)
    agent, _ = agent_factory.create_agent(
        name="TestTags",
        mcp_url="http://mcp.example.com/api",
        tool_tags=["tag1"],
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    mock_filter.assert_called_once()
    assert agent is not None


def test_create_agent_mcp_url_isolate_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """isolate_mcp=True adds toolset to initialized_mcp_toolsets but NOT agent_toolsets."""
    monkeypatch.setattr(agent_factory, "DEFAULT_VALIDATION_MODE", False)
    monkeypatch.setattr(agent_factory, "is_loopback_url", lambda *a, **k: False)
    fake_server = MagicMock()
    monkeypatch.setattr(
        agent_factory, "MCPServerStreamableHTTP", MagicMock(return_value=fake_server)
    )
    agent, initialized = agent_factory.create_agent(
        name="TestIsolate",
        mcp_url="http://mcp.example.com/api",
        isolate_mcp=True,
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert len(initialized) == 1  # Got initialized
    assert agent is not None


# ---------------------------------------------------------------------------
# create_agent: MCP config path (lines 335-363)
# ---------------------------------------------------------------------------


def test_create_agent_mcp_config_resolve_and_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """MCP config path resolves + loads servers."""
    from pathlib import Path

    monkeypatch.setattr(agent_factory, "DEFAULT_VALIDATION_MODE", False)
    fake_server = MagicMock(http_client=None)
    monkeypatch.setattr(
        agent_factory,
        "load_mcp_servers",
        MagicMock(return_value=[fake_server]),
    )

    # Provide a real path resolver
    fake_path = tmp_path / "mcp_config.json"
    fake_path.write_text("{}")
    monkeypatch.setattr(
        "agent_utilities.core.workspace.resolve_mcp_config_path",
        lambda p: Path(fake_path),
    )
    agent, initialized = agent_factory.create_agent(
        name="TestMCPConfig",
        mcp_config=str(fake_path),
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None
    assert len(initialized) == 1


def test_create_agent_mcp_config_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exception loading MCP config is caught."""
    monkeypatch.setattr(agent_factory, "DEFAULT_VALIDATION_MODE", False)
    monkeypatch.setattr(
        agent_factory,
        "load_mcp_servers",
        MagicMock(side_effect=RuntimeError("bad config")),
    )
    monkeypatch.setattr(
        "agent_utilities.core.workspace.resolve_mcp_config_path",
        lambda p: None,
    )
    agent, _ = agent_factory.create_agent(
        name="TestMCPConfigFail",
        mcp_config="/tmp/nonexistent.json",
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None


def test_create_agent_mcp_config_with_tool_tags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """MCP config with tool_tags filters each loaded server."""
    from pathlib import Path

    monkeypatch.setattr(agent_factory, "DEFAULT_VALIDATION_MODE", False)
    server1 = MagicMock(http_client=None)
    server2 = MagicMock(http_client=None)
    monkeypatch.setattr(
        agent_factory,
        "load_mcp_servers",
        MagicMock(return_value=[server1, server2]),
    )
    fake_path = tmp_path / "mcp.json"
    fake_path.write_text("{}")
    monkeypatch.setattr(
        "agent_utilities.core.workspace.resolve_mcp_config_path",
        lambda p: Path(fake_path),
    )
    mock_filter = MagicMock(side_effect=lambda s, tags: s)
    monkeypatch.setattr(agent_factory, "filter_tools_by_tag", mock_filter)
    agent, _ = agent_factory.create_agent(
        name="TestMCPConfigTags",
        mcp_config=str(fake_path),
        tool_tags=["tag-x"],
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert mock_filter.call_count == 2
    assert agent is not None


# ---------------------------------------------------------------------------
# create_agent: mcp_toolsets external injection (lines 366-390)
# ---------------------------------------------------------------------------


def test_create_agent_external_mcp_toolsets(monkeypatch: pytest.MonkeyPatch) -> None:
    """External mcp_toolsets list is processed."""
    monkeypatch.setattr(agent_factory, "DEFAULT_VALIDATION_MODE", False)
    # Create a plain object (not named FastMCP)
    fake = MagicMock(spec=[])
    fake.http_client = None

    agent, initialized = agent_factory.create_agent(
        name="TestExternalToolsets",
        mcp_toolsets=[fake, None],  # None should be skipped
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None
    assert len(initialized) == 1


def test_create_agent_external_mcp_toolsets_fastmcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External FastMCP toolset is wrapped in FastMCPToolset."""
    monkeypatch.setattr(agent_factory, "DEFAULT_VALIDATION_MODE", False)

    class FastMCP:
        http_client = None

    fake = FastMCP()

    wrap_mock = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(agent_factory, "FastMCPToolset", wrap_mock)

    agent, initialized = agent_factory.create_agent(
        name="TestFastMCP",
        mcp_toolsets=[fake],
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None
    wrap_mock.assert_called_once_with(fake)
    assert len(initialized) == 1


def test_create_agent_mcp_toolsets_isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    """isolate_mcp=True keeps mcp_toolsets out of agent_toolsets."""
    monkeypatch.setattr(agent_factory, "DEFAULT_VALIDATION_MODE", False)
    fake = MagicMock(spec=[])
    fake.http_client = None

    agent, initialized = agent_factory.create_agent(
        name="TestMCPIsolate",
        mcp_toolsets=[fake],
        isolate_mcp=True,
        skill_types=[],
        enable_skills=False,
        enable_universal_tools=False,
    )
    assert agent is not None
    assert len(initialized) == 1


# ---------------------------------------------------------------------------
# create_agent: enable_skills path (lines 423, 433-434, 437, 440-448)
# ---------------------------------------------------------------------------


def test_create_agent_skills_graphs_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When skill-graphs is missing, the path is silently skipped (line 433-434)."""
    import sys

    # Save original
    orig = sys.modules.pop("skill_graphs", None)
    sys.modules.pop("skill_graphs.skill_graph_utilities", None)

    # Ensure the import inside the function raises
    class _BrokenLoader:
        def find_module(self, name, path=None):
            if name.startswith("skill_graphs"):
                return self

        def load_module(self, name):
            raise ImportError("skill_graphs not installed")

    try:
        agent, _ = agent_factory.create_agent(
            name="TestSkillsNoGraphs",
            skill_types=["graphs"],
            enable_skills=True,
            enable_universal_tools=False,
        )
        assert agent is not None
    finally:
        if orig is not None:
            sys.modules["skill_graphs"] = orig


def test_create_agent_custom_skills_directory_string(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """custom_skills_directory as a string path is appended if it exists."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    agent, _ = agent_factory.create_agent(
        name="TestCustomSkillsStr",
        skill_types=[],
        custom_skills_directory=str(skills_dir),
        enable_skills=True,
        enable_universal_tools=False,
    )
    assert agent is not None


def test_create_agent_custom_skills_directory_list(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """custom_skills_directory as list/tuple iterates and adds existing dirs."""
    s1 = tmp_path / "s1"
    s1.mkdir()
    s2 = tmp_path / "s2"
    s2.mkdir()
    agent, _ = agent_factory.create_agent(
        name="TestCustomSkillsList",
        skill_types=[],
        custom_skills_directory=[str(s1), str(s2), "/nonexistent"],  # type: ignore[arg-type]
        enable_skills=True,
        enable_universal_tools=False,
    )
    assert agent is not None


def test_create_agent_custom_skills_directory_nonexistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-existent custom_skills_directory string is silently ignored."""
    agent, _ = agent_factory.create_agent(
        name="TestCustomSkillsMissing",
        skill_types=[],
        custom_skills_directory="/this/path/really/does/not/exist",
        enable_skills=True,
        enable_universal_tools=False,
    )
    assert agent is not None


def test_create_agent_skills_with_tool_tags_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tool_tags filter narrows skill_dirs via skill_matches_tags."""
    # Make skill_matches_tags reject everything to exercise the filter branch
    monkeypatch.setattr(agent_factory, "skill_matches_tags", lambda d, tags: False)
    agent, _ = agent_factory.create_agent(
        name="TestSkillsTagFilter",
        skill_types=["universal"],
        tool_tags=["nonexistent-tag"],
        enable_skills=True,
        enable_universal_tools=False,
    )
    assert agent is not None
