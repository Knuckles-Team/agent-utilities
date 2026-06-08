from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities.mcp.server_factory import create_mcp_server
from agent_utilities.tools.dynamic_tool_orchestrator import DynamicToolOrchestrator


class DummyComponent:
    def __init__(self, name, tags=None):
        self.name = name
        self.tags = set(tags) if tags else set()


# ---------------------------------------------------------------------------
# DynamicToolOrchestrator Tests
# ---------------------------------------------------------------------------


def test_assign_tools_for_task_no_backend() -> None:
    """If backend is None, returns empty list."""
    mock_engine = MagicMock()
    mock_engine.backend = None
    orchestrator = DynamicToolOrchestrator(mock_engine)
    assert orchestrator.assign_tools_for_task("test task", "developer") == []


def test_assign_tools_for_task_success() -> None:
    """If backend is configured, queries KG and returns matching tools."""
    mock_engine = MagicMock()
    mock_backend = MagicMock()
    mock_engine.backend = mock_backend
    mock_backend.execute.return_value = [
        {
            "tool_name": "docker_ps",
            "tool_desc": "List containers",
            "schema": '{"type": "object"}',
        },
        {
            "tool_name": "docker_run",
            "tool_desc": "Run container",
            "schema": '{"type": "object"}',
        },
    ]

    orchestrator = DynamicToolOrchestrator(mock_engine)
    tools = orchestrator.assign_tools_for_task("docker list", "developer")

    assert len(tools) == 2
    assert tools[0]["name"] == "docker_ps"
    assert tools[0]["description"] == "List containers"
    assert tools[0]["schema"] == '{"type": "object"}'
    assert mock_backend.execute.called


def test_assign_tools_for_task_exception() -> None:
    """If backend raises exception, returns empty list without crashing."""
    mock_engine = MagicMock()
    mock_backend = MagicMock()
    mock_engine.backend = mock_backend
    mock_backend.execute.side_effect = Exception("DB error")

    orchestrator = DynamicToolOrchestrator(mock_engine)
    tools = orchestrator.assign_tools_for_task("docker list", "developer")
    assert tools == []


def test_resolve_mcp_tools_no_backend() -> None:
    """If backend is None, resolve_mcp_tools returns empty list."""
    mock_engine = MagicMock()
    mock_engine.backend = None
    orchestrator = DynamicToolOrchestrator(mock_engine)
    assert orchestrator.resolve_mcp_tools("dns") == []


def test_resolve_mcp_tools_success() -> None:
    """resolve_mcp_tools runs multi-vector matching successfully."""
    mock_engine = MagicMock()
    mock_backend = MagicMock()
    mock_engine.backend = mock_backend
    mock_backend.execute.side_effect = [
        [{"name": "add_rewrite"}, {"name": "delete_rewrite"}],
        [],
    ]

    orchestrator = DynamicToolOrchestrator(mock_engine)
    tools = orchestrator.resolve_mcp_tools("rewrite", "technitium")

    assert tools == ["add_rewrite", "delete_rewrite"]
    assert mock_backend.execute.call_count == 2


def test_resolve_mcp_tools_fallback() -> None:
    """If multi-vector matching yields 0 results, fall back to all tools of the server."""
    mock_engine = MagicMock()
    mock_backend = MagicMock()
    mock_engine.backend = mock_backend
    # First execute (multi-vector) returns [], second execute (fallback) returns all tools, third (ts sweep) returns []
    mock_backend.execute.side_effect = [
        [],
        [{"name": "all_tool_1"}, {"name": "all_tool_2"}],
        [],
    ]

    orchestrator = DynamicToolOrchestrator(mock_engine)
    tools = orchestrator.resolve_mcp_tools("nonexistent", "technitium")

    assert tools == ["all_tool_1", "all_tool_2"]
    assert mock_backend.execute.call_count == 3


@pytest.mark.asyncio
async def test_refresh_cached_tools_no_backend() -> None:
    """refresh_cached_tools returns False if backend is None."""
    mock_engine = MagicMock()
    mock_engine.backend = None
    orchestrator = DynamicToolOrchestrator(mock_engine)
    result = await orchestrator.refresh_cached_tools("test-server")
    assert result is False


@pytest.mark.asyncio
async def test_refresh_cached_tools_success() -> None:
    """refresh_cached_tools queries server config, calls discover_mcp_tools and updates DB."""
    mock_engine = MagicMock()
    mock_backend = MagicMock()
    mock_engine.backend = mock_backend

    mock_backend.execute.side_effect = [
        [
            {
                "command": "python",
                "args": '["-m", "mcp"]',
                "env": '{"KEY": "val"}',
                "source_config": "config.json",
            }
        ],
        [],  # update server statement
    ]

    mock_engine.discover_mcp_tools = AsyncMock(return_value=["tool_1", "tool_2"])
    mock_engine.ingest_mcp_server = MagicMock()

    orchestrator = DynamicToolOrchestrator(mock_engine)
    result = await orchestrator.refresh_cached_tools("test-server")

    assert result is True
    mock_engine.discover_mcp_tools.assert_called_once()
    mock_engine.ingest_mcp_server.assert_called_once()
    assert mock_backend.execute.call_count == 2


# ---------------------------------------------------------------------------
# DynamicVisibilityTransform / create_mcp_server Tests
# ---------------------------------------------------------------------------


def test_dynamic_visibility_transform_env_filtering() -> None:
    """Test DynamicVisibilityTransform filters components based on environment variables."""
    with patch.dict(
        os.environ,
        {
            "MCP_ENABLED_TAGS": "docker,git",
            "MCP_DISABLED_TAGS": "unstable",
            "MCP_ENABLED_TOOLS": "docker_ps,git_commit",
            "MCP_DISABLED_TOOLS": "git_push",
        },
    ):
        # Mock create_mcp_server args and transform execution
        args, mcp, _ = create_mcp_server("Test Server", command_args=[])
        transform = mcp._transforms[0]

        components = [
            DummyComponent("docker_ps", ["docker"]),
            DummyComponent("git_commit", ["git"]),
            DummyComponent("git_push", ["git"]),
            DummyComponent("unstable_tool", ["docker", "unstable"]),
            DummyComponent("other_tool", ["other"]),
        ]

        filtered = transform._filter_components(components)
        filtered_names = [c.name for c in filtered]

        assert "docker_ps" in filtered_names
        assert "git_commit" in filtered_names
        assert "git_push" not in filtered_names
        assert "unstable_tool" not in filtered_names
        assert "other_tool" not in filtered_names


def test_dynamic_visibility_transform_cli_override() -> None:
    """Test DynamicVisibilityTransform filters components based on CLI arguments."""
    args, mcp, _ = create_mcp_server(
        "Test Server",
        command_args=[
            "--tools",
            "docker_ps,git_commit",
            "--disabled-tools",
            "git_push",
        ],
    )
    transform = mcp._transforms[0]

    components = [
        DummyComponent("docker_ps"),
        DummyComponent("git_commit"),
        DummyComponent("git_push"),
        DummyComponent("other_tool"),
    ]

    filtered = transform._filter_components(components)
    filtered_names = [c.name for c in filtered]

    assert "docker_ps" in filtered_names
    assert "git_commit" in filtered_names
    assert "git_push" not in filtered_names
    assert "other_tool" not in filtered_names


@patch("fastmcp.server.dependencies.get_http_request")
def test_dynamic_visibility_transform_http_request_query_params(
    mock_get_request,
) -> None:
    """Test DynamicVisibilityTransform parses and filters based on HTTP request query parameters."""
    mock_request = MagicMock()
    mock_q_params = MagicMock()
    mock_q_params.get.side_effect = lambda k: {
        "tools": "docker_ps",
        "disabled_tools": "git_push",
        "tags": "docker",
        "disabled_tags": "unstable",
    }.get(k)
    # Simulate not having getlist method to test safety fallback
    del mock_q_params.getlist

    mock_request.query_params = mock_q_params
    mock_request.headers = {}
    mock_get_request.return_value = mock_request

    args, mcp, _ = create_mcp_server("Test Server", command_args=[])
    transform = mcp._transforms[0]

    components = [
        DummyComponent("docker_ps", ["docker"]),
        DummyComponent("docker_run", ["docker"]),
        DummyComponent("git_push", ["git"]),
        DummyComponent("unstable_tool", ["docker", "unstable"]),
    ]

    filtered = transform._filter_components(components)
    filtered_names = [c.name for c in filtered]

    assert "docker_ps" in filtered_names
    assert "docker_run" not in filtered_names
    assert "git_push" not in filtered_names
    assert "unstable_tool" not in filtered_names


@patch("fastmcp.server.dependencies.get_http_request")
def test_dynamic_visibility_transform_http_request_headers(mock_get_request) -> None:
    """Test DynamicVisibilityTransform parses and filters based on HTTP request headers."""
    mock_request = MagicMock()
    mock_request.query_params = MagicMock()
    mock_request.query_params.get.return_value = None
    del mock_request.query_params.getlist

    mock_headers = MagicMock()
    mock_headers.get.side_effect = lambda k: {
        "x-mcp-enabled-tools": "docker_ps",
        "x-mcp-disabled-tools": "git_push",
        "x-mcp-enabled-tags": "docker",
        "x-mcp-disabled-tags": "unstable",
    }.get(k)
    del mock_headers.getlist

    mock_request.headers = mock_headers
    mock_get_request.return_value = mock_request

    args, mcp, _ = create_mcp_server("Test Server", command_args=[])
    transform = mcp._transforms[0]

    components = [
        DummyComponent("docker_ps", ["docker"]),
        DummyComponent("docker_run", ["docker"]),
        DummyComponent("git_push", ["git"]),
        DummyComponent("unstable_tool", ["docker", "unstable"]),
    ]

    filtered = transform._filter_components(components)
    filtered_names = [c.name for c in filtered]

    assert "docker_ps" in filtered_names
    assert "docker_run" not in filtered_names
    assert "git_push" not in filtered_names
    assert "unstable_tool" not in filtered_names


@patch("fastmcp.server.dependencies.get_http_request")
@patch("agent_utilities.knowledge_graph.core.engine.IntelligenceGraphEngine.get_active")
def test_dynamic_visibility_transform_kg_filtering(
    mock_get_active, mock_get_request
) -> None:
    """Test DynamicVisibilityTransform resolves matching tools from the KG using active query filter."""
    mock_request = MagicMock()
    mock_q_params = MagicMock()
    mock_q_params.get.side_effect = lambda k: {
        "q": "dns",
    }.get(k)
    del mock_q_params.getlist
    mock_request.query_params = mock_q_params
    mock_request.headers = {}
    mock_get_request.return_value = mock_request

    mock_engine = MagicMock()
    mock_backend = MagicMock()
    mock_engine.backend = mock_backend
    mock_get_active.return_value = mock_engine

    # KG returns specific matched tools
    mock_backend.execute.return_value = [
        {"name": "add_rewrite"},
        {"name": "get_metrics"},
    ]

    args, mcp, _ = create_mcp_server("Test Server", command_args=[])
    transform = mcp._transforms[0]

    components = [
        DummyComponent("add_rewrite"),
        DummyComponent("get_metrics"),
        DummyComponent("other_tool"),
    ]

    filtered = transform._filter_components(components)
    filtered_names = [c.name for c in filtered]

    assert "add_rewrite" in filtered_names
    assert "get_metrics" in filtered_names
    assert "other_tool" not in filtered_names
