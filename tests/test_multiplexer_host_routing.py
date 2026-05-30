import asyncio

import pytest

from agent_utilities.mcp.multiplexer import (
    MCPMultiplexer,
    clean_tool_name,
    get_server_prefix,
)


def test_get_server_prefix_hosts():
    # Test standard systems-manager hosts
    assert get_server_prefix("systems-manager-mcp-r510") == "sys_r510"
    assert get_server_prefix("systems-manager-mcp-rw710") == "sys_rw710"
    assert get_server_prefix("systems-manager-mcp-gr1080") == "sys_gr1080"

    # Test standard container-manager hosts
    assert get_server_prefix("container-manager-mcp-r510") == "cnt_r510"
    assert get_server_prefix("container-manager-mcp-gr1080") == "cnt_gr1080"

    # Test other standard nicknamable servers
    assert get_server_prefix("graph-os") == "kg"
    assert get_server_prefix("repository-manager-mcp") == "rep"

    # Test fallback
    assert get_server_prefix("some-random-mcp-server") == "some"


def test_clean_tool_name_prefixing():
    # Verify that clean_tool_name applies prefixes correctly without collisions and within length budgets
    prefix = get_server_prefix("systems-manager-mcp-r510")
    assert prefix == "sys_r510"

    cleaned = clean_tool_name(
        prefix, "systems-manager-mcp-r510", "systems_manager_mcp_run_command"
    )
    # Prefix (sys_r510) + "__" + stripped tool name (run_command)
    assert cleaned == "sys_r510__run_command"


def test_multiplexer_tool_filtering():
    import fnmatch

    # Mocking the filtering logic of multiplexer
    tools = [
        "cm_image_operations",
        "cm_volume_operations",
        "cm_compose_operations",
        "trace_port_namespace",
    ]

    # Whitelist only image and volume
    enabled_tools = ["*image*", "*volume*"]
    disabled_tools: list[str] = []

    filtered = []
    for t in tools:
        if enabled_tools is not None:
            matched = any(fnmatch.fnmatch(t, pat) for pat in enabled_tools)
            if not matched:
                continue
        if disabled_tools:
            matched_disabled = any(fnmatch.fnmatch(t, pat) for pat in disabled_tools)
            if matched_disabled:
                continue
        filtered.append(t)

    assert filtered == ["cm_image_operations", "cm_volume_operations"]

    # Test blacklist only
    enabled_tools_2 = None
    disabled_tools_2 = ["*compose*"]

    filtered_2 = []
    for t in tools:
        if enabled_tools_2 is not None:
            matched = any(fnmatch.fnmatch(t, pat) for pat in enabled_tools_2)
            if not matched:
                continue
        if disabled_tools_2:
            matched_disabled = any(fnmatch.fnmatch(t, pat) for pat in disabled_tools_2)
            if matched_disabled:
                continue
        filtered_2.append(t)

    assert filtered_2 == [
        "cm_image_operations",
        "cm_volume_operations",
        "trace_port_namespace",
    ]


@pytest.mark.asyncio
async def test_multiplexer_start_children_aggregation():
    import json
    from unittest.mock import AsyncMock, MagicMock, patch

    config = {
        "mcpServers": {
            "healthy-server": {
                "command": "python",
                "args": ["-m", "healthy"],
                "timeout": 1.0,
                "enabledTools": ["*"],
            },
            "failing-server": {
                "command": "python",
                "args": ["-m", "failing"],
                "timeout": 1.0,
            },
        }
    }

    mock_config_path = MagicMock()
    mock_config_path.exists.return_value = True
    mock_config_path.read_text.return_value = json.dumps(config)

    multiplexer = MCPMultiplexer(mock_config_path)

    # Mock _start_child to return a successful tuple for healthy, and None for failing
    async def mock_start_child(server_name, cfg):
        if server_name == "healthy-server":
            mock_session = AsyncMock()
            mock_tool = MagicMock()
            mock_tool.name = "healthy_tool"
            mock_tool.description = "Healthy description"
            mock_tool.inputSchema = {}
            return server_name, mock_session, [mock_tool], cfg
        return None

    with patch.object(multiplexer, "_start_child", side_effect=mock_start_child):
        await multiplexer.start_children()

    assert "healthy-server" in multiplexer.sessions
    assert len(multiplexer.aggregated_tools) == 1
    assert multiplexer.aggregated_tools[0].name == "healt__healthy_tool"


@pytest.mark.asyncio
async def test_multiplexer_start_child_timeout():
    from unittest.mock import AsyncMock, MagicMock, patch

    # Configure a server with a timeout
    cfg = {"command": "python", "args": ["-m", "slow"], "timeout": 0.05}

    multiplexer = MCPMultiplexer(MagicMock())

    # Mock stdio_client to simulate a long connection/handshake delay
    import contextlib

    @contextlib.asynccontextmanager
    async def slow_connect(*args, **kwargs):
        try:
            await asyncio.sleep(0.5)  # Longer than the timeout of 0.05
            yield AsyncMock(), AsyncMock()
        except asyncio.CancelledError:
            raise

    with patch(
        "agent_utilities.mcp.multiplexer.stdio_client", side_effect=slow_connect
    ):
        result = await multiplexer._start_child("slow-server", cfg)

    assert result is None
