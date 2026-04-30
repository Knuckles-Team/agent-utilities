from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities.mcp.agent_manager import (
    compute_agent_metadata_score,
    compute_tool_relevance_score,
    extract_tool_metadata,
    partition_tools,
    should_sync,
)
from agent_utilities.models import MCPToolInfo


def test_compute_tool_relevance_score():
    tool = MCPToolInfo(
        name="test_tool",
        description="A very long and detailed description for the test tool that should score high.",
        tag="test",
        all_tags=["test", "extra"],
        mcp_server="test-server"
    )
    score = compute_tool_relevance_score(tool)
    assert score > 50

    tool_minimal = MCPToolInfo(name="run", description="", tag="gen", mcp_server="s")
    score_min = compute_tool_relevance_score(tool_minimal)
    assert score_min < score

def test_compute_agent_metadata_score():
    assert compute_agent_metadata_score("desc", ["skill1"]) == 15
    assert compute_agent_metadata_score("a" * 200, ["s"] * 15) == 100

@pytest.mark.asyncio
async def test_partition_tools():
    tools = [
        MCPToolInfo(name="t1", description="", tag="git", mcp_server="s1"),
        MCPToolInfo(name="t2", description="", tag="docker", mcp_server="s2"),
        MCPToolInfo(name="t3", description="", tag="git", mcp_server="s1"),
    ]
    partitions = await partition_tools(tools)
    assert "git" in partitions
    assert "docker" in partitions
    assert len(partitions["git"]) == 2
    assert len(partitions["docker"]) == 1

@pytest.mark.asyncio
async def test_should_sync_no_config():
    assert should_sync(Path("/nonexistent")) is False

@pytest.mark.asyncio
async def test_should_sync_needed():
    config_path = Path("mcp_config.json")
    with patch("pathlib.Path.exists", return_value=True), \
         patch("agent_utilities.knowledge_graph.engine.IntelligenceGraphEngine.get_active", return_value=None):
        assert should_sync(config_path) is True

@pytest.mark.asyncio
async def test_extract_tool_metadata():
    config_path = Path("mcp_config.json")

    mock_server = AsyncMock()
    mock_server.name = "test-server"

    mock_session = AsyncMock()
    mock_tool = MagicMock()
    mock_tool.name = "tool1"
    mock_tool.description = "desc1"
    mock_session.list_tools.return_value = [mock_tool]
    # Handle the context manager (async with server as session)
    mock_server.__aenter__.return_value = mock_session

    with patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value={"mcpServers": {}}), \
         patch("agent_utilities.mcp.agent_manager.load_mcp_config", return_value=[mock_server]):

        tools = await extract_tool_metadata(config_path)
        assert len(tools) == 1
        assert tools[0].name == "tool1"
