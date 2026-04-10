import json
import pytest
from unittest.mock import patch
from agent_utilities.mcp_agent_manager import sync_mcp_agents, extract_tool_metadata


@pytest.mark.skip(reason="ImportError for StdioServer in pydantic-ai")
@pytest.mark.asyncio
async def test_extract_tool_metadata_real(tmp_path):
    """Test extracting tool metadata from a real (stdio) MCP server."""
    # Create a tiny MCP server script
    mcp_script = tmp_path / "tiny_mcp.py"
    mcp_script.write_text("""
from fastmcp import FastMCP
mcp = FastMCP("TinyMCP")
@mcp.tool()
def add(a: int, b: int) -> int:
    "Add two numbers"
    return a + b
if __name__ == "__main__":
    mcp.run()
""")

    # Create mcp_config.json
    config_path = tmp_path / "mcp_config.json"
    config_data = {
        "mcpServers": {"tiny-mcp": {"command": "python", "args": [str(mcp_script)]}}
    }
    config_path.write_text(json.dumps(config_data))

    # We need to mock get_workspace_path to find this config
    with (
        patch(
            "agent_utilities.mcp_agent_manager.get_workspace_path",
            return_value=config_path,
        ),
        patch("agent_utilities.mcp_agent_manager.load_mcp_config") as mock_load,
    ):
        # We'll use the real load_mcp_servers if possible,
        # but to be safe and fast, let's mock the server session but use the real logic
        from pydantic_ai.mcp import StdioServer

        server = StdioServer(command="python", args=[str(mcp_script)])
        mock_load.return_value = [server]

        tools = await extract_tool_metadata(config_path)
        assert len(tools) > 0
        assert tools[0].name == "add"
        assert tools[0].mcp_server == "TinyMCP"


@pytest.mark.asyncio
async def test_sync_mcp_agents_flow(tmp_path):
    """Test the full sync flow."""
    config_path = tmp_path / "mcp_config.json"
    config_data = {"mcpServers": {}}
    config_path.write_text(json.dumps(config_data))

    tmp_path / "NODE_AGENTS.md"

    from agent_utilities.models import MCPToolInfo

    mock_tool = MCPToolInfo(
        name="test_tool",
        description="A test tool",
        tag="test",
        mcp_server="test_server",
    )

    with (
        patch(
            "agent_utilities.mcp_agent_manager.extract_tool_metadata",
            return_value=[mock_tool],
        ),
        patch("agent_utilities.mcp_agent_manager.get_workspace_path") as mock_ws_path,
        patch("agent_utilities.mcp_agent_manager.load_workspace_file", return_value=""),
        patch("agent_utilities.mcp_agent_manager.write_workspace_file") as mock_write,
    ):
        mock_ws_path.side_effect = lambda f: tmp_path / f

        await sync_mcp_agents(config_path=config_path)

        # Verify write was called with NODE_AGENTS.md content
        mock_write.assert_called_once()
        args, kwargs = mock_write.call_args
        assert "NODE_AGENTS.md" in args[0]
        assert "test_tool" in args[1]
        assert "Test Server Test Specialist" in args[1]
