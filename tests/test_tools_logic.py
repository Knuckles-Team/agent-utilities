import pytest
import json
import os
from unittest.mock import MagicMock, patch
from pathlib import Path
from agent_utilities.mcp_utilities import load_mcp_config, create_mcp_parser
from agent_utilities.models import MCPToolInfo

def test_mcp_parser_defaults():
    """Test MCP argument parser default values."""
    parser = create_mcp_parser()
    args = parser.parse_args([])
    assert args.transport == "stdio"
    assert args.auth_type == "none"

def test_load_mcp_config_expansion(tmp_path):
    """Test environment variable expansion in MCP config Loading."""
    mcp_config = {
        "mcpServers": {
            "test-server": {
                "command": "echo",
                "args": ["${TEST_KEY}"]
            }
        }
    }
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(json.dumps(mcp_config))

    captured_content = {}

    def mock_load_side_effect(path):
        with open(path, "r") as f:
            captured_content["data"] = json.load(f)
        mock_server = MagicMock()
        return [mock_server]

    with patch.dict(os.environ, {"TEST_KEY": "expanded-value"}):
        with patch("pydantic_ai.mcp.load_mcp_servers", side_effect=mock_load_side_effect) as mock_load:
            servers = load_mcp_config(config_path)

            assert len(servers) == 1
            assert mock_load.called

            # Verify the temp file passed to load_mcp_servers had expanded content
            content = captured_content["data"]
            assert content["mcpServers"]["test-server"]["args"] == ["expanded-value"]
            # Verify Suppress RequestsDependencyWarning is added
            assert "PYTHONWARNINGS" in content["mcpServers"]["test-server"]["env"]

def test_mcp_tool_info_model():
    """Test the MCPToolInfo Pydantic model."""
    tool = MCPToolInfo(
        name="test_tool",
        description="A test tool",
        tag="test",
        mcp_server="test-server"
    )
    assert tool.name == "test_tool"
    assert tool.tag == "test"
