"""Tests for MCP tool loading, parsing, and configuration.

CONCEPT:AU-ECO.messaging.native-backend-abstraction — Agent Tool System
"""

import json
import os
from unittest.mock import MagicMock, patch

from agent_utilities.core.config import load_mcp_servers_from_config
from agent_utilities.mcp.server_factory import create_mcp_parser
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
        "mcpServers": {"test-server": {"command": "echo", "args": ["${TEST_KEY}"]}}
    }
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(json.dumps(mcp_config))

    captured_content = {}

    def mock_load_side_effect(path):
        with open(path) as f:
            captured_content["data"] = json.load(f)
        mock_server = MagicMock()
        return [mock_server]

    with patch.dict(os.environ, {"TEST_KEY": "expanded-value"}):
        with patch(
            "pydantic_ai.mcp.load_mcp_toolsets", side_effect=mock_load_side_effect
        ) as mock_load:
            servers = load_mcp_servers_from_config(config_path)

            assert len(servers) == 1
            assert mock_load.called

            # Verify the temp file passed to load_mcp_toolsets had expanded content
            content = captured_content["data"]
            assert content["mcpServers"]["test-server"]["args"] == ["expanded-value"]
            # Deliberate contract update: child MCP subprocesses now get a
            # ``PYTHONWARNINGS`` env var that specifically suppresses the noisy
            # urllib3/chardet dependency warning (see
            # ``load_mcp_servers_from_config``'s "Suppress RequestsDependencyWarning
            # in subprocesses" step) — narrowly scoped, not a blanket silence.
            assert (
                content["mcpServers"]["test-server"]["env"]["PYTHONWARNINGS"]
                == "ignore:urllib3 (2.3.0) or chardet"
            )


def test_mcp_tool_info_model():
    """Test the MCPToolInfo Pydantic model."""
    tool = MCPToolInfo(
        name="test_tool",
        description="A test tool",
        tag="test",
        mcp_server="test-server",
    )
    assert tool.name == "test_tool"
    assert tool.tag == "test"
