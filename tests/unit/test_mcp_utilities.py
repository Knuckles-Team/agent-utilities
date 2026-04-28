import pytest
import os
import json
import tempfile
from unittest.mock import MagicMock, patch, patch
from agent_utilities import mcp_utilities

def test_create_mcp_parser():
    parser = mcp_utilities.create_mcp_parser()
    assert parser.description == "MCP Server"
    # Check some default arguments
    args = parser.parse_args(["--transport", "sse", "--port", "9000"])
    assert args.transport == "sse"
    assert args.port == 9000

@patch("fastmcp.FastMCP")
def test_create_mcp_server_basic(mock_fastmcp):
    # Mocking parse_known_args to return default values
    with patch("argparse.ArgumentParser.parse_known_args") as mock_parse:
        mock_args = MagicMock()
        mock_args.port = 8000
        mock_args.enable_delegation = False
        mock_args.auth_type = "none"
        mock_args.help = False
        mock_parse.return_value = (mock_args, [])

        args, mcp, middlewares = mcp_utilities.create_mcp_server(name="TestServer")

        assert args == mock_args
        mock_fastmcp.assert_called_once_with("TestServer", auth=None, instructions="")
        assert len(middlewares) >= 4 # Default middlewares

def test_load_mcp_servers_from_config_missing():
    servers = mcp_utilities.load_mcp_servers_from_config("non_existent.json")
    assert servers == []

@patch("pydantic_ai.mcp.load_mcp_servers")
def test_load_mcp_servers_from_config_success(mock_load):
    config_data = {
        "mcpServers": {
            "test-server": {
                "command": "python",
                "args": ["-m", "test"],
                "env": {"FOO": "BAR"}
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(config_data, tmp)
        tmp_path = tmp.name

    try:
        mock_server = MagicMock()
        mock_load.return_value = [mock_server]

        servers = mcp_utilities.load_mcp_servers_from_config(tmp_path)

        assert len(servers) == 1
        assert servers[0].id == "test-server"
        mock_load.assert_called_once()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_mcp_auth_config_defaults():
    # Should use env vars
    assert isinstance(mcp_utilities.mcp_auth_config, dict)
    assert "enable_delegation" in mcp_utilities.mcp_auth_config

@patch("requests.get")
def test_create_mcp_server_delegation_error(mock_get):
    # Test that delegation requires oidc-proxy
    with patch("argparse.ArgumentParser.parse_known_args") as mock_parse:
        mock_args = MagicMock()
        mock_args.port = 8000
        mock_args.help = False
        mock_args.enable_delegation = True
        mock_args.auth_type = "jwt" # Wrong type
        mock_args.audience = "aud"
        mock_args.delegated_scopes = "api"
        mock_args.oidc_config_url = None
        mock_args.oidc_client_id = None
        mock_args.oidc_client_secret = None
        mock_parse.return_value = (mock_args, [])

        with pytest.raises(SystemExit) as excinfo:
            mcp_utilities.create_mcp_server()
        assert excinfo.value.code == 1

def test_create_mcp_server_invalid_port():
    with patch("argparse.ArgumentParser.parse_known_args") as mock_parse:
        mock_args = MagicMock()
        mock_args.help = False
        mock_args.port = 70000 # Invalid
        mock_parse.return_value = (mock_args, [])

        with pytest.raises(SystemExit) as excinfo:
            mcp_utilities.create_mcp_server()
        assert excinfo.value.code == 1
