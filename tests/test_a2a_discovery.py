import pytest
from unittest.mock import patch, MagicMock
from agent_utilities.a2a import (
    register_a2a_peer,
    delete_a2a_peer,
    A2AClient,
)
from agent_utilities.discovery import discover_agents


@pytest.fixture
def mock_a2a_workspace(tmp_path):
    with patch("agent_utilities.a2a.get_workspace_path") as mock_ws_path:
        a2a_file = tmp_path / "A2A_AGENTS.md"
        mock_ws_path.return_value = a2a_file
        yield a2a_file


def test_register_delete_peer(mock_a2a_workspace):
    # Initial empty
    with patch("agent_utilities.a2a.load_workspace_file", return_value=""):
        res = register_a2a_peer("TestPeer", "http://localhost:9001", "Desc")
        assert "Registered" in res

    # Reload and check
    content = mock_a2a_workspace.read_text()
    assert "TestPeer" in content
    assert "http://localhost:9001" in content

    # Delete
    with patch("agent_utilities.a2a.load_workspace_file", return_value=content):
        res = delete_a2a_peer("TestPeer")
        assert "Removed" in res

    new_content = mock_a2a_workspace.read_text()
    assert "TestPeer" not in new_content


@pytest.mark.asyncio
async def test_a2a_client_fetch_card():
    client = A2AClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"name": "RemoteAgent", "description": "RemoteDesc"}

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        card = await client.fetch_card("http://remote:8000")
        assert card["name"] == "RemoteAgent"


def test_discover_agents_local_mcp():
    mcp_content = """## Agent Mapping Table
| Name | Description | System Prompt | Tools | Tag | Source MCP |
|------|-------------|---------------|-------|-----|------------|
| specialist | A test specialist | Be helpful | tool1 | expert | mcp1 |
"""
    with patch("agent_utilities.discovery.load_workspace_file") as mock_load:
        # First call for NODE_AGENTS.md, second for A2A_AGENTS.md
        mock_load.side_effect = [mcp_content, ""]

        agents = discover_agents()
        assert "expert" in agents
        assert agents["expert"]["package"] == "specialist"
        assert agents["expert"]["type"] == "local_mcp"


def test_discover_agents_remote_a2a():
    a2a_content = """| Name | Endpoint URL | Description | Capabilities | Auth | Notes |
|------|--------------|-------------|--------------|------|-------|
| RemotePeer | http://remote:8000 | Remote agent | cap1 | none | - |
"""
    with patch("agent_utilities.discovery.load_workspace_file") as mock_load:
        mock_load.side_effect = [""]
        with patch("agent_utilities.a2a.load_workspace_file", return_value=a2a_content):
            with patch("agent_utilities.a2a.A2AClient.fetch_card_sync", return_value=None):
                agents = discover_agents()
                assert "remotepeer" in agents
                assert agents["remotepeer"]["url"] == "http://remote:8000"
                assert agents["remotepeer"]["type"] == "remote_a2a"
