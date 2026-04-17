import pytest
from unittest.mock import patch, MagicMock
from agent_utilities.a2a import (
    register_a2a_peer,
    delete_a2a_peer,
    A2AClient,
)
from agent_utilities.discovery import discover_agents


def test_register_delete_peer():
    """Verify that register and delete peer NO LONGER write to the filesystem."""

    # Register
    with patch("agent_utilities.a2a.load_workspace_file", return_value=""):
        res = register_a2a_peer("TestPeer", "http://localhost:9001", "Desc")
        assert "Registered" in res
        assert "Unified Registry" in res

    # Delete
    res = delete_a2a_peer("TestPeer")
    assert "Removed" in res
    assert "unified registry" in res


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
| Name | Description | System Prompt | Tag | Skills | Tools | Skill Count | Tool Count | Avg Score |
|------|-------------|---------------|-----|--------|-------|-------------|------------|-----------|
| specialist | A test specialist | Be helpful | expert | skill1 | - | 1 | 0 | 100 |
"""
    with patch("agent_utilities.discovery.load_workspace_file") as mock_load:
        # NODE_AGENTS.md has content, A2A_AGENTS.md is missing (raises or returns empty)
        mock_load.side_effect = [mcp_content, ""]

        agents = discover_agents()
        assert "specialist" in agents
        assert agents["specialist"]["package"] == "specialist"
        assert agents["specialist"]["type"] == "local_mcp"


def test_discover_agents_remote_a2a():
    """Verify discovery still works if A2A_AGENTS.md exists (legacy support)."""
    a2a_content = """| Name | Endpoint URL | Description | Capabilities | Auth | Notes |
|------|--------------|-------------|--------------|------|-------|
| RemotePeer | http://remote:8000 | Remote agent | cap1 | none | - |
"""
    with patch("agent_utilities.discovery.load_workspace_file") as mock_load:
        mock_load.side_effect = ["", a2a_content]  # NODE_AGENTS.md empty, A2A_AGENTS.md has content
        with patch("agent_utilities.a2a.A2AClient.fetch_card_sync", return_value=None):
            agents = discover_agents()
            assert "remotepeer" in agents
            assert agents["remotepeer"]["url"] == "http://remote:8000"
            assert agents["remotepeer"]["type"] == "remote_a2a"
