import pytest
from unittest.mock import patch, MagicMock
from agent_utilities.a2a import (
    register_a2a_peer,
    delete_a2a_peer,
    A2AClient,
)
from agent_utilities.discovery import discover_agents


@pytest.mark.asyncio
async def test_a2a_client_fetch_card():
    client = A2AClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"name": "RemoteAgent", "description": "RemoteDesc"}

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        card = await client.fetch_card("http://remote:8000")
        assert card is not None
        assert card["name"] == "RemoteAgent"
