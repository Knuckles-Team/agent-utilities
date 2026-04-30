from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.protocols.a2a import (
    A2AClient,
)


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
