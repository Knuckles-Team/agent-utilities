"""CONCEPT:AU-ECO.messaging.native-backend-abstraction"""

from unittest.mock import AsyncMock, patch

import pytest

from agent_utilities.protocols.a2a import (
    A2AClient,
)


@pytest.mark.asyncio
async def test_a2a_client_fetch_card():
    """``fetch_card`` fetches through the SSRF-safe ``safe_get_json_async``
    wrapper (``protocols/source_connectors/http_safety.py``), not a raw
    ``httpx.AsyncClient.get`` — mocking the raw client never intercepted the
    call, so the request always fell into ``fetch_card``'s ``except Exception``
    and silently returned ``None``. Mirrors the already-correct pattern in
    ``tests/unit/protocols/test_a2a.py::test_a2a_client_fetch_card_async_success``."""
    client = A2AClient()

    with patch(
        "agent_utilities.protocols.a2a.safe_get_json_async",
        new=AsyncMock(
            return_value={"name": "RemoteAgent", "description": "RemoteDesc"}
        ),
    ):
        card = await client.fetch_card("http://remote:8000")
        assert card is not None
        assert card["name"] == "RemoteAgent"
