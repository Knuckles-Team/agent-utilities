import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.backends.sparql.jena_fuseki_backend import (
    JenaFusekiBackend,
)
from agent_utilities.knowledge_graph.core.event_backend import (
    TOPIC_MUTATIONS,
    EventBackend,
)


@pytest.mark.asyncio
async def test_jena_fuseki_publishes_event_on_update():
    # Mock EventBackend
    event_backend_mock = MagicMock(spec=EventBackend)
    event_backend_mock.publish = AsyncMock()

    # Mock HTTPX response
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    backend = JenaFusekiBackend(
        jena_fuseki_url="http://localhost:3030",
        dataset="test",
        event_backend=event_backend_mock,
    )

    with patch.object(backend._client, "post", return_value=mock_resp):
        # Execute update
        backend.execute_sparql_update("INSERT DATA { <A> <B> <C> }")

        # Yield to event loop to let asyncio.create_task run
        await asyncio.sleep(0.01)

        # Verify publish was called
        event_backend_mock.publish.assert_called_once()
        args, _ = event_backend_mock.publish.call_args
        assert args[0] == TOPIC_MUTATIONS
        assert args[1]["event_type"] == "TRIPLE_INSERT"
        assert args[1]["query"] == "INSERT DATA { <A> <B> <C> }"


@pytest.mark.asyncio
async def test_jena_fuseki_publishes_delete_event():
    event_backend_mock = MagicMock(spec=EventBackend)
    event_backend_mock.publish = AsyncMock()

    mock_resp = MagicMock()
    mock_resp.status_code = 204

    backend = JenaFusekiBackend(
        jena_fuseki_url="http://localhost:3030",
        dataset="test",
        event_backend=event_backend_mock,
    )

    with patch.object(backend._client, "post", return_value=mock_resp):
        backend.execute_sparql_update("DELETE DATA { <A> <B> <C> }")
        await asyncio.sleep(0.01)

        event_backend_mock.publish.assert_called_once()
        args, _ = event_backend_mock.publish.call_args
        assert args[0] == TOPIC_MUTATIONS
        assert args[1]["event_type"] == "TRIPLE_DELETE"
