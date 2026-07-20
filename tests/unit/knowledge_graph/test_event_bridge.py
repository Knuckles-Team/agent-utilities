from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.core.event_backend import get_event_backend


@pytest.mark.asyncio
async def test_graph_compute_event_bridge():
    """Test that GraphComputeEngine forwards local EventBus events to the Rust client."""

    # Mock the entire epistemic_graph package since the compiled extension might not match
    # the test runner's Python version (e.g. 3.11 vs 3.14).
    mock_client_module = MagicMock()
    mock_sync_client = MagicMock()
    mock_client_module.SyncEpistemicGraphClient = mock_sync_client
    with patch.dict(
        "sys.modules",
        {"epistemic_graph": MagicMock(), "epistemic_graph.client": mock_client_module},
    ):
        mock_client_instance = mock_sync_client.connect.return_value
        import agent_utilities.knowledge_graph.core.event_backend as eb_module
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )

        eb_module._GLOBAL_EVENT_BACKEND = None
        with patch.dict(
            "os.environ", {"KAFKA_BOOTSTRAP_SERVERS": "", "EVENT_BACKEND": "memory"}
        ):
            GraphComputeEngine(graph_name="test_graph")

            # The bridge starts in a background thread. We can simulate the EventBus emission
            # and then check if the client's apply_mutation was called.
            # However, since it's a separate thread, it's easier to just test the bridge logic directly.

            from agent_utilities.knowledge_graph.core.event_backend import (
                MemoryEventBackend,
            )

            eb = get_event_backend()
            assert isinstance(eb, MemoryEventBackend)

            import asyncio

            await asyncio.sleep(0.1)  # Wait for the background thread to subscribe

            # Find the subscriber that was added for "kg.mutations"
            assert "kg.mutations" in eb._subscriptions
            subscribers = eb._subscriptions["kg.mutations"]
            assert len(subscribers) > 0

            # Get the callback function and call it manually to test the inner logic
            callback = subscribers[0][1]

            test_payload = {
                "event_type": "TRIPLE_INSERT",
                "query": "INSERT DATA { <A> <B> <C> }",
                "source": "jena_fuseki_backend",
            }

            await callback("kg.mutations", test_payload)

            mock_client_instance.apply_mutation.assert_called_once_with(
                "TRIPLE_INSERT", "INSERT DATA { <A> <B> <C> }"
            )
