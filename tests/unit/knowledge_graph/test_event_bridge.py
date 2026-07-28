from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.core.event_backend import get_event_backend


@pytest.mark.asyncio
async def test_graph_compute_event_bridge():
    """Test that GraphComputeEngine forwards local EventBus events to the Rust client."""
    import asyncio

    import agent_utilities.knowledge_graph.core.event_backend as eb_module
    from agent_utilities.knowledge_graph.core.event_backend import MemoryEventBackend
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    eb_module._GLOBAL_EVENT_BACKEND = None
    with patch.dict(
        "os.environ", {"KAFKA_BOOTSTRAP_SERVERS": "", "EVENT_BACKEND": "memory"}
    ):
        engine = GraphComputeEngine.__new__(GraphComputeEngine)
        engine._process_root = engine
        engine._client = MagicMock()
        engine._event_bridge_stop = None
        engine._event_bridge_thread = None
        engine._event_bridge_loop = None
        engine._event_bridge_async_stop = None
        engine._start_event_bridge()

        eb = get_event_backend()
        assert isinstance(eb, MemoryEventBackend)

        for _ in range(20):
            if eb._subscriptions.get("kg.mutations"):
                break
            await asyncio.sleep(0.05)

        assert "kg.mutations" in eb._subscriptions
        subscribers = eb._subscriptions["kg.mutations"]
        assert len(subscribers) > 0
        callback = subscribers[-1][1]

        test_payload = {
            "event_type": "TRIPLE_INSERT",
            "query": "INSERT DATA { <A> <B> <C> }",
            "source": "jena_fuseki_backend",
        }

        await callback("kg.mutations", test_payload)

        engine._client.apply_mutation.assert_called_once_with(
            "TRIPLE_INSERT", "INSERT DATA { <A> <B> <C> }"
        )
        engine._stop_event_bridge()
        assert "kg.mutations" not in eb._subscriptions
