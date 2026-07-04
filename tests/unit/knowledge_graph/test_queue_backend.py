"""CONCEPT:AU-ECO.bus.pluggable-event-queue Pluggable event queue backend unit tests."""

import asyncio
from unittest.mock import patch

import pytest

from agent_utilities.knowledge_graph.core.kafka_queue_backend import KafkaQueueBackend
from agent_utilities.knowledge_graph.core.queue_backend import (
    MemoryQueueBackend,
    QueueBackend,
)


@pytest.mark.anyio
async def test_memory_queue_backend_pub_sub():
    """Verify pub/sub operations on the default local MemoryQueueBackend."""
    backend = QueueBackend.create("memory")
    assert isinstance(backend, MemoryQueueBackend)

    await backend.connect()

    received_events = []

    async def handler(topic, data):
        received_events.append((topic, data))

    # Subscribe to test topics
    await backend.subscribe("agent.lifecycle", handler)
    await backend.subscribe("agent.*.metrics", handler)

    # Publish events
    await backend.publish("agent.lifecycle", {"status": "started"})
    await backend.publish("agent.123.metrics", {"cpu": 12})
    await backend.publish("unrelated.topic", {"ignored": True})

    # Allow local async loop to propagate
    await asyncio.sleep(0.1)

    assert len(received_events) == 2
    assert received_events[0] == ("agent.lifecycle", {"status": "started"})
    assert received_events[1] == ("agent.123.metrics", {"cpu": 12})

    await backend.disconnect()


@pytest.mark.anyio
async def test_memory_queue_operations():
    """Verify standard queue operations on MemoryQueueBackend."""
    backend = QueueBackend.create("memory")

    # Test task put and get
    backend.put({"task": "test_1"})
    backend.put({"task": "test_2"})
    assert backend.get_queue_size() == 2

    item = backend.get()
    assert item is not None
    item_id, payload = item
    assert payload["task"] == "test_1"

    # Ack first item
    backend.ack(item_id)
    assert backend.get_queue_size() == 1

    # Test staged graph
    backend.put_staged_graph(
        "job_1", [{"id": "n1"}], [{"source": "n1", "target": "n2"}]
    )
    staged = backend.get_staged_graph()
    assert staged is not None
    staged_id, job_id, graph_data = staged
    assert job_id == "job_1"
    assert len(graph_data["nodes"]) == 1

    backend.ack_staged_graph(staged_id)
    assert backend.get_staged_graph() is None


@pytest.mark.anyio
async def test_queue_backend_factory():
    """Verify backend factory creation and configuration options."""
    backend_memory = QueueBackend.create("memory")
    assert isinstance(backend_memory, MemoryQueueBackend)

    backend_kafka = QueueBackend.create(
        "kafka", fallback_db_path="test_kafka.db", bootstrap_servers="localhost:9092"
    )
    assert isinstance(backend_kafka, KafkaQueueBackend)


@pytest.mark.anyio
async def test_kafka_backend_mocked():
    """Verify KafkaQueueBackend integration with mocked kafka components."""
    with patch("agent_utilities.knowledge_graph.core.kafka_queue_backend.json"):
        # Instantiate with fallback path to avoid parameter issues
        backend = KafkaQueueBackend(
            fallback_db_path="test_kafka_mock.db", bootstrap_servers="localhost:9092"
        )

        backend.put({"task": "kafka_task"})
        if backend._fallback_queue is not None:
            assert backend.get_queue_size() == 1
        else:
            assert backend.get_queue_size() == 0
