import time
from unittest.mock import MagicMock

from agent_utilities.knowledge_graph.memory import EvolvingMemoryAPI


def test_store_new_memory():
    engine_mock = MagicMock()
    memory_api = EvolvingMemoryAPI(engine=engine_mock)

    node_id = memory_api.store_new_memory(
        "user_123", "User likes Python", confidence=0.9
    )
    assert node_id.startswith("fact_user_123_")

    # Check that add_node and link_nodes were called
    assert engine_mock.add_node.call_count == 2
    engine_mock.link_nodes.assert_called_once()

    args, kwargs = engine_mock.link_nodes.call_args
    assert kwargs["source_id"] == "user_123"
    assert kwargs["target_id"] == node_id
    assert kwargs["rel_type"] == "HAS_FACT"
    assert kwargs["properties"]["confidence"] == 0.9
    assert kwargs["properties"]["valid_from"] <= int(time.time())


def test_retrieve_personalized_context_with_decay():
    engine_mock = MagicMock()
    memory_api = EvolvingMemoryAPI(engine=engine_mock)

    now = int(time.time())

    # Mock search_hybrid to return one valid fact and one decayed fact
    engine_mock.search_hybrid.return_value = [
        {
            "id": "fact_1",
            "user_id": "user_123",
            "content": "Valid fact",
            "valid_until": now + 10000,
        },
        {
            "id": "fact_2",
            "user_id": "user_123",
            "content": "Decayed fact",
            "valid_until": now - 10000,
        },
        {
            "id": "fact_3",
            "user_id": "other_user",
            "content": "Other user's fact",
            "valid_until": now + 10000,
        },
    ]

    # Mock get_blast_radius to return some neighbors
    engine_mock.get_blast_radius.return_value = [
        {"id": "neighbor_1", "content": "Related node"}
    ]

    results = memory_api.retrieve_personalized_context(
        "user_123", "query", top_k=5, max_hops=1
    )

    # Should only contain fact_1 and neighbor_1
    result_ids = {r["id"] for r in results}
    assert "fact_1" in result_ids
    assert "neighbor_1" in result_ids
    assert "fact_2" not in result_ids
    assert "fact_3" not in result_ids


def test_retrieve_fallback_cypher():
    engine_mock = MagicMock()
    memory_api = EvolvingMemoryAPI(engine=engine_mock)

    engine_mock.search_hybrid.side_effect = Exception("Semantic search failure")
    now = int(time.time())
    engine_mock.query_cypher.return_value = [
        {
            "id": "fact_fb",
            "user_id": "user_123",
            "content": "Fallback fact",
            "valid_until": now + 10000,
        }
    ]
    engine_mock.get_blast_radius.return_value = []

    results = memory_api.retrieve_personalized_context("user_123", "query")

    assert len(results) == 1
    assert results[0]["id"] == "fact_fb"
