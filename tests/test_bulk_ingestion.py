"""CONCEPT:KG-2.0"""

"""Tests for the Enterprise Hub-and-Spoke ingestion and batch writing capabilities."""

from typing import Any

import networkx as nx

from agent_utilities.knowledge_graph.backends.base import GraphBackend
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine


class MockBackend(GraphBackend):
    def __init__(self):
        self.queries = []
        self.batches = []

    def execute(self, query: str, params: dict | None = None) -> list[dict]:
        self.queries.append((query, params))
        return []

    def execute_batch(
        self, query: str, batch_params: list[dict]
    ) -> list[dict[str, Any]]:
        self.batches.append((query, batch_params))
        return []

    def add_embedding(self, *args, **kwargs):
        pass

    def create_schema(self, *args, **kwargs):
        pass

    def prune(self, *args, **kwargs):
        pass

    def semantic_search(self, *args, **kwargs):
        return []

    def close(self):
        pass


def test_ingest_external_batch():
    graph = nx.MultiDiGraph()
    engine = IntelligenceGraphEngine(graph=graph)
    engine.backend = MockBackend()

    entities = [
        {
            "id": "user:1",
            "name": "Alice",
            "type": "Employee",
            "department": "Engineering",
        },
        {"id": "user:2", "name": "Bob", "type": "Employee", "department": "Sales"},
    ]

    relationships = [
        {
            "source": "user:1",
            "target": "user:2",
            "type": "WORKS_WITH",
            "properties": {"since": "2023"},
        }
    ]

    result = engine.ingest_external_batch("active_directory", entities, relationships)

    assert result["status"] == "success"
    assert result["nodes"] == 2
    assert result["backend"] is True

    # Verify the backend received the UNWIND batch execution
    mock_backend = engine.backend
    assert len(mock_backend.batches) == 2

    node_query, node_params = mock_backend.batches[0]
    assert "UNWIND $batch AS row" in node_query
    assert "MERGE (n:DomainEntity {id: row.id})" in node_query
    assert len(node_params) == 2
    assert node_params[0]["id"] == "user:1"

    rel_query, rel_params = mock_backend.batches[1]
    assert "UNWIND $batch AS row" in rel_query
    assert "MERGE (s)-[r:EXTERNAL_LINK {type: row.type}]->(t)" in rel_query
    assert len(rel_params) == 1
    assert rel_params[0]["source"] == "user:1"


def test_ingest_external_batch_fallback():
    # If backend is None, it should fall back to NetworkX loop
    graph = nx.MultiDiGraph()
    engine = IntelligenceGraphEngine(graph=graph)
    engine.backend = None

    entities = [
        {
            "id": "user:1",
            "name": "Alice",
            "type": "Employee",
            "department": "Engineering",
        }
    ]

    result = engine.ingest_external_batch("active_directory", entities)

    assert result["status"] == "success"
    assert result["backend"] is False

    # Check NetworkX mirror
    assert "user:1" in engine.graph.nodes
    assert engine.graph.nodes["user:1"]["name"] == "Alice"
