"""Regression coverage for fail-closed graph snapshot replacement."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


def test_from_json_propagates_failure_to_clear_existing_graph(monkeypatch):
    """A failed clear must not produce a mixed old/new graph snapshot."""
    engine = GraphComputeEngine.__new__(GraphComputeEngine)
    monkeypatch.setattr(engine, "_get_all_nodes", lambda: ["existing"])

    def fail_remove(_node_id: str) -> None:
        raise RuntimeError("clear failed")

    monkeypatch.setattr(engine, "remove_node", fail_remove)

    with pytest.raises(RuntimeError, match="clear failed"):
        engine.from_json('{"nodes": [{"id": "replacement"}], "edges": []}')
