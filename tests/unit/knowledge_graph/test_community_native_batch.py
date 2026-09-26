"""Native community persistence preserves the legacy IDs and relationship shape."""

import pytest

from agent_utilities.knowledge_graph.core.topological_analysis_engine import (
    TopologicalAnalysisEngine,
)
from agent_utilities.knowledge_graph.core.topological_partition import (
    persist_stable_communities,
)


class _Graph:
    def community_detection(self):
        return [["a", "b", "c"], ["singleton"], ["d", "e", "f"]]

    def _get_all_edges(self):
        return [("a", "b"), ("b", "c"), ("a", "c"), ("d", "e")]


class _Engine:
    def __init__(self, accepted=True):
        self.graph = _Graph()
        self.accepted = accepted
        self.batches = []
        self.legacy_writes = []

    def batch_typed_mutations(self, mutations, *, upsert, edge_upsert_scope):
        self.batches.append((mutations, upsert, edge_upsert_scope))
        return self.accepted

    def upsert_node(self, node):
        self.legacy_writes.append(node)

    def upsert_edge(self, edge):
        self.legacy_writes.append(edge)


def test_native_batch_preserves_ids_properties_and_member_direction():
    engine = _Engine()
    assert persist_stable_communities(engine, native_batch=True) == 2
    assert engine.legacy_writes == []
    assert len(engine.batches) == 1
    operations, upsert, scope = engine.batches[0]
    assert upsert is True
    assert scope == "relationship"
    nodes = [row for row in operations if row["kind"] == "node"]
    edges = [row for row in operations if row["kind"] == "edge"]
    assert [row["id"] for row in nodes] == [
        "community_cluster_0",
        "community_cluster_1",
    ]
    assert all(row["node_type"] == "community" for row in nodes)
    assert [row["properties"]["member_count"] for row in nodes] == [3, 3]
    assert all(row["properties"]["is_permanent"] for row in nodes)
    assert [row["properties"]["coherence_score"] for row in nodes] == [1.0, 1 / 3]
    assert all("type" not in row["properties"] for row in nodes)
    assert [(row["source"], row["target"]) for row in edges] == [
        ("a", "community_cluster_0"),
        ("b", "community_cluster_0"),
        ("c", "community_cluster_0"),
        ("d", "community_cluster_1"),
        ("e", "community_cluster_1"),
        ("f", "community_cluster_1"),
    ]
    assert all(row["rel_type"] == "part_of_community" for row in edges)
    assert all(row["properties"]["weight"] in (1.0, 1 / 3) for row in edges)

    legacy = _Engine()
    assert persist_stable_communities(legacy) == 2
    legacy_nodes = [row for row in legacy.legacy_writes if hasattr(row, "member_count")]
    legacy_edges = [row for row in legacy.legacy_writes if hasattr(row, "weight")]
    assert [row.id for row in legacy_nodes] == [row["id"] for row in nodes]
    assert [
        row.model_dump(mode="json", exclude={"id", "type"}, exclude_none=True)
        for row in legacy_nodes
    ] == [row["properties"] for row in nodes]
    assert sorted(
        (row.source, row.target, row.type.value, row.weight) for row in legacy_edges
    ) == sorted(
        (
            row["source"],
            row["target"],
            row["rel_type"],
            row["properties"]["weight"],
        )
        for row in edges
    )


def test_native_batch_failure_does_not_fall_back_to_legacy_writes():
    engine = _Engine(accepted=False)
    with pytest.raises(RuntimeError, match="not accepted"):
        persist_stable_communities(engine, native_batch=True)
    assert engine.legacy_writes == []


def test_topology_facade_exposes_explicit_native_batch_opt_in():
    engine = _Engine()
    facade = TopologicalAnalysisEngine(engine.graph)
    assert facade.persist_stable_communities(engine, native_batch=True) == 2
    assert len(engine.batches) == 1
