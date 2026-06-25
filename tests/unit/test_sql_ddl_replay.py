"""Tests for replaying engine SQL-DDL ParseResults into the registry graph.

CONCEPT:KG-2.212 — the engine emits DatabaseTable/DatabaseColumn/DatabaseView nodes
plus hasColumn / referencesTable / referencesColumn / references edges; the parse
phase must map each node_type to its RegistryNodeType and pass every edge through.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.pipeline.phases.parse import _replay_parse_result
from agent_utilities.models.knowledge_graph import RegistryNodeType


class FakeGraph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props

    def add_edge(self, source, target, **props):
        self.edges.append((source, target, props.get("type")))


# Mirrors the Rust extractor's output for the tutorial schema.sql.
_RESULT = {
    "symbols_extracted": 5,
    "nodes": [
        {
            "node_id": "table:users",
            "node_type": "DatabaseTable",
            "properties": {"name": "users"},
        },
        {
            "node_id": "column:users.user_id",
            "node_type": "DatabaseColumn",
            "properties": {"name": "user_id", "primary_key": "true"},
        },
        {
            "node_id": "view:active_sessions",
            "node_type": "DatabaseView",
            "properties": {"name": "active_sessions"},
        },
        # A code SYMBOL still flows through unchanged.
        {
            "node_id": "symbol:abc",
            "node_type": "SYMBOL",
            "properties": {"name": "run", "symbol_type": "Function", "line": "10"},
        },
        # An unexpected node_type is ignored.
        {"node_id": "file:x", "node_type": "FILE", "properties": {}},
    ],
    "edges": [
        {
            "source": "table:users",
            "target": "column:users.user_id",
            "edge_type": "hasColumn",
            "properties": {},
        },
        {
            "source": "table:sessions",
            "target": "table:users",
            "edge_type": "referencesTable",
            "properties": {"confidence": "EXTRACTED"},
        },
        {
            "source": "view:active_sessions",
            "target": "table:users",
            "edge_type": "references",
            "properties": {},
        },
    ],
}


def test_replay_maps_db_node_types():
    g = FakeGraph()
    count = _replay_parse_result(_RESULT, g, RegistryNodeType)
    assert count == 5
    assert g.nodes["table:users"]["type"] == RegistryNodeType.DATABASE_TABLE
    assert g.nodes["column:users.user_id"]["type"] == RegistryNodeType.DATABASE_COLUMN
    assert g.nodes["column:users.user_id"]["primary_key"] == "true"
    assert g.nodes["view:active_sessions"]["type"] == RegistryNodeType.DATABASE_VIEW
    # SYMBOL path preserved (line coerced to int).
    assert g.nodes["symbol:abc"]["type"] == RegistryNodeType.SYMBOL
    assert g.nodes["symbol:abc"]["line"] == 10
    # FILE ignored.
    assert "file:x" not in g.nodes


def test_replay_passes_all_edges():
    g = FakeGraph()
    _replay_parse_result(_RESULT, g, RegistryNodeType)
    assert ("table:users", "column:users.user_id", "hasColumn") in g.edges
    assert ("table:sessions", "table:users", "referencesTable") in g.edges
    assert ("view:active_sessions", "table:users", "references") in g.edges
