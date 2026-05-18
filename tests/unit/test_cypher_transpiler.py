"""Unit tests for the Cypher-to-SQL transpiler."""

from typing import Any

import pytest

from agent_utilities.knowledge_graph.backends.cypher_transpiler import (
    QueryType,
    transpile,
)

KNOWN_TABLES = {"Agent", "Tool", "Memory", "Code", "Skill", "Article"}


class TestCreateNode:
    def test_basic_create(self):
        cypher = "CREATE (n:Agent {id: $id, name: $name, type: $type})"
        params = {"id": "agent-1", "name": "Test", "type": "agent"}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type == QueryType.INSERT
        assert tq.target_table == "Agent"
        assert '"Agent"' in tq.sql
        assert "INSERT INTO" in tq.sql
        assert "ON CONFLICT" in tq.sql
        assert tq.params == ["agent-1", "Test", "agent"]


class TestMatchById:
    def test_match_with_set(self):
        cypher = "MATCH (n:Tool) WHERE n.id = $id SET n.name = $name RETURN n.id"
        params = {"id": "tool-1", "name": "Updated"}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type == QueryType.UPDATE
        assert tq.target_table == "Tool"
        assert "UPDATE" in tq.sql
        assert "Updated" in tq.params
        assert "tool-1" in tq.params

    def test_match_return_only(self):
        cypher = "MATCH (n:Memory) WHERE n.id = $id RETURN n"
        params = {"id": "mem-1"}
        tq = transpile(cypher, params, KNOWN_TABLES)
        # Should be a SELECT since there's no SET
        assert tq.query_type == QueryType.SELECT


class TestLabelLookup:
    def test_label_function(self):
        cypher = "MATCH (n) WHERE n.id = $id RETURN label(n) as lbl"
        params = {"id": "some-id"}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type == QueryType.LABEL_LOOKUP
        assert "UNION ALL" in tq.sql
        assert "lbl" in tq.sql


class TestContainsSearch:
    def test_tolower_contains(self):
        cypher = (
            "MATCH (n) WHERE (toLower(n.name) CONTAINS toLower($k0)) "
            "AND coalesce(n.status, '') <> 'ARCHIVED' RETURN n"
        )
        params = {"k0": "search_term"}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type == QueryType.SELECT
        assert "LIKE" in tq.sql


class TestMergeRelationship:
    def test_merge_edge(self):
        cypher = (
            "MATCH (s:Agent {id: $sid}), (t:Tool {id: $tid}) "
            "MERGE (s)-[r:PROVIDES]->(t)"
        )
        params = {"sid": "agent-1", "tid": "tool-1"}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type == QueryType.UPSERT_EDGE
        assert "kg_edges" in tq.sql
        assert "ON CONFLICT" in tq.sql


class TestCountQuery:
    def test_count_pattern(self):
        cypher = "MATCH (t:Task {status: 'pending'}) RETURN count(t) as c"
        params: dict[str, Any] = {}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type in (QueryType.COUNT, QueryType.SELECT)


class TestDeleteQuery:
    def test_detach_delete(self):
        cypher = "MATCH (n:Memory) WHERE n.id = $id DETACH DELETE n"
        params = {"id": "mem-1"}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type == QueryType.DELETE
        assert "DELETE" in tq.sql


class TestLimitAndOrder:
    def test_limit_with_order(self):
        cypher = "MATCH (n:Agent) RETURN n ORDER BY n.name LIMIT 10"
        params: dict[str, Any] = {}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert tq.query_type == QueryType.SELECT
        assert "ORDER BY" in tq.sql
        assert "LIMIT" in tq.sql


class TestUnknownPattern:
    def test_unsupported_cypher(self):
        cypher = "CALL db.schema.visualization()"
        tq = transpile(cypher, {}, KNOWN_TABLES)
        assert tq.query_type == QueryType.UNKNOWN


class TestEdgeCases:
    def test_empty_params(self):
        cypher = "MATCH (n:Agent) RETURN n"
        tq = transpile(cypher, {}, KNOWN_TABLES)
        assert tq.query_type == QueryType.SELECT

    def test_none_params(self):
        cypher = "MATCH (n:Agent) RETURN n"
        tq = transpile(cypher, None, KNOWN_TABLES)
        assert tq.query_type == QueryType.SELECT

    def test_internal_params_stripped(self):
        cypher = "MATCH (n:Agent) WHERE n.id = $id RETURN n"
        params = {"id": "a1", "_clearance_level": 999}
        tq = transpile(cypher, params, KNOWN_TABLES)
        assert "_clearance_level" not in str(tq.params)
